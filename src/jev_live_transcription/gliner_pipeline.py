"""Local, device-configurable candidate-extraction stage for the live-transcription benchmark.

Runs two GLiNER models against the transcript-so-far on every tick, both stateless: a PII
checkpoint (`knowledgator/gliner-stream-pii-v1.0`) scoped to just the caller's current turn for
`caller_name`/`email`/`phone_number` (this checkpoint's "person" label loses essentially all
confidence once any prior conversational turn is present in its input, confirmed empirically
against this project's own corpus -- so it is deliberately given the least context that still
contains an entity, not the most), and a standard zero-shot checkpoint
(`urchade/gliner_medium-v2.1`) that re-encodes a bounded trailing window of the transcript-so-far
(`config.GLINER_ZERO_SHOT_WINDOW_CHARS`) for the remaining 8 domain fields, keeping its latency
flat regardless of call length. `extract_candidates` is the single entry point the jev resolver
stage consumes.
"""

from __future__ import annotations

import asyncio
import logging
import threading
import time

import torch
from gliner import GLiNER

from . import config

_LOGGER = logging.getLogger(__name__)

PII_MODEL_NAME = "knowledgator/gliner-stream-pii-v1.0"
ZERO_SHOT_MODEL_NAME = "urchade/gliner_medium-v2.1"

# Native labels the PII checkpoint was trained on, mapped to this project's field names.
PII_FIELD_LABELS: dict[str, str] = {
    "caller_name": "person",
    "email": "email address",
    "phone_number": "phone number",
}

# Zero-shot labels for the remaining domain fields, handed to the standard
# GLiNER checkpoint on every tick alongside the full transcript-so-far.
ZERO_SHOT_FIELD_LABELS: dict[str, str] = {
    "unit_number": "apartment unit number",
    "amenities_requested": "requested apartment amenity",
    "pet_info": "pet type, breed, or description",
    "permission_to_enter": "statement about permission to enter the unit",
    "work_order_issue": "maintenance or work order issue description",
    "move_in_date": "move-in date or lease date",
    "price_quoted": "quoted rent price or dollar amount",
    "budget_amount": "budget or price range the caller can afford",
}

_PII_LABEL_TO_FIELD = {label: field for field, label in PII_FIELD_LABELS.items()}
# Identity mapping: `ZERO_SHOT_FIELD_LABELS` is passed to GLiNER as a label-description dict (see
# `_run_zero_shot_tick`), so `entity["label"]` already comes back as the field name itself. Kept as
# an explicit dict, rather than skipping the lookup entirely, so `_entities_to_candidates` can share
# the same signature across both the PII and zero-shot call sites.
_ZERO_SHOT_LABEL_TO_FIELD = {field: field for field in ZERO_SHOT_FIELD_LABELS}

# Module-level singletons, lazily loaded on first use and reused for the rest
# of the process's lifetime. Lazy (rather than eager at import time) so that
# importing this module in tests doesn't require a real model download.
_pii_model: GLiNER | None = None
_zero_shot_model: GLiNER | None = None
# Guards the lazy singleton loads above: both models are used from worker
# threads (via asyncio.to_thread), so two calls' first ticks can race here
# without a lock, each loading its own model instance and silently dropping
# whichever one loses the assignment.
_singleton_load_lock = threading.Lock()

# Guards `model.predict_entities(...)` for the PII model below. GLiNER's stateless inference has
# no internal lock of its own -- only `torch.no_grad()`, no serialization -- so without this,
# concurrent PII ticks for different calls (once call_concurrency/gliner_concurrency are ever
# raised above their current default of 1) would race on the shared `_pii_model` singleton with
# nothing guarding it on either side. This is the same reasoning as `_zero_shot_inference_lock`
# below, for the same kind of call.
_pii_inference_lock = threading.Lock()

# Guards `model.predict_entities(...)` for the zero-shot model below, for the same reason as
# `_pii_inference_lock` above.
_zero_shot_inference_lock = threading.Lock()


def _resolve_device() -> str:
    """Resolve `config.GLINER_DEVICE` to a concrete `"cpu"`/`"cuda"` string.

    GLiNER's own `from_pretrained(map_location=...)` defaults unconditionally to `"cpu"` -- it
    does no autodetection, and passing `"cuda"` with no CUDA device present raises a
    `RuntimeError` rather than falling back -- so `"auto"` must be resolved to a concrete value
    here before ever calling `from_pretrained`. An explicit `"cpu"`/`"cuda"` override passes
    through unchanged.
    """
    if config.GLINER_DEVICE == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return config.GLINER_DEVICE


def _get_pii_model() -> GLiNER:
    global _pii_model
    if _pii_model is None:
        with _singleton_load_lock:
            if _pii_model is None:
                device = _resolve_device()
                _LOGGER.info("Loading %s onto device=%s", PII_MODEL_NAME, device)
                _pii_model = GLiNER.from_pretrained(PII_MODEL_NAME, map_location=device)
    return _pii_model


def _get_zero_shot_model() -> GLiNER:
    global _zero_shot_model
    if _zero_shot_model is None:
        with _singleton_load_lock:
            if _zero_shot_model is None:
                device = _resolve_device()
                _LOGGER.info("Loading %s onto device=%s", ZERO_SHOT_MODEL_NAME, device)
                _zero_shot_model = GLiNER.from_pretrained(ZERO_SHOT_MODEL_NAME, map_location=device)
    return _zero_shot_model


def _entities_to_candidates(
    entities: list[dict], label_to_field: dict[str, str]
) -> dict[str, list[dict]]:
    """Group raw GLiNER entity dicts into `{field: [candidate, ...]}` buckets."""
    candidates: dict[str, list[dict]] = {field: [] for field in label_to_field.values()}
    for entity in entities:
        field = label_to_field.get(entity.get("label"))
        if field is None:
            continue
        candidates[field].append(
            {
                "text": entity.get("text"),
                "score": entity.get("score"),
                "start": entity.get("start"),
                "end": entity.get("end"),
            }
        )
    return candidates


_CALLER_TURN_PREFIX = "Caller: "


def _current_caller_turn(transcript_snapshot: str) -> tuple[str, int]:
    """Return `(text, start)` for the caller's current, possibly still-growing turn -- the last
    line of `transcript_snapshot` if it's a caller turn, along with that text's start offset
    within the full snapshot (mirroring `_zero_shot_window`'s return shape, for the same reason:
    so callers can translate entity spans back to full-snapshot coordinates).

    Returns `("", 0)` when the last line isn't a caller turn (the agent is currently speaking) or
    the snapshot is empty. Only the current turn is used -- not the full transcript-so-far, and
    not a raw trailing character window -- because the PII checkpoint's "person" label loses
    essentially all confidence the moment ANY prior turn (even the caller's own previous turn) is
    present in its input, confirmed empirically against this project's real corpus. `email`/
    `phone_number` do not share this failure mode -- they score confidently on minimal context
    too -- so the same single window is used for all three PII labels rather than giving each its
    own.
    """
    line_start = transcript_snapshot.rfind("\n") + 1
    line = transcript_snapshot[line_start:]
    if not line.startswith(_CALLER_TURN_PREFIX):
        return "", 0
    text_start = line_start + len(_CALLER_TURN_PREFIX)
    return transcript_snapshot[text_start:], text_start


def _run_pii_tick(transcript_snapshot: str) -> dict[str, list[dict]]:
    """Extract PII candidates from the caller's current turn only (see `_current_caller_turn`).

    A name split across two of the caller's own turns (e.g. first name in one turn, last name
    given after an agent question in the next) is not reconstructed here -- concatenating the
    caller's own turns collapses the model's confidence just as much as including the agent's
    turn does, so this deliberately doesn't try. `pipeline_core`'s jev resolution stage builds its
    own context window around whatever candidate this does find (a sentence before and after, by
    full-snapshot position -- which the offset translation below makes correct), so jev's own
    reasoning gets a real chance to stitch a split name back together even though GLiNER itself
    cannot.
    """
    turn_text, turn_start = _current_caller_turn(transcript_snapshot)
    if not turn_text:
        return {field: [] for field in PII_FIELD_LABELS}
    model = _get_pii_model()
    with _pii_inference_lock:
        entities = model.predict_entities(turn_text, list(PII_FIELD_LABELS.values()), threshold=0.5)
    candidates = _entities_to_candidates(entities, _PII_LABEL_TO_FIELD)
    if turn_start:
        for spans in candidates.values():
            for span in spans:
                if span["start"] is not None:
                    span["start"] += turn_start
                if span["end"] is not None:
                    span["end"] += turn_start
    return candidates


def _zero_shot_window(transcript_snapshot: str) -> tuple[str, int]:
    """Return the trailing slice of `transcript_snapshot` fed to the standard zero-shot model this
    tick -- up to `config.GLINER_ZERO_SHOT_WINDOW_CHARS` characters -- along with that slice's
    start offset within the full snapshot, so callers can translate entity spans the model reports
    (relative to the slice) back into full-snapshot character offsets.

    The raw character cut is advanced to the next word boundary rather than used as-is, so the
    window never starts mid-word -- a hard cut could otherwise split the very entity the window
    exists to still capture (e.g. turning "apartment 204" into "...rtment 204", which GLiNER may
    fail to recognize as a unit number). This can only shrink the window below the configured
    size, never grow it past it.
    """
    raw_start = max(0, len(transcript_snapshot) - config.GLINER_ZERO_SHOT_WINDOW_CHARS)
    window_start = raw_start
    if raw_start > 0:
        boundary = transcript_snapshot.find(" ", raw_start)
        if boundary != -1:
            window_start = boundary + 1
    return transcript_snapshot[window_start:], window_start


def _run_zero_shot_tick(transcript_snapshot: str) -> dict[str, list[dict]]:
    """Re-encode a bounded trailing window of `transcript_snapshot` against the zero-shot domain
    labels, instead of the full growing transcript, so latency stays flat regardless of call
    length. Candidate spans are translated back to full-snapshot character offsets before being
    returned, since every downstream consumer (jev's context window) indexes into the full
    snapshot, not the windowed slice actually fed to the model.

    `ZERO_SHOT_FIELD_LABELS` is passed directly as a label-description mapping: GLiNER prompts the
    model with each value (the descriptive text) but reports the corresponding key (this project's
    field name) back in each entity's `label`, so the lookup passed to `_entities_to_candidates`
    below (`_ZERO_SHOT_LABEL_TO_FIELD`) is a trivial identity mapping, kept only so both call sites
    share the same helper signature.
    """
    model = _get_zero_shot_model()
    window_text, window_start = _zero_shot_window(transcript_snapshot)
    with _zero_shot_inference_lock:
        entities = model.predict_entities(
            window_text,
            ZERO_SHOT_FIELD_LABELS,
            multi_label=True,
            threshold=config.GLINER_ZERO_SHOT_THRESHOLD,
        )
    candidates = _entities_to_candidates(entities, _ZERO_SHOT_LABEL_TO_FIELD)
    if window_start:
        for spans in candidates.values():
            for span in spans:
                if span["start"] is not None:
                    span["start"] += window_start
                if span["end"] is not None:
                    span["end"] += window_start
    return candidates


async def _timed_to_thread(func, *args):
    """Run `func` in a worker thread and return `(result, latency_ms)` for that call alone.

    Timed around the individual `asyncio.to_thread` call rather than around a caller's outer
    `asyncio.gather` of several such calls, so each call's own latency is never conflated with a
    concurrently-running sibling's.
    """
    start = time.monotonic()
    result = await asyncio.to_thread(func, *args)
    return result, (time.monotonic() - start) * 1000


async def extract_candidates_timed(
    transcript_snapshot: str, call_id: str
) -> tuple[dict[str, list[dict]], float, float]:
    """Same as `extract_candidates`, but also returns each model's own latency in milliseconds as
    `(candidates, pii_latency_ms, zero_shot_latency_ms)` -- timed independently per model (see
    `_timed_to_thread`) so a caller recording per-stage latency (e.g. `pipeline_core`) never
    attributes one model's wall-clock time to the other.
    """
    pii_task = _timed_to_thread(_run_pii_tick, transcript_snapshot)
    zero_shot_task = _timed_to_thread(_run_zero_shot_tick, transcript_snapshot)
    (pii_candidates, pii_latency_ms), (zero_shot_candidates, zero_shot_latency_ms) = await asyncio.gather(
        pii_task, zero_shot_task
    )
    return {**pii_candidates, **zero_shot_candidates}, pii_latency_ms, zero_shot_latency_ms


async def extract_candidates(transcript_snapshot: str, call_id: str) -> dict[str, list[dict]]:
    """Return per-field candidate spans for one call's transcript-so-far.

    Runs the PII checkpoint (`caller_name`/`email`/`phone_number`, scoped to just the caller's
    current turn) and the zero-shot domain checkpoint (the other 8 fields, re-encoding only a
    bounded trailing window of the transcript-so-far) concurrently via `asyncio.to_thread`, so
    wall-clock latency is the max of the two rather than their sum. The result covers
    all 11 fields; each maps to a list of candidate spans shaped
    `{"text", "score", "start", "end"}`, empty when no candidate was found
    this tick. Every span's offsets are in full-transcript coordinates
    regardless of which model found it.

    A thin wrapper over `extract_candidates_timed` for callers that only need the merged
    candidates, not each model's individual latency.
    """
    candidates, _pii_latency_ms, _zero_shot_latency_ms = await extract_candidates_timed(
        transcript_snapshot, call_id
    )
    return candidates


def reset_call(call_id: str) -> None:
    """No-op: both GLiNER models are called statelessly per tick, so there is no per-call state
    left to discard once a call ends. Kept as a callable, rather than removed, so callers (call
    teardown in `pipeline_core`, `batch_runner`'s warm-up, `scripts/smoke_gliner.py`) don't need to
    know which extraction strategy is in use behind this module's API.
    """
