"""Local, device-configurable candidate-extraction stage for the live-transcription benchmark.

Runs one GLiNER checkpoint (`urchade/gliner_medium-v2.1`) against a bounded trailing window of
the transcript-so-far (`config.GLINER_ZERO_SHOT_WINDOW_CHARS`) on every tick, stateless, covering
all 11 fields including `caller_name`/`email`/`phone_number`. A separate streaming PII checkpoint
was tried for those three fields (session-based, scoped to just the caller's current turn) but
this checkpoint's "person" label lost essentially all detection confidence given any prior
conversational turn as input -- a failure mode this model does not share, and windowed context
actively helps it rather than hurting it. `extract_candidates` is the single entry point the jev
resolver stage consumes.
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

ZERO_SHOT_MODEL_NAME = "urchade/gliner_medium-v2.1"

# Label-description mapping for all 11 target fields, handed to GLiNER on every tick alongside a
# bounded trailing window of the transcript-so-far. GLiNER prompts the model with each value (the
# descriptive text) but reports the corresponding key (this project's field name) back in each
# entity's `label`.
ZERO_SHOT_FIELD_LABELS: dict[str, str] = {
    "caller_name": "caller's full name",
    "email": "email address",
    "phone_number": "phone number",
    "unit_number": "apartment unit number",
    "amenities_requested": "requested apartment amenity",
    "pet_info": "pet type, breed, or description",
    "permission_to_enter": "statement about permission to enter the unit",
    "work_order_issue": "maintenance or work order issue description",
    "move_in_date": "move-in date or lease date",
    "price_quoted": "quoted rent price or dollar amount",
    "budget_amount": "budget or price range the caller can afford",
}

# Identity mapping: since `ZERO_SHOT_FIELD_LABELS` is passed to GLiNER as a label-description dict,
# `entity["label"]` already comes back as the field name itself. Kept as an explicit dict, rather
# than skipping the lookup entirely, so `_entities_to_candidates` has one shape regardless of how
# its `label_to_field` argument was built.
_ZERO_SHOT_LABEL_TO_FIELD = {field: field for field in ZERO_SHOT_FIELD_LABELS}

# Module-level singleton, lazily loaded on first use and reused for the rest of the process's
# lifetime. Lazy (rather than eager at import time) so that importing this module in tests doesn't
# require a real model download.
_zero_shot_model: GLiNER | None = None
_singleton_load_lock = threading.Lock()

# Guards `model.predict_entities(...)` below. GLiNER's stateless inference has no internal lock of
# its own -- only `torch.no_grad()`, no serialization -- so without this, concurrent ticks for
# different calls (once call_concurrency/gliner_concurrency are ever raised above their current
# default of 1) would race on the shared `_zero_shot_model` singleton with nothing guarding it on
# either side.
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


def _zero_shot_window(transcript_snapshot: str) -> tuple[str, int]:
    """Return the trailing slice of `transcript_snapshot` fed to the model this tick -- up to
    `config.GLINER_ZERO_SHOT_WINDOW_CHARS` characters -- along with that slice's start offset
    within the full snapshot, so callers can translate entity spans the model reports (relative to
    the slice) back into full-snapshot character offsets.

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
    """Re-encode a bounded trailing window of `transcript_snapshot` against all 11 field labels,
    instead of the full growing transcript, so latency stays flat regardless of call length.
    Candidate spans are translated back to full-snapshot character offsets before being returned,
    since every downstream consumer (jev's context window) indexes into the full snapshot, not the
    windowed slice actually fed to the model.
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


async def extract_candidates_timed(
    transcript_snapshot: str, call_id: str
) -> tuple[dict[str, list[dict]], float]:
    """Same as `extract_candidates`, but also returns the model's own latency in milliseconds as
    `(candidates, latency_ms)`.
    """
    start = time.monotonic()
    candidates = await asyncio.to_thread(_run_zero_shot_tick, transcript_snapshot)
    return candidates, (time.monotonic() - start) * 1000


async def extract_candidates(transcript_snapshot: str, call_id: str) -> dict[str, list[dict]]:
    """Return per-field candidate spans for one call's transcript-so-far.

    Re-encodes a bounded trailing window of the transcript-so-far against all 11 fields. Each
    field maps to a list of candidate spans shaped `{"text", "score", "start", "end"}`, empty when
    no candidate was found this tick. Every span's offsets are in full-transcript coordinates.

    A thin wrapper over `extract_candidates_timed` for callers that only need the candidates, not
    the model's latency.
    """
    candidates, _latency_ms = await extract_candidates_timed(transcript_snapshot, call_id)
    return candidates


def reset_call(call_id: str) -> None:
    """No-op: the model is called statelessly per tick, so there is no per-call state left to
    discard once a call ends. Kept as a callable, rather than removed, so callers (call teardown
    in `pipeline_core`, `batch_runner`'s warm-up, `scripts/smoke_gliner.py`) don't need to know
    whether any extraction strategy in use ever needs one.
    """
