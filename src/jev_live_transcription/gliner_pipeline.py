"""Local, CPU-only candidate-extraction stage for the live-transcription benchmark.

Runs two GLiNER models against the transcript-so-far on every tick: a streaming
PII checkpoint (`knowledgator/gliner-stream-pii-v1.0`) that reuses its decoder
KV cache incrementally per call for `caller_name`/`email`/`phone_number`, and a
standard zero-shot checkpoint (`urchade/gliner_medium-v2.1`) that re-encodes a
bounded trailing window of the transcript-so-far (`config.GLINER_ZERO_SHOT_WINDOW_CHARS`) for the
remaining 8 domain fields, keeping its latency flat regardless of call length.
`extract_candidates` is the single entry point the jev resolver stage
consumes.
"""

from __future__ import annotations

import asyncio
import threading
import time

from gliner import GLiNER

from . import config

PII_MODEL_NAME = "knowledgator/gliner-stream-pii-v1.0"
ZERO_SHOT_MODEL_NAME = "urchade/gliner_medium-v2.1"

# Native labels the streaming PII checkpoint was trained on, mapped to this
# project's field names.
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
_ZERO_SHOT_LABEL_TO_FIELD = {label: field for field, label in ZERO_SHOT_FIELD_LABELS.items()}

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

# Count of transcript characters already fed into the PII model's streaming
# session per call_id. Only the new suffix is sent each tick; the checkpoint's
# cached-incremental mode reuses decoder KV state, labels, words, and span
# history for everything already seen. Never shared across call_ids.
_pii_sent_length: dict[str, int] = {}
# Per-call_id lock serializing the read-delta/advance-session sequence in
# `_run_pii_tick`, so overlapping ticks for the same call can't race on
# `_pii_sent_length` or send overlapping chunks into the streaming session.
_pii_call_locks: dict[str, threading.Lock] = {}
_pii_call_locks_guard = threading.Lock()

# The streaming PII checkpoint keeps its own per-session KV-cache/decoder state inside the model
# object, keyed by session_id -- unlike a stateless forward pass, there is no upstream guarantee
# that calling `model.inference(...)` concurrently from multiple threads with *different*
# session_ids is safe against that shared internal state. The per-call_id lock above only
# serializes ticks for the *same* call; this lock serializes the actual inference call itself
# across every call_id, so batch_runner's call_concurrency/gliner_concurrency > 1 can never send
# two calls' inferences into the model at once.
_pii_inference_lock = threading.Lock()


def _get_pii_model() -> GLiNER:
    global _pii_model
    if _pii_model is None:
        with _singleton_load_lock:
            if _pii_model is None:
                _pii_model = GLiNER.from_pretrained(PII_MODEL_NAME)
    return _pii_model


def _get_zero_shot_model() -> GLiNER:
    global _zero_shot_model
    if _zero_shot_model is None:
        with _singleton_load_lock:
            if _zero_shot_model is None:
                _zero_shot_model = GLiNER.from_pretrained(ZERO_SHOT_MODEL_NAME)
    return _zero_shot_model


def _get_pii_call_lock(call_id: str) -> threading.Lock:
    with _pii_call_locks_guard:
        lock = _pii_call_locks.get(call_id)
        if lock is None:
            lock = threading.Lock()
            _pii_call_locks[call_id] = lock
        return lock


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


def _run_pii_tick(transcript_snapshot: str, call_id: str) -> dict[str, list[dict]]:
    """Advance `call_id`'s streaming PII session to `transcript_snapshot` and return its candidates."""
    model = _get_pii_model()
    with _get_pii_call_lock(call_id):
        sent = _pii_sent_length.get(call_id, 0)
        delta = transcript_snapshot[sent:]
        if not delta:
            # No new text this tick. The streaming API requires a non-empty
            # chunk per call, so there's nothing to feed the session; report
            # no candidates rather than re-querying stale state.
            return {field: [] for field in PII_FIELD_LABELS}
        with _pii_inference_lock:
            entities = model.inference(
                [delta],
                list(PII_FIELD_LABELS.values()),
                session_id=[call_id],
                threshold=0.5,
            )[0]
        _pii_sent_length[call_id] = len(transcript_snapshot)
    return _entities_to_candidates(entities, _PII_LABEL_TO_FIELD)


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
    """
    model = _get_zero_shot_model()
    window_text, window_start = _zero_shot_window(transcript_snapshot)
    entities = model.predict_entities(
        window_text,
        list(ZERO_SHOT_FIELD_LABELS.values()),
        multi_label=True,
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
    pii_task = _timed_to_thread(_run_pii_tick, transcript_snapshot, call_id)
    zero_shot_task = _timed_to_thread(_run_zero_shot_tick, transcript_snapshot)
    (pii_candidates, pii_latency_ms), (zero_shot_candidates, zero_shot_latency_ms) = await asyncio.gather(
        pii_task, zero_shot_task
    )
    return {**pii_candidates, **zero_shot_candidates}, pii_latency_ms, zero_shot_latency_ms


async def extract_candidates(transcript_snapshot: str, call_id: str) -> dict[str, list[dict]]:
    """Return per-field candidate spans for one call's transcript-so-far.

    Runs the streaming PII checkpoint (`caller_name`/`email`/`phone_number`,
    cached incrementally per `call_id`) and the zero-shot domain checkpoint
    (the other 8 fields, re-encoding only a bounded trailing window of the
    transcript-so-far) concurrently via `asyncio.to_thread`, so wall-clock
    latency is the max of the two rather than their sum. The result covers
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
    """Discard cached streaming state for `call_id`, e.g. once its call has ended."""
    _pii_sent_length.pop(call_id, None)
    with _pii_call_locks_guard:
        _pii_call_locks.pop(call_id, None)
    if _pii_model is not None:
        _pii_model.clear_session(call_id)
