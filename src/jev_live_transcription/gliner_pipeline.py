"""Local, device-configurable candidate-extraction stage for the live-transcription benchmark.

Runs one GLiNER checkpoint (`urchade/gliner_medium-v2.1`) against a bounded trailing window of
the transcript-so-far (`config.GLINER_ZERO_SHOT_WINDOW_CHARS`) on every tick, stateless, covering
all 11 fields including `caller_name`/`email`/`phone_number`. `extract_candidates` is the single
entry point the jev resolver stage consumes.
"""

from __future__ import annotations

import logging
import threading
import time
from typing import Literal

import httpx
import torch
from gliner import GLiNER

from . import config
from .gliner_batch_engine import GlinerBatchEngine, get_or_create_engine

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

# Generalization mechanism for downstream, per-field behavior (thresholds, jev's commit path, the
# determination classifier): each field is classified once here rather than special-cased by name
# elsewhere.
#   - "span": a single free-text span (a name, number, date, or amount).
#   - "list_span": zero or more free-text spans that should all be kept (e.g. multiple amenities).
#   - "determination": a yes/no/unclear judgment rather than an extracted span.
#   - "multi_fact": a compound description that may bundle more than one distinct fact.
FieldTaxonomy = Literal["span", "list_span", "determination", "multi_fact"]

FIELD_TAXONOMY: dict[str, FieldTaxonomy] = {
    "caller_name": "span",
    "email": "span",
    "phone_number": "span",
    "unit_number": "span",
    "move_in_date": "span",
    "price_quoted": "span",
    "budget_amount": "span",
    "amenities_requested": "list_span",
    "pet_info": "list_span",
    "permission_to_enter": "determination",
    "work_order_issue": "multi_fact",
}

# Identity mapping: since `ZERO_SHOT_FIELD_LABELS` is passed to GLiNER as a label-description dict,
# `entity["label"]` already comes back as the field name itself. Kept as an explicit dict, rather
# than skipping the lookup entirely, so `_postprocess_entities` has one shape regardless of how
# its `label_to_field` argument was built.
_ZERO_SHOT_LABEL_TO_FIELD = {field: field for field in ZERO_SHOT_FIELD_LABELS}

# Module-level singleton, lazily loaded on first use and reused for the rest of the process's
# lifetime. Lazy (rather than eager at import time) so that importing this module in tests doesn't
# require a real model download.
_zero_shot_model: GLiNER | None = None
_singleton_load_lock = threading.Lock()

# Shared httpx client for GLINER_SERVING_MODE="http", lazily created on first use so importing
# this module never requires a running event loop.
_http_client: httpx.AsyncClient | None = None


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


def _postprocess_entities(
    entities: list[dict], window_start: int, label_to_field: dict[str, str]
) -> dict[str, list[dict]]:
    """Group raw GLiNER entity dicts into `{field: [candidate, ...]}` buckets and translate their
    spans (reported relative to the windowed slice actually fed to the model) back to
    full-snapshot character offsets, since every downstream consumer (jev's context window) indexes
    into the full snapshot, not the windowed slice.

    Shared by both `GLINER_SERVING_MODE` paths (inline batch engine, HTTP service) -- neither the
    engine's `submit()` nor the serving app's `/extract` response does this translation itself,
    since it depends on `window_start`, a caller-side concept.
    """
    candidates: dict[str, list[dict]] = {field: [] for field in label_to_field.values()}
    for entity in entities:
        field = label_to_field.get(entity.get("label"))
        if field is None:
            continue
        start = entity.get("start")
        end = entity.get("end")
        candidates[field].append(
            {
                "text": entity.get("text"),
                "score": entity.get("score"),
                "start": start + window_start if start is not None else None,
                "end": end + window_start if end is not None else None,
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


def _make_batch_engine(
    *, max_batch_size: int | None = None, batch_wait_timeout_ms: float | None = None
) -> GlinerBatchEngine:
    """Construct (but don't start) a `GlinerBatchEngine` against the shared model singleton.

    `max_batch_size`/`batch_wait_timeout_ms` default to `config`'s values when omitted -- the
    single construction path for both the inline engine cache below and the standalone HTTP
    serving app (`serving/app.py`), so the two `GLINER_SERVING_MODE` paths can't drift apart on how
    the engine gets built.
    """
    return GlinerBatchEngine(
        _get_zero_shot_model(),
        ZERO_SHOT_FIELD_LABELS,
        max_batch_size=max_batch_size if max_batch_size is not None else config.GLINER_BATCH_MAX_SIZE,
        batch_wait_timeout_ms=(
            batch_wait_timeout_ms
            if batch_wait_timeout_ms is not None
            else config.GLINER_BATCH_WAIT_TIMEOUT_MS
        ),
        threshold=config.GLINER_ZERO_SHOT_THRESHOLD,
        multi_label=True,
    )


async def _get_batch_engine() -> GlinerBatchEngine:
    """Return the batch engine bound to the current event loop, creating one if needed.

    Loop-keyed rather than a bare module-level singleton: the vendor engine this design is modeled
    on (`AsyncStreamingEngine`) raises if reused across event loops, and this repo's own test suite
    calls `asyncio.run(...)` once per test function -- a fresh loop every time -- so a bare
    singleton would break the second test that touched it.
    """
    return await get_or_create_engine(_make_batch_engine)


def _get_http_client() -> httpx.AsyncClient:
    global _http_client
    if _http_client is None:
        _http_client = httpx.AsyncClient(base_url=config.GLINER_SERVICE_URL, timeout=5.0)
    return _http_client


async def _infer_entities(window_text: str) -> list[dict]:
    """Return raw GLiNER entity dicts (offsets relative to `window_text`) for one windowed tick,
    dispatched via whichever `config.GLINER_SERVING_MODE` is selected.
    """
    if config.GLINER_SERVING_MODE == "http":
        response = await _get_http_client().post("/extract", json={"text": window_text})
        response.raise_for_status()
        return response.json()["entities"]
    engine = await _get_batch_engine()
    return await engine.submit(window_text)


async def extract_candidates_timed(
    transcript_snapshot: str, call_id: str
) -> tuple[dict[str, list[dict]], float]:
    """Same as `extract_candidates`, but also returns the model's own latency in milliseconds as
    `(candidates, latency_ms)`.

    `call_id` isn't used by extraction itself (the model call is stateless) -- kept so callers
    (`pipeline_core`, `batch_runner`, `scripts/smoke_gliner.py`) can pass the same per-call
    identity through every stage of the pipeline without a signature that varies by which
    extraction strategy is in use.
    """
    window_text, window_start = _zero_shot_window(transcript_snapshot)
    start = time.monotonic()
    entities = await _infer_entities(window_text)
    latency_ms = (time.monotonic() - start) * 1000
    candidates = _postprocess_entities(entities, window_start, _ZERO_SHOT_LABEL_TO_FIELD)
    return candidates, latency_ms


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
