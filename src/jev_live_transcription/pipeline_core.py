"""Orchestrates one call end to end: replays its transcript via the pacer and, on every tick the
transcript grows, runs the GLiNER+jev pipeline; the GPT-5.1 baseline, when `run_call`'s
`enable_llm_baseline` is opted in, runs concurrently alongside it on every
`config.LLM_CADENCE_TICKS`-th such tick. Every tick's activity is captured to the capture DB.

Two logical pipelines are recorded, named by the `pipeline` column on `pipeline_runs` and
`field_extractions`:

- `"gliner_jev"`: local GLiNER candidate extraction (stages `"gliner_standard"` and
  `"gliner_stream_pii"`) feeding the hosted jev resolver (stage `"jev"`).
- `"llm"`: the GPT-5.1 baseline (stage `"llm"`).

Both `gliner_pipeline.extract_candidates` and `jev_pipeline.JevFieldResolver.resolve_field`
report `is_committed` only for the single call just made; holding a commit steady across ticks
where a pipeline comes back under threshold (or doesn't run at all) is explicitly documented as
the calling orchestrator's job (see `jev_pipeline.JevResolution`), so that state is tracked here,
per (pipeline, field_name), and is never exposed to the extraction/resolution/baseline modules
themselves.
"""

from __future__ import annotations

import asyncio
import contextlib
import json
import logging
import re
import time
from pathlib import Path
from typing import Callable

from . import config, corpus, gliner_pipeline, llm_baseline
from .jev_pipeline import JevFieldResolver
from .pacer import CallPacer, iter_batch_ticks, iter_realtime_ticks
from . import db as db_module

_LOGGER = logging.getLogger(__name__)

GLINER_JEV_PIPELINE = "gliner_jev"
LLM_PIPELINE = "llm"

# Splits transcript text into sentence-like chunks on standard end-of-sentence punctuation. The
# transcript is plain "Speaker: text" lines rather than prose, so this is a heuristic, not a real
# sentence tokenizer -- good enough to bound a jev context window to the text around one candidate
# span instead of the whole growing transcript.
_SENTENCE_END_RE = re.compile(r"[.!?](?=\s|$)")

# Committed-value state per (pipeline, field_name), carried across ticks within one call. Reset
# per call by constructing a fresh dict in `run_call` -- never shared across calls.
_CommittedState = dict[tuple[str, str], tuple[str, float]]


def _sentence_spans(text: str) -> list[tuple[int, int]]:
    """Return non-overlapping `(start, end)` character spans covering every sentence in `text`."""
    spans = []
    start = 0
    for match in _SENTENCE_END_RE.finditer(text):
        end = match.end()
        spans.append((start, end))
        start = end
    if start < len(text):
        spans.append((start, len(text)))
    return spans


def context_window(
    snapshot: str, start: int, end: int, *, spans: list[tuple[int, int]] | None = None
) -> str:
    """Slice `snapshot` to the sentence(s) spanning `[start, end)` plus one sentence on each side.

    Falls back to the full snapshot when `start`/`end` don't land inside any sentence span (e.g.
    missing offsets from an upstream candidate). `spans` lets a caller that already computed
    `_sentence_spans(snapshot)` for this tick (e.g. once, shared across every field's candidates)
    pass it in instead of paying to re-tokenize the same snapshot again per field.
    """
    spans = _sentence_spans(snapshot) if spans is None else spans
    if not spans:
        return snapshot
    covering = [i for i, (s, e) in enumerate(spans) if e > start and s < end]
    if not covering:
        return snapshot
    lo = max(0, covering[0] - 1)
    hi = min(len(spans) - 1, covering[-1] + 1)
    window_start = spans[lo][0]
    window_end = spans[hi][1]
    return snapshot[window_start:window_end].strip()


def _field_context_window(
    snapshot: str, candidate_spans: list[dict], *, spans: list[tuple[int, int]] | None = None
) -> str:
    """Context window covering every candidate span found for one field on one tick.

    Only spans with both a `start` and an `end` contribute to the window -- a span missing just
    one of the pair is dropped instead of letting its lone coordinate pair up with another span's,
    which would stitch together a window spanning two unrelated candidates.
    """
    bounded = [c for c in candidate_spans if c.get("start") is not None and c.get("end") is not None]
    if not bounded:
        return snapshot
    return context_window(
        snapshot, min(c["start"] for c in bounded), max(c["end"] for c in bounded), spans=spans
    )


def _apply_carry_forward(
    committed: _CommittedState,
    pipeline: str,
    field_name: str,
    candidate_value: str | None,
    confidence: float,
    is_committed_now: bool,
) -> tuple[str | None, float, bool]:
    """Fold one tick's raw resolution into `committed` and return what to persist for this tick.

    A fresh commit supersedes whatever was held before. Otherwise, whatever was already committed
    for this (pipeline, field_name) holds steady -- a later call coming back under threshold, or a
    tick where the pipeline didn't run at all, must not revert scoring's view of "the pipeline's
    current answer" back to uncommitted.
    """
    key = (pipeline, field_name)
    if is_committed_now and candidate_value is not None:
        committed[key] = (candidate_value, confidence)
        return candidate_value, confidence, True
    held = committed.get(key)
    if held is not None:
        held_value, held_confidence = held
        return held_value, held_confidence, True
    return candidate_value, confidence, False


async def _persist_field(
    store,
    committed: _CommittedState,
    *,
    pipeline: str,
    call_id: int,
    tick_number: int,
    run_id: int,
    field_name: str,
    candidate_value: str | None,
    confidence: float,
    is_committed_now: bool,
) -> None:
    """Fold one field's raw resolution through `_apply_carry_forward` and persist the row.

    Shared by both pipelines' per-field loops below. Awaits the enqueued write's future (unlike a
    fire-and-forget put) so a caller who awaits the enclosing step -- and, transitively, `run_call`
    -- knows every `field_extractions` row for this tick has actually committed, not just been
    queued; this matters when `run_call` is given an externally-owned store it never closes (and
    so never blocks on draining) to wait for.
    """
    persisted_value, persisted_confidence, persisted_committed = _apply_carry_forward(
        committed, pipeline, field_name, candidate_value, confidence, is_committed_now
    )
    future = store.enqueue_field_extraction(
        run_id=run_id,
        call_id=call_id,
        tick_number=tick_number,
        pipeline=pipeline,
        field_name=field_name,
        candidate_value=persisted_value,
        confidence=persisted_confidence,
        is_committed=int(persisted_committed),
    )
    await asyncio.wrap_future(future)


def _estimate_jev_cost_usd(input_tokens: int | None, output_tokens: int | None) -> float | None:
    if input_tokens is None or output_tokens is None:
        return None
    pricing = config.PRICING_PER_MILLION_TOKENS["jev"]
    return (
        input_tokens * pricing["input"] / 1_000_000
        + output_tokens * pricing["output"] / 1_000_000
    )


async def _enqueue_pipeline_run(
    store,
    *,
    tick_id: int,
    call_id: int,
    pipeline: str,
    stage: str | None,
    latency_ms: float | None = None,
    input_tokens: int | None = None,
    cached_input_tokens: int | None = None,
    output_tokens: int | None = None,
    estimated_cost_usd: float | None = None,
    raw_output_json: str | None = None,
    error: str | None = None,
) -> int:
    """Enqueue a `pipeline_runs` row and await its committed `run_id`.

    Awaiting is required (not just fire-and-forget) because `field_extractions` rows below need
    the real, generated `run_id` to satisfy its foreign key -- there is no other way to learn it.
    """
    future = store.enqueue_pipeline_run(
        tick_id=tick_id,
        call_id=call_id,
        pipeline=pipeline,
        stage=stage,
        latency_ms=latency_ms,
        input_tokens=input_tokens,
        cached_input_tokens=cached_input_tokens,
        output_tokens=output_tokens,
        estimated_cost_usd=estimated_cost_usd,
        raw_output_json=raw_output_json,
        error=error,
    )
    return await asyncio.wrap_future(future)


async def _timed_resolve_field(coro):
    """Await one `resolver.resolve_field` call and return `(result_or_exception, latency_ms)`.

    Timed individually per field rather than around a caller's outer `asyncio.gather` of several
    such calls, so concurrently-resolved fields never share one wall-clock number. Never raises --
    an exception is captured and returned alongside its own latency instead, so a caller can
    `asyncio.gather` every field's call without `return_exceptions=True` losing per-field timing.
    """
    start = time.monotonic()
    try:
        result = await coro
    except Exception as exc:  # noqa: BLE001 -- surfaced to the caller as data, not raised here
        return exc, (time.monotonic() - start) * 1000
    return result, (time.monotonic() - start) * 1000


async def _run_gliner_jev_step(
    store,
    *,
    call_id: int,
    pipeline_call_id: str,
    tick_id: int,
    tick_number: int,
    snapshot: str,
    resolver: JevFieldResolver,
    committed: _CommittedState,
    gliner_semaphore: asyncio.Semaphore | None = None,
) -> None:
    # `start` is taken *inside* the semaphore, matching where extract_candidates_timed's own
    # internal per-model timers start -- taking it before acquiring the semaphore would fold
    # queueing wait (real under gliner_concurrency > 1) into the error path's latency_ms while the
    # success path's pii_latency_ms/zero_shot_latency_ms never include it, making the two
    # incomparable for the same stage.
    semaphore_ctx = gliner_semaphore if gliner_semaphore is not None else contextlib.nullcontext()
    try:
        async with semaphore_ctx:
            start = time.monotonic()
            candidates, pii_latency_ms, zero_shot_latency_ms = await gliner_pipeline.extract_candidates_timed(
                snapshot, pipeline_call_id
            )
    except Exception as exc:  # noqa: BLE001 -- GLiNER failures are captured as data, not raised
        # extract_candidates runs both underlying models concurrently and surfaces whichever one
        # raised first -- there is no way to tell from here whether the PII or zero-shot model (or
        # both) actually failed, so both stages get an error row rather than misattributing the
        # failure to just one.
        latency_ms = (time.monotonic() - start) * 1000
        await asyncio.gather(
            _enqueue_pipeline_run(
                store,
                tick_id=tick_id,
                call_id=call_id,
                pipeline=GLINER_JEV_PIPELINE,
                stage="gliner_stream_pii",
                latency_ms=latency_ms,
                error=str(exc),
            ),
            _enqueue_pipeline_run(
                store,
                tick_id=tick_id,
                call_id=call_id,
                pipeline=GLINER_JEV_PIPELINE,
                stage="gliner_standard",
                latency_ms=latency_ms,
                error=str(exc),
            ),
        )
        return

    pii_candidates = {field: candidates.get(field, []) for field in gliner_pipeline.PII_FIELD_LABELS}
    zero_shot_candidates = {
        field: candidates.get(field, []) for field in gliner_pipeline.ZERO_SHOT_FIELD_LABELS
    }
    await asyncio.gather(
        _enqueue_pipeline_run(
            store,
            tick_id=tick_id,
            call_id=call_id,
            pipeline=GLINER_JEV_PIPELINE,
            stage="gliner_stream_pii",
            latency_ms=pii_latency_ms,
            raw_output_json=json.dumps(pii_candidates),
        ),
        _enqueue_pipeline_run(
            store,
            tick_id=tick_id,
            call_id=call_id,
            pipeline=GLINER_JEV_PIPELINE,
            stage="gliner_standard",
            latency_ms=zero_shot_latency_ms,
            raw_output_json=json.dumps(zero_shot_candidates),
        ),
    )

    # Computed once per tick and shared across every field's context window below, instead of
    # every field re-tokenizing the same (potentially long, ever-growing) snapshot from scratch.
    sentence_spans = _sentence_spans(snapshot)

    resolve_tasks = []
    resolve_field_names = []
    for field_name, field_spans in candidates.items():
        values = [span["text"] for span in field_spans if span.get("text")]
        if not values:
            continue
        context = _field_context_window(snapshot, field_spans, spans=sentence_spans)
        resolve_tasks.append(
            _timed_resolve_field(resolver.resolve_field(pipeline_call_id, field_name, values, context))
        )
        resolve_field_names.append(field_name)
    if not resolve_tasks:
        return

    results_with_latency = await asyncio.gather(*resolve_tasks)
    persist_tasks = []
    for field_name, (result, jev_latency_ms) in zip(resolve_field_names, results_with_latency):
        if isinstance(result, Exception):
            await _enqueue_pipeline_run(
                store,
                tick_id=tick_id,
                call_id=call_id,
                pipeline=GLINER_JEV_PIPELINE,
                stage="jev",
                latency_ms=jev_latency_ms,
                error=str(result),
            )
            continue
        if result is None:
            continue
        if result.is_none_of_these and result.is_committed:
            # A confident "none of these" is jev actively rejecting every candidate seen so far
            # for this field, not merely "nothing new this tick" -- clear whatever was
            # previously held so the rejection isn't masked by carrying the stale value forward.
            committed.pop((GLINER_JEV_PIPELINE, field_name), None)
        run_id = await _enqueue_pipeline_run(
            store,
            tick_id=tick_id,
            call_id=call_id,
            pipeline=GLINER_JEV_PIPELINE,
            stage="jev",
            latency_ms=jev_latency_ms,
            input_tokens=result.input_tokens,
            output_tokens=result.output_tokens,
            estimated_cost_usd=_estimate_jev_cost_usd(result.input_tokens, result.output_tokens),
            raw_output_json=json.dumps(
                {
                    "question_type": result.question_type,
                    "candidate": result.candidate,
                    "confidence": result.confidence,
                    "is_none_of_these": result.is_none_of_these,
                    "distinct_candidates": list(result.distinct_candidates),
                }
            ),
        )
        persist_tasks.append(
            _persist_field(
                store,
                committed,
                pipeline=GLINER_JEV_PIPELINE,
                call_id=call_id,
                tick_number=tick_number,
                run_id=run_id,
                field_name=field_name,
                candidate_value=result.candidate if not result.is_none_of_these else None,
                confidence=result.confidence,
                is_committed_now=result.is_committed and not result.is_none_of_these,
            )
        )
    if persist_tasks:
        await asyncio.gather(*persist_tasks)


async def _run_llm_step(
    store,
    *,
    call_id: int,
    pipeline_call_id: str,
    tick_id: int,
    tick_number: int,
    snapshot: str,
    committed: _CommittedState,
) -> None:
    start = time.monotonic()
    try:
        result = await llm_baseline.extract(pipeline_call_id, snapshot)
    except Exception as exc:  # noqa: BLE001 -- LLM failures are captured as data, not raised
        latency_ms = (time.monotonic() - start) * 1000
        await _enqueue_pipeline_run(
            store,
            tick_id=tick_id,
            call_id=call_id,
            pipeline=LLM_PIPELINE,
            stage="llm",
            latency_ms=latency_ms,
            error=str(exc),
        )
        return

    run_id = await _enqueue_pipeline_run(
        store,
        tick_id=tick_id,
        call_id=call_id,
        pipeline=LLM_PIPELINE,
        stage="llm",
        latency_ms=result["latency_ms"],
        input_tokens=result["input_tokens"],
        cached_input_tokens=result["cached_tokens"],
        output_tokens=result["output_tokens"],
        estimated_cost_usd=result["estimated_cost_usd"],
        raw_output_json=json.dumps(result["fields"]),
    )

    persist_tasks = []
    for field_name in llm_baseline.FIELDS:
        value = result["fields"][field_name]
        is_committed_now = result["is_committed"][field_name]
        if value is None and is_committed_now:
            # A confident null is the LLM actively reporting "no value" (its confidence reflects
            # certainty in *that* answer, not in a candidate value that doesn't exist) -- clear any
            # previously held commit instead of letting it fall through to _apply_carry_forward's
            # "nothing new, hold steady" branch and silently re-persist a stale value as current.
            committed.pop((LLM_PIPELINE, field_name), None)
        persist_tasks.append(
            _persist_field(
                store,
                committed,
                pipeline=LLM_PIPELINE,
                call_id=call_id,
                tick_number=tick_number,
                run_id=run_id,
                field_name=field_name,
                candidate_value=value,
                confidence=result["confidences"][field_name],
                is_committed_now=is_committed_now,
            )
        )
    await asyncio.gather(*persist_tasks)


async def run_call(
    call_id: int,
    db_path: str | Path | object,
    *,
    pacer_mode: str = "batch",
    calls: dict[int, dict] | None = None,
    resolver: JevFieldResolver | None = None,
    on_tick: Callable[[int, int, _CommittedState], None] | None = None,
    gliner_semaphore: asyncio.Semaphore | None = None,
    enable_llm_baseline: bool = False,
) -> None:
    """Replay `call_id`'s transcript and capture both pipelines' activity to the capture DB.

    `db_path` is either a filesystem path (a `CaptureStore` is created and closed around this
    call) or an already-constructed `CaptureStore`-like object (reused as-is, e.g. shared across
    concurrently orchestrated calls against the same database). `calls` defaults to
    `corpus.load_all()`; callers running many calls can load the corpus once and pass it in to
    avoid re-reading every transcript/metadata file per call. `resolver` follows the same
    reuse-or-own pattern as `db_path`: pass an already-constructed `JevFieldResolver` to share its
    underlying HTTP connection pool across concurrently orchestrated calls, or omit it to have
    `run_call` create and close its own for just this call. `gliner_semaphore`, when given, is
    acquired around every GLiNER inference this call makes -- shared across concurrently
    orchestrated calls to bound total GLiNER concurrency independently of how many calls are
    running at once.

    `enable_llm_baseline` defaults to `False`: the GPT-5.1 comparison arm costs real OpenAI API
    usage on every `config.LLM_CADENCE_TICKS`-th grown tick, so it never runs unless a caller opts
    in explicitly -- it is a deliberate, approved "final benchmark" comparison run, not routine
    GLiNER+jev data collection. When disabled, no `llm_baseline.extract` call is made and no `llm`
    pipeline rows are written for any tick.

    `pacer_mode` is `"batch"` (replay every tick back-to-back, no sleeping) or `"realtime"`
    (replay paced to wall-clock time).

    `on_tick`, if given, is called synchronously after every tick (whether or not the transcript
    grew that tick) with `(tick_number, total_ticks, committed_snapshot)`, where
    `committed_snapshot` is a shallow copy of the internal `(pipeline, field_name) -> (value,
    confidence)` carry-forward state at that point -- e.g. for a live-progress display. Exceptions
    raised by `on_tick` propagate, so a caller that wires this up to UI rendering is responsible
    for its own error handling.
    """
    calls = calls if calls is not None else corpus.load_all()
    if call_id not in calls:
        raise KeyError(f"call_id {call_id!r} not found in corpus")
    call_data = calls[call_id]
    scenario = call_data["scenario"]
    ground_truth = call_data["ground_truth"]
    transcript_turns = call_data["transcript_turns"]

    # gliner_pipeline/jev_pipeline/llm_baseline all key their internal per-call state off
    # call_id, and llm_baseline/gliner_pipeline type it as `str` -- normalize once here so every
    # pipeline call below shares one consistent key, independent of the `calls` table's integer
    # primary key.
    pipeline_call_id = str(call_id)

    call_pacer = CallPacer(call_id=call_id, transcript_turns=transcript_turns)

    if isinstance(db_path, (str, Path)):
        store = db_module.CaptureStore(db_path)
        owns_store = True
    else:
        store = db_path
        owns_store = False

    # `store` is already a live resource (a background writer thread plus an open sqlite
    # connection) by this point when `owns_store` is True, so its cleanup lives in its own
    # `finally` wrapping everything below -- including constructing `resolver`, which itself
    # opens an HTTP connection pool and could fail before the inner `try` even starts. Without
    # this outer layer, a `JevFieldResolver()` construction failure would leak `store`'s writer
    # thread and connection instead of closing it.
    try:
        owns_resolver = resolver is None
        resolver = resolver or JevFieldResolver()
        committed: _CommittedState = {}

        try:
            store.insert_call(
                call_id=call_id,
                scenario_json=json.dumps(scenario),
                category=scenario["category"],
                subtype=scenario["subtype"],
                edge_case=int(bool(scenario["edge_case"])),
                ground_truth_json=json.dumps(ground_truth),
                target_seconds=call_pacer.total_seconds,
                full_transcript_word_count=len(call_pacer.events),
            )

            if pacer_mode == "batch":
                tick_source = iter_batch_ticks(call_pacer)
            elif pacer_mode == "realtime":
                tick_source = iter_realtime_ticks(call_pacer)
            else:
                raise ValueError(f"unknown pacer_mode: {pacer_mode!r}")

            previous_offset = 0
            async for tick_number, snapshot, offset in _as_async_iter(tick_source):
                tick_future = store.enqueue_tick(
                    call_id=call_id,
                    tick_number=tick_number,
                    wall_clock_ts=time.time(),
                    transcript_char_offset=offset,
                    transcript_snapshot=snapshot,
                )
                tick_id = await asyncio.wrap_future(tick_future)

                grew = offset > previous_offset
                previous_offset = offset
                if grew:
                    steps = [
                        _run_gliner_jev_step(
                            store,
                            call_id=call_id,
                            pipeline_call_id=pipeline_call_id,
                            tick_id=tick_id,
                            tick_number=tick_number,
                            snapshot=snapshot,
                            resolver=resolver,
                            committed=committed,
                            gliner_semaphore=gliner_semaphore,
                        )
                    ]
                    if enable_llm_baseline and tick_number % config.LLM_CADENCE_TICKS == 0:
                        steps.append(
                            _run_llm_step(
                                store,
                                call_id=call_id,
                                pipeline_call_id=pipeline_call_id,
                                tick_id=tick_id,
                                tick_number=tick_number,
                                snapshot=snapshot,
                                committed=committed,
                            )
                        )
                    await asyncio.gather(*steps)

                if on_tick is not None:
                    on_tick(tick_number, call_pacer.total_ticks, dict(committed))
        finally:
            # Each cleanup step is isolated so one raising (e.g. a GLiNER session that was never
            # created because every tick had grew=False) doesn't skip the rest -- in particular,
            # `resolver.aclose()` must still run to avoid leaking its HTTP connection pool.
            try:
                gliner_pipeline.reset_call(pipeline_call_id)
            except Exception:
                _LOGGER.exception("gliner_pipeline.reset_call failed for call_id=%r", call_id)
            try:
                llm_baseline.reset_call(pipeline_call_id)
            except Exception:
                _LOGGER.exception("llm_baseline.reset_call failed for call_id=%r", call_id)
            if owns_resolver:
                try:
                    await resolver.aclose()
                except Exception:
                    _LOGGER.exception("resolver.aclose failed for call_id=%r", call_id)
    finally:
        if owns_store:
            store.close()


async def _as_async_iter(tick_source):
    """Adapt `iter_batch_ticks`'s sync generator and `iter_realtime_ticks`'s async generator to a
    single `async for`-able interface, so `run_call`'s tick loop doesn't need to know which one
    `pacer_mode` selected.
    """
    if hasattr(tick_source, "__anext__"):
        async for item in tick_source:
            yield item
    else:
        for item in tick_source:
            yield item
