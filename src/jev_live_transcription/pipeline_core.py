"""Orchestrates one call end to end: replays its transcript via the pacer and, at every tick,
runs the GLiNER+jev pipeline and the GPT-5.1 baseline concurrently, capturing every tick's
activity to the capture DB.

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
import json
import re
import time
from pathlib import Path

from . import config, corpus, gliner_pipeline, llm_baseline
from .jev_pipeline import JevFieldResolver
from .pacer import CallPacer, iter_batch_ticks, iter_realtime_ticks
from . import db as db_module

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


def context_window(snapshot: str, start: int, end: int) -> str:
    """Slice `snapshot` to the sentence(s) spanning `[start, end)` plus one sentence on each side.

    Falls back to the full snapshot when `start`/`end` don't land inside any sentence span (e.g.
    missing offsets from an upstream candidate).
    """
    spans = _sentence_spans(snapshot)
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


def _field_context_window(snapshot: str, candidate_spans: list[dict]) -> str:
    """Context window covering every candidate span found for one field on one tick."""
    starts = [c["start"] for c in candidate_spans if c.get("start") is not None]
    ends = [c["end"] for c in candidate_spans if c.get("end") is not None]
    if not starts or not ends:
        return snapshot
    return context_window(snapshot, min(starts), max(ends))


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
) -> None:
    start = time.monotonic()
    try:
        candidates = await gliner_pipeline.extract_candidates(snapshot, pipeline_call_id)
    except Exception as exc:  # noqa: BLE001 -- GLiNER failures are captured as data, not raised
        latency_ms = (time.monotonic() - start) * 1000
        await _enqueue_pipeline_run(
            store,
            tick_id=tick_id,
            call_id=call_id,
            pipeline=GLINER_JEV_PIPELINE,
            stage="gliner_standard",
            latency_ms=latency_ms,
            error=str(exc),
        )
        return
    latency_ms = (time.monotonic() - start) * 1000

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
            latency_ms=latency_ms,
            raw_output_json=json.dumps(pii_candidates),
        ),
        _enqueue_pipeline_run(
            store,
            tick_id=tick_id,
            call_id=call_id,
            pipeline=GLINER_JEV_PIPELINE,
            stage="gliner_standard",
            latency_ms=latency_ms,
            raw_output_json=json.dumps(zero_shot_candidates),
        ),
    )

    resolve_tasks = []
    resolve_field_names = []
    for field_name, spans in candidates.items():
        values = [span["text"] for span in spans if span.get("text")]
        if not values:
            continue
        context = _field_context_window(snapshot, spans)
        resolve_tasks.append(
            resolver.resolve_field(pipeline_call_id, field_name, values, context)
        )
        resolve_field_names.append(field_name)
    if not resolve_tasks:
        return

    results = await asyncio.gather(*resolve_tasks, return_exceptions=True)
    for field_name, result in zip(resolve_field_names, results):
        if isinstance(result, Exception):
            await _enqueue_pipeline_run(
                store,
                tick_id=tick_id,
                call_id=call_id,
                pipeline=GLINER_JEV_PIPELINE,
                stage="jev",
                error=str(result),
            )
            continue
        if result is None:
            continue
        persisted_value, persisted_confidence, persisted_committed = _apply_carry_forward(
            committed,
            GLINER_JEV_PIPELINE,
            field_name,
            result.candidate if not result.is_none_of_these else None,
            result.confidence,
            result.is_committed and not result.is_none_of_these,
        )
        run_id = await _enqueue_pipeline_run(
            store,
            tick_id=tick_id,
            call_id=call_id,
            pipeline=GLINER_JEV_PIPELINE,
            stage="jev",
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
        store.enqueue_field_extraction(
            run_id=run_id,
            call_id=call_id,
            tick_number=tick_number,
            pipeline=GLINER_JEV_PIPELINE,
            field_name=field_name,
            candidate_value=persisted_value,
            confidence=persisted_confidence,
            is_committed=int(persisted_committed),
        )


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
    try:
        result = await llm_baseline.extract(pipeline_call_id, snapshot)
    except Exception as exc:  # noqa: BLE001 -- LLM failures are captured as data, not raised
        await _enqueue_pipeline_run(
            store,
            tick_id=tick_id,
            call_id=call_id,
            pipeline=LLM_PIPELINE,
            stage="llm",
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

    for field_name in llm_baseline.FIELDS:
        value = result["fields"][field_name]
        confidence = result["confidences"][field_name]
        is_committed_now = result["is_committed"][field_name]
        persisted_value, persisted_confidence, persisted_committed = _apply_carry_forward(
            committed, LLM_PIPELINE, field_name, value, confidence, is_committed_now
        )
        store.enqueue_field_extraction(
            run_id=run_id,
            call_id=call_id,
            tick_number=tick_number,
            pipeline=LLM_PIPELINE,
            field_name=field_name,
            candidate_value=persisted_value,
            confidence=persisted_confidence,
            is_committed=int(persisted_committed),
        )


async def run_call(
    call_id: int,
    db_path: str | Path | object,
    *,
    pacer_mode: str = "batch",
    calls: dict[int, dict] | None = None,
) -> None:
    """Replay `call_id`'s transcript and capture both pipelines' activity to the capture DB.

    `db_path` is either a filesystem path (a `CaptureStore` is created and closed around this
    call) or an already-constructed `CaptureStore`-like object (reused as-is, e.g. shared across
    concurrently orchestrated calls against the same database). `calls` defaults to
    `corpus.load_all()`; callers running many calls can load the corpus once and pass it in to
    avoid re-reading every transcript/metadata file per call.

    `pacer_mode` is `"batch"` (replay every tick back-to-back, no sleeping) or `"realtime"`
    (replay paced to wall-clock time).
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

    resolver = JevFieldResolver()
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
            if not grew:
                continue

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
                )
            ]
            if tick_number % config.LLM_CADENCE_TICKS == 0:
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
    finally:
        gliner_pipeline.reset_call(pipeline_call_id)
        llm_baseline.reset_call(pipeline_call_id)
        await resolver.aclose()
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
