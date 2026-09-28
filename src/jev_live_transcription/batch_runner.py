"""Drives the full call corpus through `pipeline_core.run_call`.

Constructs and owns exactly one `CaptureStore` and one `JevFieldResolver` for the whole batch --
shared across every call rather than one per call, per `run_call`'s reuse-or-own pattern -- and
closes both once every call has finished, regardless of individual failures.

Two independent semaphores bound how many calls, and how many of those calls' GLiNER inferences,
may run at once: `call_concurrency` and `gliner_concurrency`. `call_concurrency` defaults to `1`
-- fully sequential, one call fully processed before the next starts -- because this benchmark's
actual purpose is measuring how fast the pipeline extracts fields from *one* live call, and any
concurrency above 1 makes calls contend for the same GLiNER model, which is a single shared
resource regardless of which device it runs on. That contention shows up as queueing delay inside
`pipeline_runs.latency_ms` indistinguishably from real inference time, even though a real live
call would never experience it -- so it silently inflates the exact number this benchmark exists
to measure. Raising `call_concurrency` is a throughput/correctness trade a caller can make
deliberately (e.g. a quick smoke run across the whole corpus to check for crashes, not to read its
latency numbers), never the right choice for collecting real benchmark data. `gliner_concurrency`
behaves differently and defaults to `config.GLINER_CONCURRENCY` (sized to match
`GlinerBatchEngine`'s own batching capacity, not `1`): `GlinerBatchEngine` batches concurrent
GLiNER calls into one real batched forward pass rather than serializing them, so raising it can
yield genuine throughput instead of the pure queueing delay `call_concurrency` above always adds --
though with `call_concurrency` at its own default of `1`, only one call's ticks are ever in flight
at a time regardless of `gliner_concurrency`, so this only matters once `call_concurrency` is
raised too. `run_batch` warns, but does not clamp, if it's passed a value other than
`config.GLINER_CONCURRENCY`.
"""

from __future__ import annotations

import asyncio
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

from . import config, corpus, gliner_pipeline, pipeline_core
from . import db as db_module
from .jev_pipeline import JevFieldResolver

_LOGGER = logging.getLogger(__name__)

_WARMUP_CALL_ID = "__warmup__"
_WARMUP_TRANSCRIPT = "Agent: hello, thanks for calling. Caller: hi, I have a question."


@dataclass
class BatchResult:
    """Outcome of one `run_batch` call: which call_ids completed and which raised."""

    succeeded: list[int] = field(default_factory=list)
    failed: list[tuple[int, BaseException]] = field(default_factory=list)


async def warm_up_gliner() -> None:
    """Force both GLiNER checkpoints to load now, on a throwaway snapshot.

    GLiNER's first inference call in a process pays a one-time model-load cost (observed at
    roughly 26-28s on a CPU-only machine; device-dependent -- see `config.GLINER_DEVICE`) on top
    of its actual per-tick latency -- calling it once here, before any real call's ticks are
    timed, keeps that load cost out of the latency numbers `pipeline_core` records for every
    subsequent (real) tick.

    jev and the LLM baseline are not warmed up here: both are plain HTTP calls over an
    already-constructed client, whose only cold-start cost is a TCP/TLS handshake -- on the order
    of the per-call latency itself, not a separate multi-second load like GLiNER's.
    """
    start = time.monotonic()
    await gliner_pipeline.extract_candidates_timed(_WARMUP_TRANSCRIPT, _WARMUP_CALL_ID)
    gliner_pipeline.reset_call(_WARMUP_CALL_ID)
    _LOGGER.info("GLiNER warm-up finished in %.1fs", time.monotonic() - start)


async def _run_one(
    call_id: int,
    *,
    store,
    resolver: JevFieldResolver,
    calls: dict[int, dict],
    call_semaphore: asyncio.Semaphore,
    gliner_semaphore: asyncio.Semaphore,
    enable_llm_baseline: bool,
    result: BatchResult,
) -> None:
    """Run one call under `call_semaphore`, recording its outcome on `result` either way.

    A single call's unhandled exception (distinct from a GLiNER/jev/LLM step's own failure, which
    `run_call` already captures as a `pipeline_runs` error row rather than raising) must not abort
    the rest of the batch or cancel other in-flight calls sharing the same store/resolver.
    """
    async with call_semaphore:
        try:
            await pipeline_core.run_call(
                call_id,
                store,
                pacer_mode="batch",
                calls=calls,
                resolver=resolver,
                gliner_semaphore=gliner_semaphore,
                enable_llm_baseline=enable_llm_baseline,
            )
        except Exception as exc:  # noqa: BLE001 -- isolated per call, see docstring
            _LOGGER.exception("batch run_call failed for call_id=%r", call_id)
            result.failed.append((call_id, exc))
        else:
            result.succeeded.append(call_id)


async def run_batch(
    call_ids: list[int] | None = None,
    *,
    db_path: str | Path,
    call_concurrency: int = 1,
    gliner_concurrency: int = config.GLINER_CONCURRENCY,
    calls: dict[int, dict] | None = None,
    warm_up: bool = True,
    enable_llm_baseline: bool = False,
) -> BatchResult:
    """Drive `call_ids` (default: every call in `calls`) through `pipeline_core.run_call`.

    Every call replays in `"batch"` pacer mode (no sleeping) against one shared `CaptureStore` at
    `db_path` and one shared `JevFieldResolver`, both constructed here and closed once at the end.
    `calls` defaults to `corpus.load_all()`, loaded once and passed to every call rather than
    re-read per call. `warm_up` (default `True`) runs `warm_up_gliner` before any call starts;
    callers that already warmed up GLiNER in this process, or tests exercising this function
    without real models, pass `warm_up=False`.

    `call_concurrency` defaults to `1` and `gliner_concurrency` to `config.GLINER_CONCURRENCY`
    (fully sequential) for the cross-call-contention reason explained in this module's docstring;
    raising `call_concurrency` above 1 trades away methodologically valid latency numbers for
    wall-clock throughput, so only do so for a run whose latency data won't be used (e.g.
    `config.CALL_CONCURRENCY` for a quick smoke pass across the corpus). `gliner_concurrency` has
    no such tradeoff to make -- a value above `config.GLINER_CONCURRENCY` logs a warning but is not
    clamped, since `scripts/measure_gliner_concurrency.py` deliberately passes higher values to
    re-measure it.

    `enable_llm_baseline` defaults to `False` and is forwarded as-is to every call's
    `pipeline_core.run_call` -- see its docstring for why the GPT-5.1 comparison arm is opt-in
    rather than routine.
    """
    # asyncio.Semaphore(0) is legal but can never be acquired -- every _run_one would block
    # forever waiting to acquire it, hanging the whole batch with no error or log line explaining
    # why. Reject non-positive concurrency up front instead.
    if call_concurrency < 1:
        raise ValueError(f"call_concurrency must be >= 1, got {call_concurrency!r}")
    if gliner_concurrency < 1:
        raise ValueError(f"gliner_concurrency must be >= 1, got {gliner_concurrency!r}")
    if gliner_concurrency > config.GLINER_CONCURRENCY:
        _LOGGER.warning(
            "gliner_concurrency=%d is above config.GLINER_CONCURRENCY=%d, which is sized to match "
            "GlinerBatchEngine's own batching capacity -- going higher just means more ticks queue "
            "for the next batch rather than any one batch growing further. Proceeding anyway.",
            gliner_concurrency,
            config.GLINER_CONCURRENCY,
        )

    calls = calls if calls is not None else corpus.load_all()
    if call_ids is None:
        call_ids = sorted(calls)

    if warm_up:
        await warm_up_gliner()

    call_semaphore = asyncio.Semaphore(call_concurrency)
    gliner_semaphore = asyncio.Semaphore(gliner_concurrency)

    store = db_module.CaptureStore(db_path)
    resolver = JevFieldResolver()
    result = BatchResult()
    try:
        await asyncio.gather(
            *(
                _run_one(
                    call_id,
                    store=store,
                    resolver=resolver,
                    calls=calls,
                    call_semaphore=call_semaphore,
                    gliner_semaphore=gliner_semaphore,
                    enable_llm_baseline=enable_llm_baseline,
                    result=result,
                )
                for call_id in call_ids
            )
        )
    finally:
        # Isolated like pipeline_core.run_call's own cleanup: resolver.aclose() raising must not
        # skip store.close(), or the CaptureStore's background writer thread and live sqlite
        # connection leak instead of shutting down cleanly.
        try:
            await resolver.aclose()
        except Exception:
            _LOGGER.exception("resolver.aclose failed")
        try:
            store.close()
        except Exception:
            _LOGGER.exception("store.close failed")
    return result
