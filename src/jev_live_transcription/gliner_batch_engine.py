"""Async dynamic-batching scheduler for GLiNER's stateless zero-shot inference.

Modeled on GLiNER's own `gliner.streaming.AsyncStreamingEngine` (asyncio.Queue + a
timeout-windowed batch collector + a dedicated single-worker `ThreadPoolExecutor` so the GPU
forward pass never blocks the event loop), but that engine only exists for the
`StreamingSpanGLiNER` architecture and this project's checkpoint (`urchade/gliner_medium-v2.1`)
loads as `UniEncoderSpanGLiNER`, which doesn't have it. This engine calls the architecture-agnostic
`GLiNER.inference(texts, labels, batch_size=len(texts), ...)` instead -- the same method
`predict_entities` already delegates to for a single text, extended here to run one real batched
forward pass across whatever concurrent requests land in the same collection window.

Unlike `AsyncStreamingEngine`, this engine has no per-session state (no KV-cache, no
`clear_session`) -- every request is a one-shot, stateless call, so there's no need for
`AsyncStreamingEngine.append`'s per-session locking.
"""

from __future__ import annotations

import asyncio
import logging
import weakref
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import partial
from typing import Any

_LOGGER = logging.getLogger(__name__)

_STOP = object()


@dataclass
class _BatchRequest:
    text: str
    future: asyncio.Future


class GlinerBatchEngine:
    """Collects concurrent `submit()` calls into batched `model.inference()` calls."""

    def __init__(
        self,
        model: Any,
        labels: dict[str, str],
        *,
        max_batch_size: int,
        batch_wait_timeout_ms: float,
        threshold: float,
        multi_label: bool = True,
        queue_capacity: int = 4096,
    ) -> None:
        if max_batch_size < 1:
            raise ValueError("max_batch_size must be positive")
        if batch_wait_timeout_ms < 0:
            raise ValueError("batch_wait_timeout_ms must be non-negative")

        self.model = model
        self.labels = labels
        self.max_batch_size = int(max_batch_size)
        self.batch_wait_timeout_s = float(batch_wait_timeout_ms) / 1000.0
        self.threshold = threshold
        self.multi_label = multi_label

        self._queue: asyncio.Queue[_BatchRequest | object] = asyncio.Queue(maxsize=queue_capacity)
        self._worker_task: asyncio.Task | None = None
        self._executor: ThreadPoolExecutor | None = None
        self._loop: asyncio.AbstractEventLoop | None = None
        self._closing = False
        self._closed = False

    async def start(self) -> GlinerBatchEngine:
        """Start the scheduler on the current event loop. Not safe to reuse across loops --
        callers that may run under more than one event loop (tests included) should key a cache of
        engines by loop rather than sharing a single instance.
        """
        if self._closed:
            raise RuntimeError("GlinerBatchEngine is closed")
        loop = asyncio.get_running_loop()
        if self._loop is not None and self._loop is not loop:
            raise RuntimeError("GlinerBatchEngine cannot move between event loops")
        self._loop = loop
        if self._worker_task is None:
            self._executor = ThreadPoolExecutor(max_workers=1, thread_name_prefix="gliner-batch")
            self._worker_task = loop.create_task(self._worker(), name="gliner-batch-scheduler")
        return self

    async def submit(self, text: str) -> list[dict]:
        """Enqueue one text for extraction and await its entities, batched with whatever other
        `submit()` calls land in the same collection window.
        """
        if self._closing or self._closed:
            raise RuntimeError("GlinerBatchEngine is closing or closed")
        await self.start()
        loop = asyncio.get_running_loop()
        future: asyncio.Future = loop.create_future()
        await self._queue.put(_BatchRequest(text=text, future=future))
        return await asyncio.shield(future)

    async def _collect_batch(self, first: _BatchRequest) -> list[_BatchRequest]:
        batch = [first]
        if self.batch_wait_timeout_s == 0:
            while len(batch) < self.max_batch_size:
                try:
                    item = self._queue.get_nowait()
                except asyncio.QueueEmpty:
                    break
                if item is _STOP:
                    self._queue.task_done()
                    self._closing = True
                    break
                batch.append(item)
            return batch

        loop = asyncio.get_running_loop()
        deadline = loop.time() + self.batch_wait_timeout_s
        while len(batch) < self.max_batch_size:
            remaining = deadline - loop.time()
            if remaining <= 0:
                break
            try:
                item = await asyncio.wait_for(self._queue.get(), timeout=remaining)
            except TimeoutError:
                break
            if item is _STOP:
                self._queue.task_done()
                self._closing = True
                break
            batch.append(item)
        return batch

    async def _worker(self) -> None:
        assert self._loop is not None
        assert self._executor is not None
        while True:
            item = await self._queue.get()
            if item is _STOP:
                self._queue.task_done()
                return
            batch = await self._collect_batch(item)
            texts = [request.text for request in batch]
            try:
                job = self._loop.run_in_executor(
                    self._executor,
                    partial(
                        self.model.inference,
                        texts,
                        self.labels,
                        batch_size=len(texts),
                        multi_label=self.multi_label,
                        threshold=self.threshold,
                    ),
                )
                # Polling keeps the event loop responsive on runtimes where a PyTorch worker's
                # cross-thread wakeup can be delayed until the selector's next timer event (same
                # workaround `AsyncStreamingEngine._worker` uses for the same reason).
                while not job.done():
                    await asyncio.sleep(0.001)
                outputs = job.result()
            except BaseException as error:
                # `model.inference()` has no per-item exception isolation (unlike the streaming
                # engine's `_run_session_items_batched(..., return_exceptions=True)`), so one bad
                # item fails the whole batch -- every request sharing this batch gets the same
                # exception. Logged here since this is the only place batch composition is visible.
                _LOGGER.warning(
                    "GlinerBatchEngine batch of %d failed: %s", len(batch), error, exc_info=True
                )
                for request in batch:
                    if not request.future.done():
                        request.future.set_exception(error)
            else:
                _LOGGER.debug("GlinerBatchEngine dispatched batch of %d", len(batch))
                for request, output in zip(batch, outputs, strict=False):
                    if not request.future.done():
                        request.future.set_result(output)
            finally:
                for _ in batch:
                    self._queue.task_done()

    async def close(self) -> None:
        """Drain queued work and stop the scheduler. Idempotent."""
        if self._closed:
            return
        self._closing = True
        if self._worker_task is not None:
            await self._queue.join()
            await self._queue.put(_STOP)
            await self._worker_task
            self._worker_task = None
        if self._executor is not None:
            self._executor.shutdown(wait=True, cancel_futures=False)
            self._executor = None
        self._closed = True

    @property
    def is_running(self) -> bool:
        """True once `start()` has run and its worker task hasn't died unexpectedly -- used by
        health checks (e.g. the serving app's `/healthz`) to distinguish a live engine from one
        whose worker crashed silently, since a dead worker would otherwise hang every subsequent
        `submit()` forever with no visible symptom until a caller's own timeout (if any) fires.
        """
        return self._worker_task is not None and not self._worker_task.done()


_engines_by_loop: "weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, GlinerBatchEngine]" = (
    weakref.WeakKeyDictionary()
)


async def get_or_create_engine(factory) -> GlinerBatchEngine:
    """Return the engine bound to the current event loop, creating one via `factory()` (a
    zero-arg callable returning an unstarted `GlinerBatchEngine`) if none exists yet for this loop.

    A bare module-level singleton would break the moment more than one event loop touches it in
    the process's lifetime (e.g. this repo's own test suite calls `asyncio.run(...)` once per test
    function) -- keying by the running loop instead means a new loop always gets its own fresh,
    correctly-bound engine rather than raising or silently reusing a stale one.
    """
    loop = asyncio.get_running_loop()
    engine = _engines_by_loop.get(loop)
    if engine is None:
        engine = factory()
        await engine.start()
        _engines_by_loop[loop] = engine
    return engine
