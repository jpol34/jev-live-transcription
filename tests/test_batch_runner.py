"""Mocked tests for batch_runner -- no real GLiNER/jev/OpenAI calls, no real DB (kept fast/offline).

Exercises the concurrency-limiting logic (both semaphores) and the shared-store/shared-resolver
construction/cleanup, against a fake `pipeline_core.run_call` -- never the real pipeline.
"""

import asyncio
from unittest.mock import AsyncMock

import pytest

from jev_live_transcription import batch_runner


@pytest.mark.asyncio
@pytest.mark.parametrize("kwarg", ["call_concurrency", "gliner_concurrency"])
@pytest.mark.parametrize("bad_value", [0, -1])
async def test_run_batch_rejects_non_positive_concurrency(monkeypatch, kwarg, bad_value):
    # asyncio.Semaphore(0) is legal but can never be acquired -- every call would block forever
    # instead of raising, silently wedging the whole batch. Reject it up front instead.
    _patch_store_and_resolver(monkeypatch)
    monkeypatch.setattr(batch_runner.pipeline_core, "run_call", AsyncMock())

    with pytest.raises(ValueError):
        await batch_runner.run_batch(
            [1], db_path="ignored.sqlite3", calls={1: {}}, warm_up=False, **{kwarg: bad_value}
        )


class FakeStore:
    def __init__(self, db_path):
        self.db_path = db_path
        self.closed = False

    def close(self, timeout=5.0):
        self.closed = True


class FakeResolver:
    def __init__(self):
        self.aclose = AsyncMock()


def _patch_store_and_resolver(monkeypatch):
    store_instances: list[FakeStore] = []
    resolver_instances: list[FakeResolver] = []

    def make_store(db_path):
        store = FakeStore(db_path)
        store_instances.append(store)
        return store

    def make_resolver():
        resolver = FakeResolver()
        resolver_instances.append(resolver)
        return resolver

    monkeypatch.setattr(batch_runner.db_module, "CaptureStore", make_store)
    monkeypatch.setattr(batch_runner, "JevFieldResolver", make_resolver)
    return store_instances, resolver_instances


@pytest.mark.asyncio
async def test_run_batch_defaults_to_fully_sequential_execution(monkeypatch):
    # call_concurrency/gliner_concurrency default to 1 -- concurrent calls contend for the shared
    # GLiNER model, which inflates recorded latency with queueing delay a real single call would
    # never see, so the default must never let two calls run at once.
    _patch_store_and_resolver(monkeypatch)

    concurrent_now = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def fake_run_call(
        call_id, store, *, pacer_mode, calls, resolver, gliner_semaphore, enable_llm_baseline
    ):
        nonlocal concurrent_now, max_concurrent
        async with lock:
            concurrent_now += 1
            max_concurrent = max(max_concurrent, concurrent_now)
        await asyncio.sleep(0.01)
        async with lock:
            concurrent_now -= 1

    monkeypatch.setattr(batch_runner.pipeline_core, "run_call", fake_run_call)

    calls = {i: {} for i in range(1, 6)}
    result = await batch_runner.run_batch(
        list(calls), db_path="ignored.sqlite3", calls=calls, warm_up=False
    )

    assert max_concurrent == 1
    assert set(result.succeeded) == set(calls)


@pytest.mark.asyncio
async def test_run_batch_respects_call_concurrency(monkeypatch):
    store_instances, resolver_instances = _patch_store_and_resolver(monkeypatch)

    concurrent_now = 0
    max_concurrent = 0
    lock = asyncio.Lock()

    async def fake_run_call(
        call_id, store, *, pacer_mode, calls, resolver, gliner_semaphore, enable_llm_baseline
    ):
        nonlocal concurrent_now, max_concurrent
        async with lock:
            concurrent_now += 1
            max_concurrent = max(max_concurrent, concurrent_now)
        await asyncio.sleep(0.01)
        async with lock:
            concurrent_now -= 1

    monkeypatch.setattr(batch_runner.pipeline_core, "run_call", fake_run_call)

    calls = {i: {} for i in range(1, 11)}
    result = await batch_runner.run_batch(
        list(calls),
        db_path="ignored.sqlite3",
        call_concurrency=3,
        gliner_concurrency=2,
        calls=calls,
        warm_up=False,
    )

    assert max_concurrent == 3
    assert set(result.succeeded) == set(calls)
    assert result.failed == []

    # Exactly one shared store and one shared resolver, constructed once and closed once --
    # never one per call.
    assert len(store_instances) == 1
    assert len(resolver_instances) == 1
    assert store_instances[0].closed is True
    resolver_instances[0].aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_batch_gliner_semaphore_limits_concurrent_gliner_work(monkeypatch):
    _patch_store_and_resolver(monkeypatch)

    concurrent_gliner = 0
    max_concurrent_gliner = 0
    lock = asyncio.Lock()
    seen_semaphores = set()

    async def fake_run_call(
        call_id, store, *, pacer_mode, calls, resolver, gliner_semaphore, enable_llm_baseline
    ):
        nonlocal concurrent_gliner, max_concurrent_gliner
        seen_semaphores.add(id(gliner_semaphore))
        # Simulates what pipeline_core._run_gliner_jev_step actually does: acquire the shared
        # semaphore around each simulated GLiNER inference.
        async with gliner_semaphore:
            async with lock:
                concurrent_gliner += 1
                max_concurrent_gliner = max(max_concurrent_gliner, concurrent_gliner)
            await asyncio.sleep(0.01)
            async with lock:
                concurrent_gliner -= 1

    monkeypatch.setattr(batch_runner.pipeline_core, "run_call", fake_run_call)

    # call_concurrency is wide open (5) so all 5 calls run at once -- only gliner_concurrency
    # should bound the observed concurrency.
    calls = {i: {} for i in range(1, 6)}
    result = await batch_runner.run_batch(
        list(calls),
        db_path="ignored.sqlite3",
        call_concurrency=5,
        gliner_concurrency=2,
        calls=calls,
        warm_up=False,
    )

    assert max_concurrent_gliner == 2
    assert len(seen_semaphores) == 1  # the same semaphore instance is shared across every call
    assert set(result.succeeded) == set(calls)


@pytest.mark.asyncio
async def test_run_batch_isolates_per_call_failures(monkeypatch):
    _patch_store_and_resolver(monkeypatch)

    async def fake_run_call(
        call_id, store, *, pacer_mode, calls, resolver, gliner_semaphore, enable_llm_baseline
    ):
        if call_id == 2:
            raise RuntimeError("boom")

    monkeypatch.setattr(batch_runner.pipeline_core, "run_call", fake_run_call)

    calls = {i: {} for i in range(1, 4)}
    result = await batch_runner.run_batch(
        list(calls), db_path="ignored.sqlite3", calls=calls, warm_up=False
    )

    assert set(result.succeeded) == {1, 3}
    assert [call_id for call_id, _ in result.failed] == [2]
    assert isinstance(result.failed[0][1], RuntimeError)


@pytest.mark.asyncio
async def test_run_batch_cleans_up_store_and_resolver_even_on_failures(monkeypatch):
    store_instances, resolver_instances = _patch_store_and_resolver(monkeypatch)

    async def fake_run_call(
        call_id, store, *, pacer_mode, calls, resolver, gliner_semaphore, enable_llm_baseline
    ):
        raise RuntimeError("every call fails")

    monkeypatch.setattr(batch_runner.pipeline_core, "run_call", fake_run_call)

    calls = {1: {}, 2: {}}
    result = await batch_runner.run_batch(
        list(calls), db_path="ignored.sqlite3", calls=calls, warm_up=False
    )

    assert len(result.failed) == 2
    assert store_instances[0].closed is True
    resolver_instances[0].aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_batch_closes_store_even_when_resolver_aclose_raises(monkeypatch):
    # resolver.aclose() raising must not skip store.close() -- otherwise the CaptureStore's
    # background writer thread and live sqlite connection leak instead of shutting down cleanly.
    store_instances, _ = _patch_store_and_resolver(monkeypatch)

    failing_resolver = FakeResolver()
    failing_resolver.aclose.side_effect = RuntimeError("teardown boom")
    monkeypatch.setattr(batch_runner, "JevFieldResolver", lambda: failing_resolver)
    monkeypatch.setattr(batch_runner.pipeline_core, "run_call", AsyncMock())

    calls = {1: {}}
    result = await batch_runner.run_batch(
        list(calls), db_path="ignored.sqlite3", calls=calls, warm_up=False
    )

    assert store_instances[0].closed is True
    assert set(result.succeeded) == {1}


@pytest.mark.asyncio
async def test_run_batch_defaults_call_ids_to_every_call(monkeypatch):
    _patch_store_and_resolver(monkeypatch)

    seen_call_ids = []

    async def fake_run_call(
        call_id, store, *, pacer_mode, calls, resolver, gliner_semaphore, enable_llm_baseline
    ):
        seen_call_ids.append(call_id)

    monkeypatch.setattr(batch_runner.pipeline_core, "run_call", fake_run_call)

    calls = {3: {}, 1: {}, 2: {}}
    result = await batch_runner.run_batch(db_path="ignored.sqlite3", calls=calls, warm_up=False)

    assert set(seen_call_ids) == {1, 2, 3}
    assert set(result.succeeded) == {1, 2, 3}


@pytest.mark.asyncio
async def test_run_batch_warms_up_gliner_before_dispatching_calls(monkeypatch):
    _patch_store_and_resolver(monkeypatch)

    events = []

    async def fake_warm_up():
        events.append("warm_up")

    async def fake_run_call(
        call_id, store, *, pacer_mode, calls, resolver, gliner_semaphore, enable_llm_baseline
    ):
        events.append(f"call-{call_id}")

    monkeypatch.setattr(batch_runner, "_warm_up_gliner", fake_warm_up)
    monkeypatch.setattr(batch_runner.pipeline_core, "run_call", fake_run_call)

    calls = {1: {}, 2: {}}
    await batch_runner.run_batch(list(calls), db_path="ignored.sqlite3", calls=calls)

    assert events[0] == "warm_up"  # warm-up finishes before any call starts
    assert set(events[1:]) == {"call-1", "call-2"}


@pytest.mark.asyncio
async def test_run_batch_skips_warm_up_when_disabled(monkeypatch):
    _patch_store_and_resolver(monkeypatch)
    warm_up_mock = AsyncMock()
    monkeypatch.setattr(batch_runner, "_warm_up_gliner", warm_up_mock)
    monkeypatch.setattr(batch_runner.pipeline_core, "run_call", AsyncMock())

    calls = {1: {}}
    await batch_runner.run_batch(list(calls), db_path="ignored.sqlite3", calls=calls, warm_up=False)

    warm_up_mock.assert_not_called()


@pytest.mark.asyncio
async def test_run_batch_disables_llm_baseline_by_default(monkeypatch):
    _patch_store_and_resolver(monkeypatch)
    seen_flags = []

    async def fake_run_call(
        call_id, store, *, pacer_mode, calls, resolver, gliner_semaphore, enable_llm_baseline
    ):
        seen_flags.append(enable_llm_baseline)

    monkeypatch.setattr(batch_runner.pipeline_core, "run_call", fake_run_call)

    calls = {1: {}, 2: {}}
    await batch_runner.run_batch(list(calls), db_path="ignored.sqlite3", calls=calls, warm_up=False)

    # No enable_llm_baseline kwarg passed to run_batch -- every call must default to disabled.
    assert seen_flags == [False, False]


@pytest.mark.asyncio
async def test_run_batch_forwards_enable_llm_baseline_when_opted_in(monkeypatch):
    _patch_store_and_resolver(monkeypatch)
    seen_flags = []

    async def fake_run_call(
        call_id, store, *, pacer_mode, calls, resolver, gliner_semaphore, enable_llm_baseline
    ):
        seen_flags.append(enable_llm_baseline)

    monkeypatch.setattr(batch_runner.pipeline_core, "run_call", fake_run_call)

    calls = {1: {}}
    await batch_runner.run_batch(
        list(calls), db_path="ignored.sqlite3", calls=calls, warm_up=False, enable_llm_baseline=True
    )

    assert seen_flags == [True]
