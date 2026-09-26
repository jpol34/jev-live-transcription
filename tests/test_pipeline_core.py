"""Mocked tests for pipeline_core -- no real GLiNER/jev/OpenAI calls (kept fast/offline)."""

import concurrent.futures
from unittest.mock import AsyncMock, Mock

import pytest

from jev_live_transcription import pipeline_core
from jev_live_transcription.jev_pipeline import JevResolution


def _resolved_future(value):
    future: concurrent.futures.Future = concurrent.futures.Future()
    future.set_result(value)
    return future


class FakeStore:
    """Records every write pipeline_core makes, mirroring db.CaptureStore's public surface."""

    def __init__(self):
        self.calls = []
        self.ticks = []
        self.pipeline_runs = []
        self.field_extractions = []
        self.closed = False

    def insert_call(self, **fields):
        self.calls.append(fields)
        return fields["call_id"]

    def enqueue_tick(self, **fields):
        tick_id = len(self.ticks) + 1
        self.ticks.append({"tick_id": tick_id, **fields})
        return _resolved_future(tick_id)

    def enqueue_pipeline_run(self, **fields):
        run_id = len(self.pipeline_runs) + 1
        self.pipeline_runs.append({"run_id": run_id, **fields})
        return _resolved_future(run_id)

    def enqueue_field_extraction(self, **fields):
        self.field_extractions.append(fields)
        return _resolved_future(None)

    def close(self, timeout=5.0):
        self.closed = True


class FakeResolver:
    def __init__(self, resolve_field_mock):
        self.resolve_field = resolve_field_mock
        self.aclose = AsyncMock()


def _calls_fixture():
    return {
        1: {
            "scenario": {"id": 1, "category": "resident", "subtype": "test_subtype", "edge_case": False},
            "ground_truth": {"caller_name": "Tim Barker"},
            "transcript_turns": [
                {"speaker": "Agent", "text": "hello there"},
                {"speaker": "Caller", "text": "hi"},
            ],
        }
    }


def _all_llm_fields(caller_name=(None, 0.0, False)):
    fields = {name: None for name in pipeline_core.llm_baseline.FIELDS}
    confidences = {name: 0.0 for name in pipeline_core.llm_baseline.FIELDS}
    is_committed = {name: False for name in pipeline_core.llm_baseline.FIELDS}
    value, confidence, committed = caller_name
    fields["caller_name"] = value
    confidences["caller_name"] = confidence
    is_committed["caller_name"] = committed
    return {
        "fields": fields,
        "confidences": confidences,
        "is_committed": is_committed,
        "response_id": "resp",
        "input_tokens": 100,
        "cached_tokens": 0,
        "output_tokens": 10,
        "estimated_cost_usd": 0.01,
        "latency_ms": 5.0,
    }


# --- _apply_carry_forward ------------------------------------------------------------------


def test_apply_carry_forward_new_commit_supersedes():
    committed = {}
    value, confidence, is_committed = pipeline_core._apply_carry_forward(
        committed, "gliner_jev", "phone_number", "555-1111", 0.9, True
    )
    assert (value, confidence, is_committed) == ("555-1111", 0.9, True)
    assert committed[("gliner_jev", "phone_number")] == ("555-1111", 0.9)

    # A later, different committed value supersedes the held one.
    value, confidence, is_committed = pipeline_core._apply_carry_forward(
        committed, "gliner_jev", "phone_number", "555-2222", 0.85, True
    )
    assert (value, confidence, is_committed) == ("555-2222", 0.85, True)
    assert committed[("gliner_jev", "phone_number")] == ("555-2222", 0.85)


def test_apply_carry_forward_holds_steady_when_dip_below_threshold():
    committed = {("gliner_jev", "phone_number"): ("555-2222", 0.85)}
    value, confidence, is_committed = pipeline_core._apply_carry_forward(
        committed, "gliner_jev", "phone_number", "555-2222", 0.4, False
    )
    # Held steady at the previously committed value/confidence, not the raw dip.
    assert (value, confidence, is_committed) == ("555-2222", 0.85, True)


def test_apply_carry_forward_uncommitted_with_nothing_held_stays_uncommitted():
    committed = {}
    value, confidence, is_committed = pipeline_core._apply_carry_forward(
        committed, "llm", "caller_name", "Tim", 0.3, False
    )
    assert (value, confidence, is_committed) == ("Tim", 0.3, False)
    assert committed == {}


# --- context window slicing ------------------------------------------------------------------


def test_context_window_includes_neighbor_sentences():
    text = "Agent: hi. Caller: my number is 555-1111. Thanks for calling."
    start = text.index("555-1111")
    end = start + len("555-1111")
    window = pipeline_core.context_window(text, start, end)
    assert "555-1111" in window
    assert "hi." in window
    assert "Thanks for calling." in window


def test_context_window_falls_back_to_full_text_without_matching_span():
    text = "One sentence only"
    assert pipeline_core.context_window(text, 0, len(text)) == text


# --- run_call tick loop: growth gating, LLM cadence gating, carry-forward -----------------------


@pytest.mark.asyncio
async def test_run_call_gating_and_carry_forward(monkeypatch):
    ticks_script = [
        (0, "", 0),
        (1, "a", 1),  # growth -> gliner+jev only (1 % 3 != 0)
        (2, "a", 1),  # no growth -> nothing fires
        (3, "ab", 2),  # growth -> gliner+jev + llm (3 % 3 == 0)
        (4, "ab", 2),  # no growth
        (5, "abc", 3),  # growth -> gliner+jev only, no candidates this tick
        (6, "abcd", 4),  # growth -> gliner+jev + llm
    ]
    monkeypatch.setattr(pipeline_core, "iter_batch_ticks", lambda call_pacer: iter(ticks_script))

    gliner_side_effects = [
        {"phone_number": [{"text": "555-1111", "score": 0.9, "start": 0, "end": 8}]},
        {
            "phone_number": [
                {"text": "555-1111", "score": 0.9, "start": 0, "end": 8},
                {"text": "555-2222", "score": 0.85, "start": 9, "end": 17},
            ]
        },
        {"phone_number": []},
        {
            "phone_number": [
                {"text": "555-1111", "score": 0.9, "start": 0, "end": 8},
                {"text": "555-2222", "score": 0.85, "start": 9, "end": 17},
            ]
        },
    ]
    extract_candidates_mock = AsyncMock(side_effect=gliner_side_effects)
    monkeypatch.setattr(pipeline_core.gliner_pipeline, "extract_candidates", extract_candidates_mock)
    gliner_reset_mock = Mock()
    monkeypatch.setattr(pipeline_core.gliner_pipeline, "reset_call", gliner_reset_mock)

    jev_results = [
        JevResolution(
            call_id="1", field_name="phone_number", question_type="noul", candidate="555-1111",
            confidence=0.9, is_committed=True, is_none_of_these=False,
            distinct_candidates=("555-1111",), input_tokens=10, output_tokens=2,
        ),
        JevResolution(
            call_id="1", field_name="phone_number", question_type="choice", candidate="555-2222",
            confidence=0.85, is_committed=True, is_none_of_these=False,
            distinct_candidates=("555-1111", "555-2222"), input_tokens=15, output_tokens=3,
        ),
        JevResolution(
            call_id="1", field_name="phone_number", question_type="choice", candidate="555-2222",
            confidence=0.4, is_committed=False, is_none_of_these=False,
            distinct_candidates=("555-1111", "555-2222"), input_tokens=15, output_tokens=3,
        ),
    ]
    resolve_field_mock = AsyncMock(side_effect=jev_results)
    monkeypatch.setattr(pipeline_core, "JevFieldResolver", lambda: FakeResolver(resolve_field_mock))

    llm_side_effects = [
        _all_llm_fields(caller_name=("Tim", 0.9, True)),
        _all_llm_fields(caller_name=("Tim", 0.3, False)),
    ]
    llm_extract_mock = AsyncMock(side_effect=llm_side_effects)
    monkeypatch.setattr(pipeline_core.llm_baseline, "extract", llm_extract_mock)
    llm_reset_mock = Mock()
    monkeypatch.setattr(pipeline_core.llm_baseline, "reset_call", llm_reset_mock)

    store = FakeStore()
    await pipeline_core.run_call(1, store, calls=_calls_fixture())

    # --- snapshot-growth gating ---
    assert extract_candidates_mock.await_count == 4  # ticks 1, 3, 5, 6
    assert len(store.ticks) == 7  # every tick gets a ticks row regardless of growth

    # --- LLM cadence gating (only on grown ticks where tick_number % 3 == 0) ---
    assert llm_extract_mock.await_count == 2  # ticks 3, 6
    llm_run_tick_ids = {run["tick_id"] for run in store.pipeline_runs if run["pipeline"] == "llm"}
    llm_run_tick_numbers = {
        tick["tick_number"] for tick in store.ticks if tick["tick_id"] in llm_run_tick_ids
    }
    assert llm_run_tick_numbers == {3, 6}

    # --- carry-forward: gliner_jev phone_number resolves wrong, then flips to correct, then
    # holds the corrected value steady through a later dip below threshold ---
    phone_rows = [
        fe
        for fe in store.field_extractions
        if fe["pipeline"] == "gliner_jev" and fe["field_name"] == "phone_number"
    ]
    assert len(phone_rows) == 3  # ticks 1, 3, 6 -- no row for tick 5 (no candidates that tick)
    assert [row["tick_number"] for row in phone_rows] == [1, 3, 6]
    assert phone_rows[0]["candidate_value"] == "555-1111"
    assert phone_rows[0]["is_committed"] == 1
    assert phone_rows[1]["candidate_value"] == "555-2222"  # flips to the corrected value
    assert phone_rows[1]["is_committed"] == 1
    assert phone_rows[2]["candidate_value"] == "555-2222"  # held steady despite this tick's dip
    assert phone_rows[2]["confidence"] == 0.85
    assert phone_rows[2]["is_committed"] == 1

    # --- carry-forward on the llm pipeline too ---
    caller_name_rows = [
        fe
        for fe in store.field_extractions
        if fe["pipeline"] == "llm" and fe["field_name"] == "caller_name"
    ]
    assert len(caller_name_rows) == 2
    assert caller_name_rows[0] == {
        "run_id": caller_name_rows[0]["run_id"],
        "call_id": 1,
        "tick_number": 3,
        "pipeline": "llm",
        "field_name": "caller_name",
        "candidate_value": "Tim",
        "confidence": 0.9,
        "is_committed": 1,
    }
    assert caller_name_rows[1]["candidate_value"] == "Tim"
    assert caller_name_rows[1]["confidence"] == 0.9  # held steady, not the dipped 0.3
    assert caller_name_rows[1]["is_committed"] == 1

    # --- referential integrity: every field_extractions row's run_id/tick_number trace back to
    # a real pipeline_runs row and a real ticks row (no orphans) ---
    run_ids = {run["run_id"] for run in store.pipeline_runs}
    tick_ids = {tick["tick_id"] for tick in store.ticks}
    for row in store.field_extractions:
        assert row["run_id"] in run_ids
    for run in store.pipeline_runs:
        assert run["tick_id"] in tick_ids

    # --- per-call cleanup ---
    gliner_reset_mock.assert_called_once_with("1")
    llm_reset_mock.assert_called_once_with("1")
    assert store.closed is False  # an externally-provided store is never closed by run_call


# --- on_tick callback ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_call_on_tick_callback_fires_every_tick_with_committed_snapshot(monkeypatch):
    ticks_script = [
        (0, "", 0),  # no growth -> on_tick still fires, snapshot is empty
        (1, "a", 1),  # growth -> gliner+jev commits phone_number
    ]
    monkeypatch.setattr(pipeline_core, "iter_batch_ticks", lambda call_pacer: iter(ticks_script))

    extract_candidates_mock = AsyncMock(
        return_value={"phone_number": [{"text": "555-1111", "score": 0.9, "start": 0, "end": 8}]}
    )
    monkeypatch.setattr(pipeline_core.gliner_pipeline, "extract_candidates", extract_candidates_mock)
    monkeypatch.setattr(pipeline_core.gliner_pipeline, "reset_call", Mock())
    monkeypatch.setattr(pipeline_core.llm_baseline, "reset_call", Mock())

    resolve_field_mock = AsyncMock(
        return_value=JevResolution(
            call_id="1", field_name="phone_number", question_type="noul", candidate="555-1111",
            confidence=0.9, is_committed=True, is_none_of_these=False,
            distinct_candidates=("555-1111",), input_tokens=10, output_tokens=2,
        )
    )
    monkeypatch.setattr(pipeline_core, "JevFieldResolver", lambda: FakeResolver(resolve_field_mock))

    store = FakeStore()
    on_tick_calls = []
    await pipeline_core.run_call(
        1,
        store,
        calls=_calls_fixture(),
        on_tick=lambda tick_number, total_ticks, committed: on_tick_calls.append(
            (tick_number, total_ticks, dict(committed))
        ),
    )

    # Fires once per tick, including the non-growth first tick.
    assert [call[0] for call in on_tick_calls] == [0, 1]
    # total_ticks (from CallPacer) is the same value on every call.
    assert on_tick_calls[0][1] == on_tick_calls[1][1]
    # Tick 0's snapshot has nothing committed yet.
    assert on_tick_calls[0][2] == {}
    # Tick 1's snapshot reflects the freshly committed value...
    assert on_tick_calls[1][2][("gliner_jev", "phone_number")] == ("555-1111", 0.9)
    # ...and each snapshot is its own copy, unaffected by state mutated on later ticks.
    assert on_tick_calls[0][2] == {}


@pytest.mark.asyncio
async def test_run_call_does_not_close_externally_provided_store(monkeypatch):
    monkeypatch.setattr(pipeline_core, "iter_batch_ticks", lambda call_pacer: iter([(0, "", 0)]))
    monkeypatch.setattr(pipeline_core.gliner_pipeline, "reset_call", Mock())
    monkeypatch.setattr(pipeline_core.llm_baseline, "reset_call", Mock())
    fake_resolver = FakeResolver(AsyncMock())
    monkeypatch.setattr(pipeline_core, "JevFieldResolver", lambda: fake_resolver)

    store = FakeStore()
    await pipeline_core.run_call(1, store, calls=_calls_fixture())

    assert store.closed is False
    fake_resolver.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_call_cleanup_steps_are_isolated_from_each_other(monkeypatch):
    # If gliner_pipeline.reset_call raises, llm_baseline.reset_call and resolver.aclose() must
    # still run rather than being skipped by the same finally block.
    monkeypatch.setattr(pipeline_core, "iter_batch_ticks", lambda call_pacer: iter([(0, "", 0)]))
    monkeypatch.setattr(
        pipeline_core.gliner_pipeline, "reset_call", Mock(side_effect=RuntimeError("session gone"))
    )
    llm_reset_mock = Mock()
    monkeypatch.setattr(pipeline_core.llm_baseline, "reset_call", llm_reset_mock)
    fake_resolver = FakeResolver(AsyncMock())
    monkeypatch.setattr(pipeline_core, "JevFieldResolver", lambda: fake_resolver)

    store = FakeStore()
    await pipeline_core.run_call(1, store, calls=_calls_fixture())  # must not raise

    llm_reset_mock.assert_called_once_with("1")
    fake_resolver.aclose.assert_awaited_once()


@pytest.mark.asyncio
async def test_run_call_gliner_failure_is_captured_as_error_row_not_raised(monkeypatch):
    # tick_number=1 (not a multiple of LLM_CADENCE_TICKS) so only the gliner+jev step fires.
    monkeypatch.setattr(pipeline_core, "iter_batch_ticks", lambda call_pacer: iter([(0, "", 0), (1, "x", 1)]))
    monkeypatch.setattr(
        pipeline_core.gliner_pipeline, "extract_candidates", AsyncMock(side_effect=RuntimeError("boom"))
    )
    monkeypatch.setattr(pipeline_core.gliner_pipeline, "reset_call", Mock())
    monkeypatch.setattr(pipeline_core.llm_baseline, "reset_call", Mock())
    fake_resolver = FakeResolver(AsyncMock())
    monkeypatch.setattr(pipeline_core, "JevFieldResolver", lambda: fake_resolver)

    store = FakeStore()
    await pipeline_core.run_call(1, store, calls=_calls_fixture())  # must not raise

    # Both gliner_jev stages get an error row -- extract_candidates runs the PII and zero-shot
    # models concurrently, so a single raised exception can't be attributed to just one of them.
    errored = [run for run in store.pipeline_runs if run["error"]]
    assert len(errored) == 2
    assert {row["stage"] for row in errored} == {"gliner_stream_pii", "gliner_standard"}
    assert all(row["pipeline"] == "gliner_jev" for row in errored)
    assert all("boom" in row["error"] for row in errored)
    fake_resolver.resolve_field.assert_not_called()
