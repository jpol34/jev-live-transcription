"""Mocked tests for pipeline_core -- no real GLiNER/jev/OpenAI calls (kept fast/offline)."""

import concurrent.futures
from types import SimpleNamespace
from unittest.mock import AsyncMock, Mock

import pytest

from jev_live_transcription import pipeline_core
from jev_live_transcription.jev_pipeline import JevFieldResolver, JevResolution


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


def test_candidate_context_windows_slices_each_candidate_individually():
    snapshot = (
        "Agent: hi. Caller: my number is 555-1111. Some unrelated note here. "
        "Caller: actually it's 555-2222. Thanks for calling."
    )
    spans = pipeline_core._sentence_spans(snapshot)
    candidate_spans = [
        {
            "text": "555-1111",
            "start": snapshot.index("555-1111"),
            "end": snapshot.index("555-1111") + len("555-1111"),
        },
        {
            "text": "555-2222",
            "start": snapshot.index("555-2222"),
            "end": snapshot.index("555-2222") + len("555-2222"),
        },
    ]
    fallback = pipeline_core._field_context_window(snapshot, candidate_spans, spans=spans)

    windows = pipeline_core._candidate_context_windows(snapshot, candidate_spans, spans=spans)

    # Each candidate's own window covers only its own justifying text, not the other candidate's --
    # unlike `fallback`, which spans both since it's merged across every span found this tick.
    assert "555-1111" in windows["555-1111"]
    assert "555-2222" not in windows["555-1111"]
    assert "555-2222" in windows["555-2222"]
    assert "555-1111" not in windows["555-2222"]
    assert "555-1111" in fallback and "555-2222" in fallback


def test_candidate_context_windows_omits_spans_missing_offsets():
    # A span missing start/end is left out entirely rather than mapped to a merged fallback -- a
    # caller (JevFieldResolver) that caches this per candidate must never lock in an imprecise
    # substitute for a candidate whose offsets simply weren't available yet.
    snapshot = "Agent: hi. Caller: my number is 555-1111."
    candidate_spans = [{"text": "555-1111", "start": None, "end": None}]

    windows = pipeline_core._candidate_context_windows(snapshot, candidate_spans)

    assert windows == {}


def test_candidate_context_windows_dedupes_repeated_span_text():
    snapshot = "Agent: hi. Caller: 555-1111, that's 555-1111 again."
    windows = pipeline_core._candidate_context_windows(
        snapshot,
        [
            {"text": "555-1111", "start": 19, "end": 27},
            {"text": "555-1111", "start": 38, "end": 46},
        ],
    )
    assert len(windows) == 1


def test_candidate_context_windows_keeps_valid_span_after_offsetless_duplicate():
    # A duplicate-text span with missing offsets encountered before one with valid offsets must
    # not shadow the valid one -- the valid span's precise window should still be used.
    snapshot = "Agent: hi. Caller: my number is 555-1111 okay."
    start = snapshot.index("555-1111")
    windows = pipeline_core._candidate_context_windows(
        snapshot,
        [
            {"text": "555-1111", "start": None, "end": None},
            {"text": "555-1111", "start": start, "end": start + 8},
        ],
    )
    assert windows["555-1111"] == pipeline_core.context_window(snapshot, start, start + 8)


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

    gliner_candidates_effects = [
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
    # Distinct latency per tick -- proves each pipeline_runs row gets its own timing rather than
    # sharing one identical number.
    gliner_side_effects = [
        (candidates, 11.0 + i) for i, candidates in enumerate(gliner_candidates_effects)
    ]
    extract_candidates_timed_mock = AsyncMock(side_effect=gliner_side_effects)
    monkeypatch.setattr(
        pipeline_core.gliner_pipeline, "extract_candidates_timed", extract_candidates_timed_mock
    )
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
    await pipeline_core.run_call(1, store, calls=_calls_fixture(), enable_llm_baseline=True)

    # --- snapshot-growth gating ---
    assert extract_candidates_timed_mock.await_count == 4  # ticks 1, 3, 5, 6
    assert len(store.ticks) == 7  # every tick gets a ticks row regardless of growth

    # --- GLiNER latency: each tick's pipeline_runs row gets its own timing ---
    gliner_stage_rows = [run for run in store.pipeline_runs if run["pipeline"] == "gliner_jev" and not run["error"]]
    gliner_latencies = [run["latency_ms"] for run in gliner_stage_rows if run["stage"] == "gliner_standard"]
    assert len(gliner_latencies) == 4
    assert len(set(gliner_latencies)) == 4

    # --- jev latency: every jev pipeline_runs row records a real latency, not NULL ---
    jev_rows = [run for run in store.pipeline_runs if run["stage"] == "jev"]
    assert jev_rows  # sanity: this scenario does exercise jev
    assert all(run["latency_ms"] is not None for run in jev_rows)

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


# --- jev context caching survives a candidate aging out of GLiNER's per-tick report -------------


@pytest.mark.asyncio
async def test_run_call_jev_choice_context_covers_candidate_that_stopped_being_reported(monkeypatch):
    # Simulates ticket #19's GLiNER windowing: tick 1 reports "555-3212" (with its own span) and
    # never again -- by tick 6, only the genuinely new "555-4321" is reported that tick, as if
    # "555-3212"'s original mention has aged out of GLiNER's bounded input. The real
    # JevFieldResolver (only its HTTP client is faked) must still send jev a Choice-call context
    # covering both candidates' own justifying text, from what pipeline_core cached the first time
    # "555-3212" was detected.
    snapshot_tick_1 = "Caller: my number is 555-3212 okay."
    snapshot_tick_6 = "Caller: actually it's 555-4321 now."
    start_1 = snapshot_tick_1.index("555-3212")
    start_6 = snapshot_tick_6.index("555-4321")
    gliner_side_effects = [
        (
            {"phone_number": [{"text": "555-3212", "score": 0.9, "start": start_1, "end": start_1 + 8}]},
            10.0,
        ),
        (
            # "555-3212" is no longer reported at all this tick -- only the new candidate is.
            {"phone_number": [{"text": "555-4321", "score": 0.9, "start": start_6, "end": start_6 + 8}]},
            11.0,
        ),
    ]

    async def fake_extract_candidates_timed(snapshot, call_id):
        return gliner_side_effects.pop(0)

    monkeypatch.setattr(
        pipeline_core.gliner_pipeline, "extract_candidates_timed", fake_extract_candidates_timed
    )
    monkeypatch.setattr(pipeline_core.gliner_pipeline, "reset_call", Mock())
    monkeypatch.setattr(pipeline_core.llm_baseline, "reset_call", Mock())

    # Hand run_call a real JevFieldResolver so its own dedup/context caching runs for real --
    # only the underlying typesafe.ai HTTP client is faked.
    system_one = AsyncMock(
        side_effect=[
            SimpleNamespace(
                nouls={"field": SimpleNamespace(noul=0.9)},
                usage=SimpleNamespace(input_tokens=10, output_tokens=2),
            ),
            SimpleNamespace(
                choices={"field": SimpleNamespace(choice="555-4321", confidence=0.85)},
                usage=SimpleNamespace(input_tokens=15, output_tokens=3),
            ),
        ]
    )
    fake_client = SimpleNamespace(system_one=system_one, aclose=AsyncMock())
    # settle_ticks=1 so the Choice call fires on the very first tick the 2-candidate set is
    # observed -- this test is about context caching, not SettleGate's own settling behavior
    # (covered separately in tests/test_jev_pipeline.py).
    real_resolver = JevFieldResolver(client=fake_client, settle_ticks=1)
    monkeypatch.setattr(pipeline_core, "JevFieldResolver", lambda: real_resolver)

    monkeypatch.setattr(
        pipeline_core,
        "iter_batch_ticks",
        lambda call_pacer: iter([(0, "", 0), (1, snapshot_tick_1, 1), (6, snapshot_tick_6, 2)]),
    )

    store = FakeStore()
    await pipeline_core.run_call(1, store, calls=_calls_fixture())

    assert system_one.await_count == 2
    _, choice_kwargs = system_one.call_args
    combined_context = choice_kwargs["state"]["context_window"]
    assert "555-3212" in combined_context
    assert "555-4321" in combined_context


# --- on_tick callback ------------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_call_on_tick_callback_fires_every_tick_with_committed_snapshot(monkeypatch):
    ticks_script = [
        (0, "", 0),  # no growth -> on_tick still fires, snapshot is empty
        (1, "a", 1),  # growth -> gliner+jev commits phone_number
    ]
    monkeypatch.setattr(pipeline_core, "iter_batch_ticks", lambda call_pacer: iter(ticks_script))

    extract_candidates_timed_mock = AsyncMock(
        return_value=(
            {"phone_number": [{"text": "555-1111", "score": 0.9, "start": 0, "end": 8}]},
            10.0,
        )
    )
    monkeypatch.setattr(
        pipeline_core.gliner_pipeline, "extract_candidates_timed", extract_candidates_timed_mock
    )
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
        on_tick=lambda tick_number, total_ticks, snapshot, committed: on_tick_calls.append(
            (tick_number, total_ticks, snapshot, dict(committed))
        ),
    )

    # Fires once per tick, including the non-growth first tick.
    assert [call[0] for call in on_tick_calls] == [0, 1]
    # total_ticks (from CallPacer) is the same value on every call.
    assert on_tick_calls[0][1] == on_tick_calls[1][1]
    # Each call's own transcript snapshot is passed through as-is.
    assert on_tick_calls[0][2] == ""
    assert on_tick_calls[1][2] == "a"
    # Tick 0's snapshot has nothing committed yet.
    assert on_tick_calls[0][3] == {}
    # Tick 1's snapshot reflects the freshly committed value...
    assert on_tick_calls[1][3][("gliner_jev", "phone_number")] == ("555-1111", 0.9)
    # ...and each snapshot is its own copy, unaffected by state mutated on later ticks.
    assert on_tick_calls[0][3] == {}


# --- confident-rejection/confident-null carry-forward clearing, at the orchestration level -------


@pytest.mark.asyncio
async def test_run_call_confident_none_of_these_clears_committed_value(monkeypatch):
    # Tick 1 commits "555-1111" for phone_number via jev. Tick 2's jev result is a confident
    # "none of these" rejection (is_none_of_these=True, is_committed=True) -- this must clear the
    # previously-committed value (persisted as candidate_value=None, is_committed=0), not hold
    # "555-1111" steady the way a below-threshold dip would.
    monkeypatch.setattr(
        pipeline_core,
        "iter_batch_ticks",
        lambda call_pacer: iter([(0, "", 0), (1, "a", 1), (2, "ab", 2)]),
    )
    gliner_candidates = {"phone_number": [{"text": "555-1111", "score": 0.9, "start": 0, "end": 8}]}
    extract_candidates_timed_mock = AsyncMock(
        side_effect=[(gliner_candidates, 10.0), (gliner_candidates, 11.0)]
    )
    monkeypatch.setattr(
        pipeline_core.gliner_pipeline, "extract_candidates_timed", extract_candidates_timed_mock
    )
    monkeypatch.setattr(pipeline_core.gliner_pipeline, "reset_call", Mock())
    monkeypatch.setattr(pipeline_core.llm_baseline, "reset_call", Mock())

    jev_results = [
        JevResolution(
            call_id="1", field_name="phone_number", question_type="noul", candidate="555-1111",
            confidence=0.9, is_committed=True, is_none_of_these=False,
            distinct_candidates=("555-1111",), input_tokens=10, output_tokens=2,
        ),
        JevResolution(
            call_id="1", field_name="phone_number", question_type="noul", candidate="none_of_these",
            confidence=0.9, is_committed=True, is_none_of_these=True,
            distinct_candidates=("555-1111",), input_tokens=10, output_tokens=2,
        ),
    ]
    resolve_field_mock = AsyncMock(side_effect=jev_results)
    monkeypatch.setattr(pipeline_core, "JevFieldResolver", lambda: FakeResolver(resolve_field_mock))

    store = FakeStore()
    await pipeline_core.run_call(1, store, calls=_calls_fixture())

    phone_rows = [
        fe
        for fe in store.field_extractions
        if fe["pipeline"] == "gliner_jev" and fe["field_name"] == "phone_number"
    ]
    assert len(phone_rows) == 2
    assert phone_rows[0]["candidate_value"] == "555-1111"
    assert phone_rows[0]["is_committed"] == 1
    # Confident rejection clears the committed value -- not held steady at "555-1111".
    assert phone_rows[1]["candidate_value"] is None
    assert phone_rows[1]["is_committed"] == 0


@pytest.mark.asyncio
async def test_run_call_llm_confident_null_clears_committed_value(monkeypatch):
    # LLM cadence tick 3 commits "Tim" for caller_name. Tick 6's LLM result confidently reports no
    # value at all (value=None, is_committed=True) -- this must clear the previously-committed
    # value, not hold "Tim" steady.
    monkeypatch.setattr(
        pipeline_core,
        "iter_batch_ticks",
        lambda call_pacer: iter([(0, "", 0), (3, "abc", 3), (6, "abcdef", 6)]),
    )
    monkeypatch.setattr(
        pipeline_core.gliner_pipeline,
        "extract_candidates_timed",
        AsyncMock(return_value=({}, 1.0)),
    )
    monkeypatch.setattr(pipeline_core.gliner_pipeline, "reset_call", Mock())
    monkeypatch.setattr(pipeline_core.llm_baseline, "reset_call", Mock())
    fake_resolver = FakeResolver(AsyncMock())
    monkeypatch.setattr(pipeline_core, "JevFieldResolver", lambda: fake_resolver)

    llm_side_effects = [
        _all_llm_fields(caller_name=("Tim", 0.9, True)),
        _all_llm_fields(caller_name=(None, 0.9, True)),  # confident null -> must clear
    ]
    llm_extract_mock = AsyncMock(side_effect=llm_side_effects)
    monkeypatch.setattr(pipeline_core.llm_baseline, "extract", llm_extract_mock)

    store = FakeStore()
    await pipeline_core.run_call(1, store, calls=_calls_fixture(), enable_llm_baseline=True)

    caller_name_rows = [
        fe
        for fe in store.field_extractions
        if fe["pipeline"] == "llm" and fe["field_name"] == "caller_name"
    ]
    assert len(caller_name_rows) == 2
    assert caller_name_rows[0]["candidate_value"] == "Tim"
    assert caller_name_rows[0]["is_committed"] == 1
    # Confident null clears the committed value -- not held steady at "Tim".
    assert caller_name_rows[1]["candidate_value"] is None
    assert caller_name_rows[1]["is_committed"] == 0
    assert caller_name_rows[1]["confidence"] == 0.9


# --- enable_llm_baseline gate --------------------------------------------------------------


@pytest.mark.asyncio
async def test_run_call_skips_llm_baseline_by_default(monkeypatch):
    # tick 3 is an LLM-cadence tick (3 % LLM_CADENCE_TICKS == 0) -- if the LLM step ran, it would
    # fire here. enable_llm_baseline is omitted (defaults to False), so it must not run at all:
    # no llm_baseline.extract call, no "llm" pipeline_runs row.
    monkeypatch.setattr(
        pipeline_core, "iter_batch_ticks", lambda call_pacer: iter([(0, "", 0), (3, "abc", 3)])
    )
    monkeypatch.setattr(
        pipeline_core.gliner_pipeline,
        "extract_candidates_timed",
        AsyncMock(return_value=({}, 1.0)),
    )
    monkeypatch.setattr(pipeline_core.gliner_pipeline, "reset_call", Mock())
    llm_extract_mock = AsyncMock()
    monkeypatch.setattr(pipeline_core.llm_baseline, "extract", llm_extract_mock)
    monkeypatch.setattr(pipeline_core.llm_baseline, "reset_call", Mock())
    fake_resolver = FakeResolver(AsyncMock())
    monkeypatch.setattr(pipeline_core, "JevFieldResolver", lambda: fake_resolver)

    store = FakeStore()
    await pipeline_core.run_call(1, store, calls=_calls_fixture())  # enable_llm_baseline omitted

    llm_extract_mock.assert_not_called()
    assert not any(run["pipeline"] == "llm" for run in store.pipeline_runs)


@pytest.mark.asyncio
async def test_run_call_runs_llm_baseline_when_explicitly_enabled(monkeypatch):
    monkeypatch.setattr(
        pipeline_core, "iter_batch_ticks", lambda call_pacer: iter([(0, "", 0), (3, "abc", 3)])
    )
    monkeypatch.setattr(
        pipeline_core.gliner_pipeline,
        "extract_candidates_timed",
        AsyncMock(return_value=({}, 1.0)),
    )
    monkeypatch.setattr(pipeline_core.gliner_pipeline, "reset_call", Mock())
    llm_extract_mock = AsyncMock(return_value=_all_llm_fields())
    monkeypatch.setattr(pipeline_core.llm_baseline, "extract", llm_extract_mock)
    monkeypatch.setattr(pipeline_core.llm_baseline, "reset_call", Mock())
    fake_resolver = FakeResolver(AsyncMock())
    monkeypatch.setattr(pipeline_core, "JevFieldResolver", lambda: fake_resolver)

    store = FakeStore()
    await pipeline_core.run_call(1, store, calls=_calls_fixture(), enable_llm_baseline=True)

    llm_extract_mock.assert_awaited_once()
    assert any(run["pipeline"] == "llm" for run in store.pipeline_runs)


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
        pipeline_core.gliner_pipeline,
        "extract_candidates_timed",
        AsyncMock(side_effect=RuntimeError("boom")),
    )
    monkeypatch.setattr(pipeline_core.gliner_pipeline, "reset_call", Mock())
    monkeypatch.setattr(pipeline_core.llm_baseline, "reset_call", Mock())
    fake_resolver = FakeResolver(AsyncMock())
    monkeypatch.setattr(pipeline_core, "JevFieldResolver", lambda: fake_resolver)

    store = FakeStore()
    await pipeline_core.run_call(1, store, calls=_calls_fixture())  # must not raise

    errored = [run for run in store.pipeline_runs if run["error"]]
    assert len(errored) == 1
    assert errored[0]["stage"] == "gliner_standard"
    assert errored[0]["pipeline"] == "gliner_jev"
    assert "boom" in errored[0]["error"]
    fake_resolver.resolve_field.assert_not_called()
