import importlib.util
import json
import sys
from pathlib import Path

import pytest

from jev_live_transcription import db

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "score_latency.py"
_spec = importlib.util.spec_from_file_location("score_latency", _SCRIPT_PATH)
score_latency = importlib.util.module_from_spec(_spec)
sys.modules["score_latency"] = score_latency
_spec.loader.exec_module(score_latency)


def _make_call_fields(call_id: int) -> dict:
    return dict(
        call_id=call_id,
        scenario_json=json.dumps({"id": call_id}),
        category="resident",
        subtype="test_subtype",
        edge_case=0,
        ground_truth_json=json.dumps({"caller_name": "Jamie Test"}),
        target_seconds=180.0,
        full_transcript_word_count=42,
    )


def _make_pipeline_run_fields(**overrides) -> dict:
    fields = dict(
        latency_ms=10.0,
        input_tokens=None,
        cached_input_tokens=None,
        output_tokens=None,
        estimated_cost_usd=None,
        raw_output_json=None,
        error=None,
    )
    fields.update(overrides)
    return fields


# -- percentile -----------------------------------------------------------------


def test_percentile_empty_is_none():
    assert score_latency.percentile([], 95) is None


def test_percentile_single_value():
    assert score_latency.percentile([42.0], 95) == 42.0


def test_percentile_p95_nearest_rank():
    values = list(range(1, 101))  # 1..100
    assert score_latency.percentile([float(v) for v in values], 95) == 95.0


# -- summarize --------------------------------------------------------------------


def test_summarize_empty_ticks_is_all_none():
    stats = score_latency.summarize({})
    assert stats == {
        "n_ticks": 0,
        "mean_ms": None,
        "median_ms": None,
        "p95_ms": None,
        "n_errored": 0,
        "error_rate": None,
    }


def test_summarize_computes_mean_median_p95_and_error_rate():
    ticks = {
        1: {"latency_ms": 100.0, "errored": False},
        2: {"latency_ms": 200.0, "errored": True},
        3: {"latency_ms": 300.0, "errored": False},
    }
    stats = score_latency.summarize(ticks)
    assert stats["n_ticks"] == 3
    assert stats["mean_ms"] == 200.0
    assert stats["median_ms"] == 200.0
    assert stats["n_errored"] == 1
    assert stats["error_rate"] == 1 / 3


# -- DB fixture helpers -------------------------------------------------------------


def _insert_gliner_standard_row(store, *, tick_id, call_id, latency_ms=50.0, error=None):
    """Insert the shared gliner_standard row (pipeline='gliner_jev') for one tick."""
    return store.enqueue_pipeline_run(
        **_make_pipeline_run_fields(
            tick_id=tick_id,
            call_id=call_id,
            pipeline="gliner_jev",
            stage="gliner_standard",
            latency_ms=latency_ms,
            error=error,
        )
    ).result(timeout=5)


def _insert_jev_row(store, *, tick_id, call_id, latency_ms=30.0, error=None):
    return store.enqueue_pipeline_run(
        **_make_pipeline_run_fields(
            tick_id=tick_id,
            call_id=call_id,
            pipeline="gliner_jev",
            stage="jev",
            latency_ms=latency_ms,
            error=error,
        )
    ).result(timeout=5)


def _insert_gliner_only_commit_row(store, *, tick_id, call_id, latency_ms=20.0, error=None):
    return store.enqueue_pipeline_run(
        **_make_pipeline_run_fields(
            tick_id=tick_id,
            call_id=call_id,
            pipeline="gliner_only",
            stage="gliner_only_commit",
            latency_ms=latency_ms,
            error=error,
        )
    ).result(timeout=5)


def _insert_llm_row(store, *, tick_id, call_id, latency_ms=500.0, error=None):
    return store.enqueue_pipeline_run(
        **_make_pipeline_run_fields(
            tick_id=tick_id,
            call_id=call_id,
            pipeline="llm",
            stage="llm",
            latency_ms=latency_ms,
            error=error,
        )
    ).result(timeout=5)


def _make_tick(store, *, call_id, tick_number, offset=10):
    return store.enqueue_tick(
        call_id=call_id,
        tick_number=tick_number,
        wall_clock_ts=float(tick_number),
        transcript_char_offset=offset,
        transcript_snapshot="Agent: Hello",
    ).result(timeout=5)


# -- per_tick_gliner_jev (DB integration) ------------------------------------------


def test_per_tick_gliner_jev_sums_standard_and_all_jev_field_rows_for_one_tick(tmp_path):
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        store.insert_call(**_make_call_fields(1))
        tick_id = _make_tick(store, call_id=1, tick_number=0)

        _insert_gliner_standard_row(store, tick_id=tick_id, call_id=1, latency_ms=50.0)
        _insert_jev_row(store, tick_id=tick_id, call_id=1, latency_ms=30.0)
        _insert_jev_row(store, tick_id=tick_id, call_id=1, latency_ms=40.0)  # a second field's jev call
    finally:
        store.close()

    conn = db.connect(store.db_path)
    try:
        ticks = score_latency.per_tick_gliner_jev(conn)
    finally:
        conn.close()

    assert ticks == {tick_id: {"latency_ms": 120.0, "errored": False}}


def test_per_tick_gliner_jev_marks_tick_errored_if_any_row_errored(tmp_path):
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        store.insert_call(**_make_call_fields(1))
        tick_id = _make_tick(store, call_id=1, tick_number=0)

        _insert_gliner_standard_row(store, tick_id=tick_id, call_id=1, latency_ms=50.0)
        _insert_jev_row(store, tick_id=tick_id, call_id=1, latency_ms=30.0, error="jev timeout")
    finally:
        store.close()

    conn = db.connect(store.db_path)
    try:
        ticks = score_latency.per_tick_gliner_jev(conn)
    finally:
        conn.close()

    # Latency is still summed (the failed call spent real wall-clock time) but flagged errored.
    assert ticks == {tick_id: {"latency_ms": 80.0, "errored": True}}


def test_per_tick_gliner_jev_ignores_other_pipelines_and_stages(tmp_path):
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        store.insert_call(**_make_call_fields(1))
        tick_id = _make_tick(store, call_id=1, tick_number=0)

        _insert_gliner_standard_row(store, tick_id=tick_id, call_id=1, latency_ms=50.0)
        _insert_gliner_only_commit_row(store, tick_id=tick_id, call_id=1, latency_ms=999.0)
        _insert_llm_row(store, tick_id=tick_id, call_id=1, latency_ms=999.0)
    finally:
        store.close()

    conn = db.connect(store.db_path)
    try:
        ticks = score_latency.per_tick_gliner_jev(conn)
    finally:
        conn.close()

    assert ticks == {tick_id: {"latency_ms": 50.0, "errored": False}}


# -- per_tick_gliner_only (the key correctness join) -------------------------------


def test_per_tick_gliner_only_joins_shared_standard_row_with_commit_row(tmp_path):
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        store.insert_call(**_make_call_fields(1))
        tick_id = _make_tick(store, call_id=1, tick_number=0)

        # Shared extraction row, recorded under gliner_jev -- not duplicated under gliner_only.
        _insert_gliner_standard_row(store, tick_id=tick_id, call_id=1, latency_ms=50.0)
        _insert_gliner_only_commit_row(store, tick_id=tick_id, call_id=1, latency_ms=20.0)
    finally:
        store.close()

    conn = db.connect(store.db_path)
    try:
        ticks = score_latency.per_tick_gliner_only(conn)
    finally:
        conn.close()

    # 70.0, not 20.0 -- omitting the shared row would undercount, per the issue's correctness bar.
    assert ticks == {tick_id: {"latency_ms": 70.0, "errored": False}}


def test_per_tick_gliner_only_does_not_double_count_jev_stage_rows(tmp_path):
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        store.insert_call(**_make_call_fields(1))
        tick_id = _make_tick(store, call_id=1, tick_number=0)

        _insert_gliner_standard_row(store, tick_id=tick_id, call_id=1, latency_ms=50.0)
        _insert_jev_row(store, tick_id=tick_id, call_id=1, latency_ms=999.0)  # gliner_jev's own arm
        _insert_gliner_only_commit_row(store, tick_id=tick_id, call_id=1, latency_ms=20.0)
    finally:
        store.close()

    conn = db.connect(store.db_path)
    try:
        ticks = score_latency.per_tick_gliner_only(conn)
    finally:
        conn.close()

    assert ticks == {tick_id: {"latency_ms": 70.0, "errored": False}}


def test_per_tick_gliner_only_errored_if_either_side_errored(tmp_path):
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        store.insert_call(**_make_call_fields(1))
        tick_id = _make_tick(store, call_id=1, tick_number=0)

        _insert_gliner_standard_row(store, tick_id=tick_id, call_id=1, latency_ms=50.0, error="gliner oom")
        _insert_gliner_only_commit_row(store, tick_id=tick_id, call_id=1, latency_ms=20.0)
    finally:
        store.close()

    conn = db.connect(store.db_path)
    try:
        ticks = score_latency.per_tick_gliner_only(conn)
    finally:
        conn.close()

    assert ticks == {tick_id: {"latency_ms": 70.0, "errored": True}}


def test_per_tick_gliner_only_excludes_commit_row_with_no_shared_standard_row(tmp_path, capsys):
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        store.insert_call(**_make_call_fields(1))
        tick_id = _make_tick(store, call_id=1, tick_number=0)

        # No matching gliner_standard row inserted for this tick_id -- shouldn't happen in a real
        # capture, but the join must not silently undercount if it does.
        _insert_gliner_only_commit_row(store, tick_id=tick_id, call_id=1, latency_ms=20.0)
    finally:
        store.close()

    conn = db.connect(store.db_path)
    try:
        ticks = score_latency.per_tick_gliner_only(conn)
    finally:
        conn.close()

    assert ticks == {}
    assert "no matching shared gliner_standard row" in capsys.readouterr().err


def test_per_tick_gliner_only_handles_multiple_ticks_across_a_call(tmp_path):
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        store.insert_call(**_make_call_fields(1))
        tick_id_0 = _make_tick(store, call_id=1, tick_number=0)
        tick_id_1 = _make_tick(store, call_id=1, tick_number=1)

        _insert_gliner_standard_row(store, tick_id=tick_id_0, call_id=1, latency_ms=50.0)
        _insert_gliner_only_commit_row(store, tick_id=tick_id_0, call_id=1, latency_ms=20.0)

        _insert_gliner_standard_row(store, tick_id=tick_id_1, call_id=1, latency_ms=55.0)
        _insert_gliner_only_commit_row(store, tick_id=tick_id_1, call_id=1, latency_ms=25.0)
    finally:
        store.close()

    conn = db.connect(store.db_path)
    try:
        ticks = score_latency.per_tick_gliner_only(conn)
    finally:
        conn.close()

    assert ticks == {
        tick_id_0: {"latency_ms": 70.0, "errored": False},
        tick_id_1: {"latency_ms": 80.0, "errored": False},
    }


# -- per_tick_llm -------------------------------------------------------------------


def test_per_tick_llm_is_self_contained(tmp_path):
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        store.insert_call(**_make_call_fields(1))
        tick_id = _make_tick(store, call_id=1, tick_number=0)

        _insert_llm_row(store, tick_id=tick_id, call_id=1, latency_ms=500.0)
        # Other pipelines' rows on the same tick must not leak into llm's total.
        _insert_gliner_standard_row(store, tick_id=tick_id, call_id=1, latency_ms=999.0)
    finally:
        store.close()

    conn = db.connect(store.db_path)
    try:
        ticks = score_latency.per_tick_llm(conn)
    finally:
        conn.close()

    assert ticks == {tick_id: {"latency_ms": 500.0, "errored": False}}


def test_per_tick_llm_errored_row(tmp_path):
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        store.insert_call(**_make_call_fields(1))
        tick_id = _make_tick(store, call_id=1, tick_number=0)

        _insert_llm_row(store, tick_id=tick_id, call_id=1, latency_ms=500.0, error="rate limited")
    finally:
        store.close()

    conn = db.connect(store.db_path)
    try:
        ticks = score_latency.per_tick_llm(conn)
    finally:
        conn.close()

    assert ticks == {tick_id: {"latency_ms": 500.0, "errored": True}}


# -- score_latency / --pipelines filter (full fixture across all three pipelines) ---


def _build_full_fixture_db(tmp_path) -> Path:
    """Build a capture DB with rows from gliner_jev, gliner_only, and llm across two ticks of one
    call, mirroring a real run_call with every arm enabled."""
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        store.insert_call(**_make_call_fields(1))

        tick_id_0 = _make_tick(store, call_id=1, tick_number=0)
        _insert_gliner_standard_row(store, tick_id=tick_id_0, call_id=1, latency_ms=50.0)
        _insert_jev_row(store, tick_id=tick_id_0, call_id=1, latency_ms=30.0)
        _insert_gliner_only_commit_row(store, tick_id=tick_id_0, call_id=1, latency_ms=20.0)
        _insert_llm_row(store, tick_id=tick_id_0, call_id=1, latency_ms=500.0)

        tick_id_1 = _make_tick(store, call_id=1, tick_number=1)
        _insert_gliner_standard_row(store, tick_id=tick_id_1, call_id=1, latency_ms=55.0)
        _insert_jev_row(store, tick_id=tick_id_1, call_id=1, latency_ms=35.0, error="jev timeout")
        _insert_gliner_only_commit_row(store, tick_id=tick_id_1, call_id=1, latency_ms=25.0)
        # llm only runs every LLM_CADENCE_TICKS ticks in a real call -- no llm row on tick 1.
    finally:
        store.close()
    return store.db_path


def test_score_latency_reports_all_three_pipelines_by_default(tmp_path):
    db_path = _build_full_fixture_db(tmp_path)

    conn = db.connect(db_path)
    try:
        stats = score_latency.score_latency(conn)
    finally:
        conn.close()

    assert set(stats) == {"gliner_jev", "gliner_only", "llm"}
    assert stats["gliner_jev"]["n_ticks"] == 2  # both ticks had gliner_standard + jev rows
    assert stats["gliner_jev"]["n_errored"] == 1  # tick 1's jev row errored
    assert stats["gliner_only"]["n_ticks"] == 2
    assert stats["gliner_only"]["n_errored"] == 0
    assert stats["llm"]["n_ticks"] == 1  # only tick 0 had an llm row
    assert stats["llm"]["n_errored"] == 0


def test_score_latency_pipelines_filter_restricts_reported_pipelines(tmp_path):
    db_path = _build_full_fixture_db(tmp_path)

    conn = db.connect(db_path)
    try:
        stats = score_latency.score_latency(conn, pipelines=("gliner_only",))
    finally:
        conn.close()

    assert set(stats) == {"gliner_only"}


def test_score_latency_unknown_pipeline_raises(tmp_path):
    db_path = tmp_path / "capture.db"
    db.init_schema(db_path)
    conn = db.connect(db_path)
    try:
        with pytest.raises(ValueError):
            score_latency.score_latency(conn, pipelines=("not_a_real_pipeline",))
    finally:
        conn.close()
