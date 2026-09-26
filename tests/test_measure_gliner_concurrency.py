import importlib.util
import sqlite3
import sys
from pathlib import Path

_SCRIPT_PATH = Path(__file__).resolve().parent.parent / "scripts" / "measure_gliner_concurrency.py"
_spec = importlib.util.spec_from_file_location("measure_gliner_concurrency", _SCRIPT_PATH)
measure_gliner_concurrency = importlib.util.module_from_spec(_spec)
sys.modules["measure_gliner_concurrency"] = measure_gliner_concurrency
_spec.loader.exec_module(measure_gliner_concurrency)


def test_percentile_empty_list_is_none():
    assert measure_gliner_concurrency.percentile([], 50) is None


def test_percentile_p50_of_odd_length():
    assert measure_gliner_concurrency.percentile([10, 30, 20], 50) == 20


def test_percentile_p95_biases_toward_high_end():
    values = list(range(1, 101))  # 1..100
    p95 = measure_gliner_concurrency.percentile(values, 95)
    assert p95 >= 95


def test_summarize_empty_latencies():
    stats = measure_gliner_concurrency.summarize([])
    assert stats == {"n": 0, "p50": None, "p95": None, "mean": None}


def test_summarize_computes_all_stats():
    stats = measure_gliner_concurrency.summarize([100.0, 200.0, 300.0])
    assert stats["n"] == 3
    assert stats["p50"] == 200.0
    assert stats["mean"] == 200.0


def _make_db_with_pipeline_runs(tmp_path, rows):
    from jev_live_transcription import db as db_module

    db_path = tmp_path / "test.sqlite3"
    db_module.init_schema(db_path)
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA foreign_keys = OFF")
    conn.execute(
        "INSERT INTO calls (call_id, scenario_json, category, subtype, edge_case, "
        "ground_truth_json, target_seconds, full_transcript_word_count) "
        "VALUES (1, '{}', 'prospect', 'x', 0, '{}', 10.0, 5)"
    )
    conn.execute(
        "INSERT INTO ticks (call_id, tick_number, wall_clock_ts, transcript_char_offset, "
        "transcript_snapshot) VALUES (1, 0, 0.0, 0, '')"
    )
    for stage, latency_ms, error in rows:
        conn.execute(
            "INSERT INTO pipeline_runs (tick_id, call_id, pipeline, stage, latency_ms, error) "
            "VALUES (1, 1, 'gliner_jev', ?, ?, ?)",
            (stage, latency_ms, error),
        )
    conn.commit()
    conn.close()
    return db_path


def test_load_stage_latencies_filters_by_stage_and_excludes_errors(tmp_path):
    db_path = _make_db_with_pipeline_runs(
        tmp_path,
        [
            ("gliner_standard", 100.0, None),
            ("gliner_standard", 200.0, None),
            ("gliner_standard", None, "boom"),  # error row -- excluded
            ("gliner_stream_pii", 999.0, None),  # different stage -- excluded
        ],
    )
    from jev_live_transcription import db as db_module

    conn = db_module.connect(db_path)
    try:
        latencies = measure_gliner_concurrency.load_stage_latencies(conn)
    finally:
        conn.close()
    assert sorted(latencies) == [100.0, 200.0]
