import importlib.util
import sqlite3
import sys
from pathlib import Path
from unittest.mock import Mock

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
    assert stats == {"n": 0, "n_failed": 0, "p50": None, "p95": None, "mean": None}


def test_summarize_records_n_failed():
    stats = measure_gliner_concurrency.summarize([100.0], n_failed=2)
    assert stats["n_failed"] == 2


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
            ("jev", 999.0, None),  # different stage -- excluded
        ],
    )
    from jev_live_transcription import db as db_module

    conn = db_module.connect(db_path)
    try:
        latencies = measure_gliner_concurrency.load_stage_latencies(conn)
    finally:
        conn.close()
    assert sorted(latencies) == [100.0, 200.0]


def test_delete_db_file_removes_main_file_and_sidecars(tmp_path):
    db_path = tmp_path / "candidate.sqlite3"
    db_path.write_text("main")
    db_path.with_name(db_path.name + "-wal").write_text("wal")
    db_path.with_name(db_path.name + "-shm").write_text("shm")

    measure_gliner_concurrency._delete_db_file(db_path)

    assert not db_path.exists()
    assert not db_path.with_name(db_path.name + "-wal").exists()
    assert not db_path.with_name(db_path.name + "-shm").exists()


def test_delete_db_file_tolerates_missing_files(tmp_path):
    db_path = tmp_path / "does-not-exist.sqlite3"
    measure_gliner_concurrency._delete_db_file(db_path)  # must not raise


def test_measure_isolates_a_failing_candidate_and_keeps_prior_results(monkeypatch, tmp_path):
    import measure_gliner_concurrency as mgc

    async def fake_run_one_concurrency(concurrency, *, call_concurrency, call_ids, calls, db_path, warm_up):
        if concurrency == 2:
            raise RuntimeError("boom")
        return {"n": 1, "n_failed": 0, "p50": 100.0, "p95": 100.0, "mean": 100.0}

    monkeypatch.setattr(mgc, "_run_one_concurrency", fake_run_one_concurrency)

    results = __import__("asyncio").run(
        mgc.measure([1, 2, 4], call_ids=[1], calls={}, db_dir=tmp_path)
    )

    assert results[1] is not None
    assert results[2] is None  # errored candidate recorded as None, not dropped
    assert results[4] is not None  # later candidate still ran despite candidate 2's failure


def test_measure_uses_call_concurrency_at_least_as_high_as_the_largest_candidate(monkeypatch, tmp_path):
    # Regression guard: gliner_concurrency's semaphore is only ever contended by coroutines that
    # already hold a call_concurrency slot, so a candidate above call_concurrency would silently
    # never see real contention at its own configured capacity.
    import measure_gliner_concurrency as mgc

    monkeypatch.setattr(mgc.config, "CALL_CONCURRENCY", 3)
    captured_call_concurrency = []

    async def fake_run_batch(call_ids, *, db_path, call_concurrency, gliner_concurrency, calls, warm_up):
        captured_call_concurrency.append(call_concurrency)

        class _Result:
            failed = []

        return _Result()

    monkeypatch.setattr(mgc.batch_runner, "run_batch", fake_run_batch)
    monkeypatch.setattr(mgc, "load_stage_latencies", lambda conn, stage=mgc.STAGE: [])
    monkeypatch.setattr(mgc.db_module, "connect", lambda db_path: Mock(close=lambda: None))

    __import__("asyncio").run(mgc.measure([1, 2, 4], call_ids=[1], calls={}, db_dir=tmp_path))

    # CALL_CONCURRENCY=3 is below the largest candidate (4) -- every candidate must run at
    # call_concurrency=4 (max(3, 4)), not 3, or the gliner_concurrency=4 candidate is undertested.
    assert captured_call_concurrency == [4, 4, 4]


def test_load_call_subset_loads_secrets_and_trims_call_ids(monkeypatch):
    import measure_gliner_concurrency as mgc

    load_secret_mock = Mock()
    monkeypatch.setattr(mgc.secrets, "load_typesafe_key", load_secret_mock)
    monkeypatch.setattr(mgc.corpus, "load_all", lambda: {3: {}, 1: {}, 2: {}})

    calls, call_ids = mgc.load_call_subset(subset=2)

    load_secret_mock.assert_called_once()
    assert call_ids == [1, 2]
    assert calls == {3: {}, 1: {}, 2: {}}


def test_load_call_subset_no_subset_returns_all_sorted_ids(monkeypatch):
    import measure_gliner_concurrency as mgc

    monkeypatch.setattr(mgc.secrets, "load_typesafe_key", Mock())
    monkeypatch.setattr(mgc.corpus, "load_all", lambda: {3: {}, 1: {}, 2: {}})

    _calls, call_ids = mgc.load_call_subset(subset=None)

    assert call_ids == [1, 2, 3]
