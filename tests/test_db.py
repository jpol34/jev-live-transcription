import json
import sqlite3
import threading

import pytest

from jev_live_transcription import db


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


def test_init_schema_creates_all_tables_in_wal_mode(tmp_path):
    db_path = tmp_path / "capture.db"

    db.init_schema(db_path)

    conn = db.connect(db_path)
    try:
        names = {
            row[0]
            for row in conn.execute(
                "SELECT name FROM sqlite_master WHERE type='table'"
            ).fetchall()
        }
        assert {"calls", "ticks", "pipeline_runs", "field_extractions"} <= names
        assert conn.execute("PRAGMA journal_mode").fetchone()[0].lower() == "wal"
    finally:
        conn.close()


def test_insert_call_blocks_until_committed(tmp_path):
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        call_id = store.insert_call(**_make_call_fields(1))
        assert call_id == 1

        conn = db.connect(store.db_path)
        try:
            row = conn.execute(
                "SELECT category FROM calls WHERE call_id = ?", (1,)
            ).fetchone()
        finally:
            conn.close()
        assert row is not None
        assert row[0] == "resident"
    finally:
        store.close()


def test_tick_insert_for_missing_call_id_raises_foreign_key_violation(tmp_path):
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        future = store.enqueue_tick(
            call_id=999,
            tick_number=0,
            wall_clock_ts=0.0,
            transcript_char_offset=0,
            transcript_snapshot="",
        )
        with pytest.raises(sqlite3.IntegrityError):
            future.result(timeout=5)

        conn = db.connect(store.db_path)
        try:
            count = conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0]
        finally:
            conn.close()
        assert count == 0
    finally:
        store.close()


def test_pipeline_run_insert_for_missing_call_id_raises_foreign_key_violation(tmp_path):
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        store.insert_call(**_make_call_fields(1))
        tick_id = store.enqueue_tick(
            call_id=1,
            tick_number=0,
            wall_clock_ts=0.0,
            transcript_char_offset=0,
            transcript_snapshot="hello",
        ).result(timeout=5)

        future = store.enqueue_pipeline_run(
            tick_id=tick_id,
            call_id=999,
            pipeline="gliner",
            stage="extract",
            latency_ms=12.5,
            input_tokens=None,
            cached_input_tokens=None,
            output_tokens=None,
            estimated_cost_usd=None,
            raw_output_json="{}",
            error=None,
        )
        with pytest.raises(sqlite3.IntegrityError):
            future.result(timeout=5)
    finally:
        store.close()


def test_full_row_chain_commits_successfully(tmp_path):
    store = db.CaptureStore(tmp_path / "capture.db")
    try:
        store.insert_call(**_make_call_fields(1))
        tick_id = store.enqueue_tick(
            call_id=1,
            tick_number=0,
            wall_clock_ts=1.0,
            transcript_char_offset=10,
            transcript_snapshot="Agent: Hello",
        ).result(timeout=5)
        run_id = store.enqueue_pipeline_run(
            tick_id=tick_id,
            call_id=1,
            pipeline="jev",
            stage="resolve",
            latency_ms=88.0,
            input_tokens=50,
            cached_input_tokens=0,
            output_tokens=5,
            estimated_cost_usd=0.0001,
            raw_output_json="{}",
            error=None,
        ).result(timeout=5)
        extraction_id = store.enqueue_field_extraction(
            run_id=run_id,
            call_id=1,
            tick_number=0,
            pipeline="jev",
            field_name="unit_number",
            candidate_value="A-101",
            confidence=0.9,
            is_committed=1,
        ).result(timeout=5)
        assert extraction_id is not None
    finally:
        store.close()

    conn = db.connect(store.db_path)
    try:
        assert conn.execute("SELECT COUNT(*) FROM ticks").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM pipeline_runs").fetchone()[0] == 1
        assert conn.execute("SELECT COUNT(*) FROM field_extractions").fetchone()[0] == 1
    finally:
        conn.close()


def test_concurrent_producers_no_lock_errors_and_no_lost_writes(tmp_path):
    store = db.CaptureStore(tmp_path / "capture.db")
    n_threads = 20
    ticks_per_thread = 25
    errors: list[Exception] = []
    errors_lock = threading.Lock()

    try:
        store.insert_call(**_make_call_fields(1))

        def producer(thread_index: int) -> None:
            try:
                futures = [
                    store.enqueue_tick(
                        call_id=1,
                        tick_number=thread_index * ticks_per_thread + i,
                        wall_clock_ts=float(i),
                        transcript_char_offset=i,
                        transcript_snapshot=f"snapshot-{thread_index}-{i}",
                    )
                    for i in range(ticks_per_thread)
                ]
                for future in futures:
                    future.result(timeout=10)
            except Exception as exc:  # noqa: BLE001 - surfaced via errors list
                with errors_lock:
                    errors.append(exc)

        threads = [threading.Thread(target=producer, args=(i,)) for i in range(n_threads)]
        for t in threads:
            t.start()
        for t in threads:
            t.join(timeout=30)

        assert not errors, f"concurrent producers raised: {errors}"

        conn = db.connect(store.db_path)
        try:
            count = conn.execute(
                "SELECT COUNT(*) FROM ticks WHERE call_id = ?", (1,)
            ).fetchone()[0]
        finally:
            conn.close()
        assert count == n_threads * ticks_per_thread
    finally:
        store.close()
