"""SQLite capture store for a live benchmark run.

Schema: `calls` (one row per call, written once at capture time — including
`ground_truth_json`, which only a later scorer may read, never the live
pipeline), `ticks` (one row per simulated clock tick of a call), and
`pipeline_runs` / `field_extractions` (one row per pipeline invocation and
per field candidate it produced).

Many call-tasks write concurrently, but SQLite tolerates only one writer
connection at a time without lock contention. `CaptureStore` owns a single
background thread holding the sole write connection; concurrent producers
push rows onto a queue that the thread drains and commits in small batches.
`calls` rows are low-volume (one per call) and are depended on by every other
table via foreign key, so `insert_call` blocks until its row has committed
rather than just being queued and forgotten.
"""

import concurrent.futures
import logging
import queue
import sqlite3
import threading
from pathlib import Path

_LOGGER = logging.getLogger(__name__)

DEFAULT_BATCH_SIZE = 100
DEFAULT_BATCH_INTERVAL_SECONDS = 0.1

SCHEMA_SQL = """
CREATE TABLE IF NOT EXISTS calls (
    call_id INTEGER PRIMARY KEY,
    scenario_json TEXT NOT NULL,
    category TEXT NOT NULL,
    subtype TEXT NOT NULL,
    edge_case INTEGER NOT NULL,
    ground_truth_json TEXT NOT NULL,
    target_seconds REAL NOT NULL,
    full_transcript_word_count INTEGER NOT NULL
);

CREATE TABLE IF NOT EXISTS ticks (
    tick_id INTEGER PRIMARY KEY AUTOINCREMENT,
    call_id INTEGER NOT NULL REFERENCES calls(call_id),
    tick_number INTEGER NOT NULL,
    wall_clock_ts REAL NOT NULL,
    transcript_char_offset INTEGER NOT NULL,
    transcript_snapshot TEXT NOT NULL,
    UNIQUE (call_id, tick_number)
);
CREATE INDEX IF NOT EXISTS idx_ticks_call_id ON ticks(call_id);

CREATE TABLE IF NOT EXISTS pipeline_runs (
    run_id INTEGER PRIMARY KEY AUTOINCREMENT,
    tick_id INTEGER NOT NULL REFERENCES ticks(tick_id),
    call_id INTEGER NOT NULL REFERENCES calls(call_id),
    pipeline TEXT NOT NULL,
    stage TEXT,
    latency_ms REAL,
    input_tokens INTEGER,
    cached_input_tokens INTEGER,
    output_tokens INTEGER,
    estimated_cost_usd REAL,
    raw_output_json TEXT,
    error TEXT
);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_call_id ON pipeline_runs(call_id);
CREATE INDEX IF NOT EXISTS idx_pipeline_runs_tick_id ON pipeline_runs(tick_id);

CREATE TABLE IF NOT EXISTS field_extractions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    run_id INTEGER NOT NULL REFERENCES pipeline_runs(run_id),
    call_id INTEGER NOT NULL REFERENCES calls(call_id),
    tick_number INTEGER NOT NULL,
    pipeline TEXT NOT NULL,
    field_name TEXT NOT NULL,
    candidate_value TEXT,
    confidence REAL,
    is_committed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_field_extractions_call_id ON field_extractions(call_id);
CREATE INDEX IF NOT EXISTS idx_field_extractions_run_id ON field_extractions(run_id);
"""

_STOP = object()


def connect(db_path: str | Path) -> sqlite3.Connection:
    """Open a connection with foreign keys enforced, for read or ad hoc use."""
    conn = sqlite3.connect(db_path, check_same_thread=False)
    conn.execute("PRAGMA foreign_keys = ON")
    return conn


def init_schema(db_path: str | Path) -> None:
    """Create the schema (if absent) and enable WAL mode on `db_path`."""
    conn = connect(db_path)
    try:
        conn.execute("PRAGMA journal_mode = WAL")
        conn.executescript(SCHEMA_SQL)
        conn.commit()
    finally:
        conn.close()


class _WriteJob:
    __slots__ = ("table", "sql", "params", "future")

    def __init__(
        self, table: str, sql: str, params: dict, future: "concurrent.futures.Future"
    ):
        self.table = table
        self.sql = sql
        self.params = params
        self.future = future


class CaptureStore:
    """Owns the single write connection for a capture run's SQLite database.

    Call `close()` (or use as a context manager) when the run is done to stop
    the writer thread cleanly.
    """

    CALL_COLUMNS = (
        "call_id",
        "scenario_json",
        "category",
        "subtype",
        "edge_case",
        "ground_truth_json",
        "target_seconds",
        "full_transcript_word_count",
    )
    TICK_COLUMNS = (
        "call_id",
        "tick_number",
        "wall_clock_ts",
        "transcript_char_offset",
        "transcript_snapshot",
    )
    PIPELINE_RUN_COLUMNS = (
        "tick_id",
        "call_id",
        "pipeline",
        "stage",
        "latency_ms",
        "input_tokens",
        "cached_input_tokens",
        "output_tokens",
        "estimated_cost_usd",
        "raw_output_json",
        "error",
    )
    FIELD_EXTRACTION_COLUMNS = (
        "run_id",
        "call_id",
        "tick_number",
        "pipeline",
        "field_name",
        "candidate_value",
        "confidence",
        "is_committed",
    )

    # Maps table name to its insert columns, so the four enqueue_* methods
    # below share one implementation instead of each hand-rolling the same
    # two lines. Each tuple is still declared as its own class attribute
    # (rather than only living in this dict) so `test_db.py` can assert it
    # matches the table's real columns and catch drift from SCHEMA_SQL.
    _TABLE_COLUMNS = {
        "calls": CALL_COLUMNS,
        "ticks": TICK_COLUMNS,
        "pipeline_runs": PIPELINE_RUN_COLUMNS,
        "field_extractions": FIELD_EXTRACTION_COLUMNS,
    }

    def __init__(
        self,
        db_path: str | Path,
        *,
        batch_size: int = DEFAULT_BATCH_SIZE,
        batch_interval: float = DEFAULT_BATCH_INTERVAL_SECONDS,
    ):
        self.db_path = Path(db_path)
        init_schema(self.db_path)
        self._batch_size = batch_size
        self._batch_interval = batch_interval
        self._queue: "queue.Queue[_WriteJob | object]" = queue.Queue()
        # Guards _closed so close() and _enqueue() agree on whether a job is
        # queued strictly before or after the _STOP sentinel — otherwise a
        # write racing a close() could land behind _STOP and its future would
        # never be resolved.
        self._closed_lock = threading.Lock()
        self._closed = False
        self._thread = threading.Thread(
            target=self._run, name="capture-store-writer", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        # Owns the sole write connection; created here (not in __init__) so
        # it's only ever touched from this thread. connect() passes
        # check_same_thread=False, so sqlite3 itself won't stop another
        # thread from using conn — the single-writer invariant is just "no
        # other code holds a reference to it," enforced by convention here.
        conn = connect(self.db_path)
        try:
            stopping = False
            while not stopping:
                try:
                    job = self._queue.get(timeout=self._batch_interval)
                except queue.Empty:
                    continue
                batch = []
                if job is _STOP:
                    stopping = True
                else:
                    batch.append(job)
                    # Drain whatever else is already waiting, up to batch_size,
                    # so a burst of writes shares one commit instead of one each.
                    while len(batch) < self._batch_size:
                        try:
                            job = self._queue.get_nowait()
                        except queue.Empty:
                            break
                        if job is _STOP:
                            stopping = True
                            break
                        batch.append(job)
                if batch:
                    try:
                        self._commit_batch(conn, batch)
                    except Exception:
                        # Last-resort guard: _commit_batch already handles
                        # per-row and commit failures internally, but if
                        # something still escapes, the writer thread must
                        # keep running rather than die silently and strand
                        # every write queued after it.
                        _LOGGER.exception("capture store writer batch failed unexpectedly")
                        for failed_job in batch:
                            if not failed_job.future.done():
                                failed_job.future.set_exception(
                                    RuntimeError("capture store writer batch failed")
                                )
        finally:
            # WAL mode (`init_schema`) keeps recent commits in a `-wal` sidecar file until a
            # checkpoint folds them back into the main database file; SQLite normally does this
            # automatically as the last connection closes, but a tool that copies just the main
            # file elsewhere (e.g. `gpu_run._retrieve_db`'s scp, which never touches `-wal`/`-shm`)
            # cannot rely on that timing. An explicit checkpoint here, before the connection ever
            # closes, guarantees the main file alone is always complete and self-contained for
            # whatever reads it next.
            try:
                conn.execute("PRAGMA wal_checkpoint(TRUNCATE)")
            except sqlite3.Error:
                _LOGGER.exception("final WAL checkpoint failed; closing anyway")
            conn.close()

    @staticmethod
    def _commit_batch(conn: sqlite3.Connection, batch: list) -> None:
        # Resolve futures only after commit() returns — a job's row must be
        # durable and visible to other connections before its caller proceeds
        # to insert dependent rows, and before it counts as a completed write
        # under concurrent load.
        outcomes = []
        for job in batch:
            try:
                cur = conn.execute(job.sql, job.params)
                outcomes.append([job, cur.lastrowid, None])
            except Exception as exc:
                outcomes.append([job, None, exc])
        try:
            conn.commit()
        except Exception as commit_exc:
            try:
                conn.rollback()
            except Exception:
                pass
            # A failed commit rolls back every execute() in this batch, even
            # ones that succeeded individually — attach the commit failure to
            # any job that doesn't already have its own per-row exception.
            for outcome in outcomes:
                if outcome[2] is None:
                    outcome[2] = commit_exc
        for job, rowid, exc in outcomes:
            if exc is not None:
                _LOGGER.error("capture store write to %s failed: %s", job.table, exc)
                job.future.set_exception(exc)
            else:
                job.future.set_result(rowid)

    def _enqueue(self, table: str, sql: str, params: dict) -> "concurrent.futures.Future":
        future: concurrent.futures.Future = concurrent.futures.Future()
        with self._closed_lock:
            if self._closed:
                future.set_exception(RuntimeError("CaptureStore is closed"))
                return future
            self._queue.put(_WriteJob(table, sql, params, future))
        return future

    @staticmethod
    def _row_sql(table: str, columns: tuple) -> str:
        cols = ", ".join(columns)
        placeholders = ", ".join(f":{c}" for c in columns)
        return f"INSERT INTO {table} ({cols}) VALUES ({placeholders})"

    @staticmethod
    def _row(columns: tuple, fields: dict) -> dict:
        missing = [c for c in columns if c not in fields]
        if missing:
            raise TypeError(f"missing required column(s): {', '.join(missing)}")
        extra = [k for k in fields if k not in columns]
        if extra:
            raise TypeError(f"unknown column(s): {', '.join(extra)}")
        return fields

    def _enqueue_row(self, table: str, fields: dict) -> "concurrent.futures.Future":
        columns = self._TABLE_COLUMNS[table]
        row = self._row(columns, fields)
        return self._enqueue(table, self._row_sql(table, columns), row)

    def enqueue_call(self, **fields) -> "concurrent.futures.Future":
        return self._enqueue_row("calls", fields)

    def enqueue_tick(self, **fields) -> "concurrent.futures.Future":
        return self._enqueue_row("ticks", fields)

    def enqueue_pipeline_run(self, **fields) -> "concurrent.futures.Future":
        return self._enqueue_row("pipeline_runs", fields)

    def enqueue_field_extraction(self, **fields) -> "concurrent.futures.Future":
        return self._enqueue_row("field_extractions", fields)

    def insert_call(self, **fields) -> int:
        """Insert a `calls` row and block until it has committed.

        `ticks`/`pipeline_runs`/`field_extractions` for this call_id have a
        foreign key on it, so callers must know the row exists before
        enqueuing those — unlike the high-volume tables, `calls` gets one row
        per call, so blocking here costs nothing.
        """
        return self.enqueue_call(**fields).result()

    def close(self, timeout: float = 5.0) -> None:
        with self._closed_lock:
            if self._closed:
                return
            self._closed = True
            self._queue.put(_STOP)
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise RuntimeError(
                f"CaptureStore writer thread did not stop within {timeout}s; "
                "queued writes may not have been flushed"
            )

    def __enter__(self) -> "CaptureStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
