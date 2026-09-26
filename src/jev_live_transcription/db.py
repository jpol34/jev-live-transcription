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
import queue
import sqlite3
import threading
from pathlib import Path

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
    transcript_snapshot TEXT NOT NULL
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
    __slots__ = ("sql", "params", "future")

    def __init__(self, sql: str, params: dict, future: "concurrent.futures.Future"):
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
        self._thread = threading.Thread(
            target=self._run, name="capture-store-writer", daemon=True
        )
        self._thread.start()

    def _run(self) -> None:
        # Owns the sole write connection; created here so it is only ever
        # touched from this thread (sqlite3 connections default to
        # single-thread affinity).
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
                    self._commit_batch(conn, batch)
        finally:
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
                outcomes.append((job, cur.lastrowid, None))
            except Exception as exc:
                outcomes.append((job, None, exc))
        conn.commit()
        for job, rowid, exc in outcomes:
            if exc is not None:
                job.future.set_exception(exc)
            else:
                job.future.set_result(rowid)

    def _enqueue(self, sql: str, params: dict) -> "concurrent.futures.Future":
        future: concurrent.futures.Future = concurrent.futures.Future()
        self._queue.put(_WriteJob(sql, params, future))
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

    def enqueue_call(self, **fields) -> "concurrent.futures.Future":
        row = self._row(self.CALL_COLUMNS, fields)
        return self._enqueue(self._row_sql("calls", self.CALL_COLUMNS), row)

    def enqueue_tick(self, **fields) -> "concurrent.futures.Future":
        row = self._row(self.TICK_COLUMNS, fields)
        return self._enqueue(self._row_sql("ticks", self.TICK_COLUMNS), row)

    def enqueue_pipeline_run(self, **fields) -> "concurrent.futures.Future":
        row = self._row(self.PIPELINE_RUN_COLUMNS, fields)
        return self._enqueue(self._row_sql("pipeline_runs", self.PIPELINE_RUN_COLUMNS), row)

    def enqueue_field_extraction(self, **fields) -> "concurrent.futures.Future":
        row = self._row(self.FIELD_EXTRACTION_COLUMNS, fields)
        return self._enqueue(
            self._row_sql("field_extractions", self.FIELD_EXTRACTION_COLUMNS), row
        )

    def insert_call(self, **fields) -> int:
        """Insert a `calls` row and block until it has committed.

        `ticks`/`pipeline_runs`/`field_extractions` for this call_id have a
        foreign key on it, so callers must know the row exists before
        enqueuing those — unlike the high-volume tables, `calls` gets one row
        per call, so blocking here costs nothing.
        """
        return self.enqueue_call(**fields).result()

    def close(self, timeout: float = 5.0) -> None:
        self._queue.put(_STOP)
        self._thread.join(timeout=timeout)

    def __enter__(self) -> "CaptureStore":
        return self

    def __exit__(self, *exc_info) -> None:
        self.close()
