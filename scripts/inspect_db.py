"""Ad hoc query helpers for eyeballing capture progress on a live run.

Usage:
    uv run python scripts/inspect_db.py <db_path> [--call-id ID] [--limit N]
"""

import argparse
import sqlite3
from pathlib import Path

from jev_live_transcription import db

TABLES = ("calls", "ticks", "pipeline_runs", "field_extractions")


def _connect(db_path: Path) -> sqlite3.Connection:
    conn = db.connect(db_path)
    conn.row_factory = sqlite3.Row
    return conn


def _positive_int(value: str) -> int:
    n = int(value)
    if n < 1:
        raise argparse.ArgumentTypeError("must be a positive integer")
    return n


def row_counts(conn: sqlite3.Connection) -> dict[str, int]:
    return {table: conn.execute(f"SELECT COUNT(*) FROM {table}").fetchone()[0] for table in TABLES}


def recent_ticks(conn: sqlite3.Connection, call_id: int, limit: int) -> list[sqlite3.Row]:
    return conn.execute(
        "SELECT tick_id, tick_number, wall_clock_ts, transcript_char_offset "
        "FROM ticks WHERE call_id = ? ORDER BY tick_number DESC LIMIT ?",
        (call_id, limit),
    ).fetchall()


def main() -> None:
    parser = argparse.ArgumentParser(description="Ad hoc capture-DB progress inspector.")
    parser.add_argument("db_path", type=Path)
    parser.add_argument(
        "--call-id", type=int, default=None, help="Show recent ticks for this call_id."
    )
    parser.add_argument(
        "--limit",
        type=_positive_int,
        default=10,
        help="Number of recent ticks to show (default 10).",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        parser.error(f"database file not found: {args.db_path}")

    conn = _connect(args.db_path)
    try:
        print("Row counts:")
        for table, count in row_counts(conn).items():
            print(f"  {table:<18} {count}")

        if args.call_id is not None:
            print(f"\nMost recent {args.limit} ticks for call_id={args.call_id}:")
            rows = recent_ticks(conn, args.call_id, args.limit)
            if not rows:
                print("  (none)")
            for row in rows:
                print(
                    f"  tick {row['tick_number']:>4}  "
                    f"t={row['wall_clock_ts']:.2f}s  "
                    f"offset={row['transcript_char_offset']}"
                )
    finally:
        conn.close()


if __name__ == "__main__":
    main()
