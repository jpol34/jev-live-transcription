"""Measures gliner_standard-stage latency across candidate `gliner_concurrency` levels.

Runs the corpus (or a subset) once per candidate `gliner_concurrency` level into its own capture
DB, holding `call_concurrency` fixed at 1 (the harness's real purpose is per-call latency, which
`batch_runner`'s own docstring already documents as invalid above `call_concurrency=1` regardless
of device), then prints a p50/p95 latency comparison table across the candidates.

This script only measures -- whether to actually raise `config.GLINER_CONCURRENCY` from its
default of 1 is a separate, deliberate step made from this printed data, per the plan's decision
to decide concurrency empirically rather than assume it.

Usage:
    uv run python scripts/measure_gliner_concurrency.py [--subset N] [--concurrencies 1,2,4]
        [--db-dir output/concurrency_measurements]
"""

import argparse
import asyncio
import sqlite3
import statistics
from pathlib import Path

from jev_live_transcription import batch_runner, corpus, secrets
from jev_live_transcription import db as db_module

DEFAULT_CONCURRENCIES = (1, 2, 4)
DEFAULT_DB_DIR = Path("output/concurrency_measurements")
STAGE = "gliner_standard"


def percentile(values: list[float], p: float) -> float | None:
    """Return the `p`th percentile (0-100) of `values` via nearest-rank, or `None` if empty."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1))))
    return ordered[rank]


def load_stage_latencies(conn: sqlite3.Connection, stage: str = STAGE) -> list[float]:
    """Return every non-error latency_ms recorded for `stage` in the gliner_jev pipeline."""
    rows = conn.execute(
        "SELECT latency_ms FROM pipeline_runs "
        "WHERE pipeline = 'gliner_jev' AND stage = ? AND error IS NULL AND latency_ms IS NOT NULL",
        (stage,),
    ).fetchall()
    return [row[0] for row in rows]


def summarize(latencies: list[float]) -> dict:
    return {
        "n": len(latencies),
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "mean": statistics.fmean(latencies) if latencies else None,
    }


def print_table(results: dict[int, dict]) -> None:
    print(f"{'gliner_concurrency':<20} {'n':<8} {'p50_ms':<10} {'p95_ms':<10} {'mean_ms':<10}")
    for concurrency, stats in results.items():

        def fmt(value):
            return f"{value:.1f}" if value is not None else "n/a"

        print(
            f"{concurrency:<20} {stats['n']:<8} {fmt(stats['p50']):<10} "
            f"{fmt(stats['p95']):<10} {fmt(stats['mean']):<10}"
        )


async def _run_one_concurrency(
    concurrency: int, *, call_ids: list[int], calls: dict, db_path: Path, warm_up: bool
) -> dict:
    await batch_runner.run_batch(
        call_ids,
        db_path=db_path,
        call_concurrency=1,
        gliner_concurrency=concurrency,
        calls=calls,
        warm_up=warm_up,
    )
    conn = db_module.connect(db_path)
    try:
        return summarize(load_stage_latencies(conn))
    finally:
        conn.close()


async def measure(
    concurrencies: list[int], *, call_ids: list[int], calls: dict, db_dir: Path
) -> dict[int, dict]:
    db_dir.mkdir(parents=True, exist_ok=True)
    results = {}
    for i, concurrency in enumerate(concurrencies):
        db_path = db_dir / f"gliner_concurrency_{concurrency}.sqlite3"
        # Only the first run pays GLiNER's one-time model-load cost -- every candidate shares one
        # warm, already-loaded process, so later candidates' latency numbers aren't inflated by a
        # cost real usage would only ever pay once per process lifetime.
        results[concurrency] = await _run_one_concurrency(
            concurrency, call_ids=call_ids, calls=calls, db_path=db_path, warm_up=(i == 0)
        )
    return results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset", type=int, default=None, help="Only run the first N calls (by call_id)."
    )
    parser.add_argument(
        "--concurrencies",
        type=str,
        default=",".join(str(c) for c in DEFAULT_CONCURRENCIES),
        help=f"Comma-separated gliner_concurrency candidates (default: {DEFAULT_CONCURRENCIES}).",
    )
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    args = parser.parse_args()

    concurrencies = [int(c) for c in args.concurrencies.split(",")]

    secrets.load_typesafe_key()
    calls = corpus.load_all()
    call_ids = sorted(calls)
    if args.subset is not None:
        call_ids = call_ids[: args.subset]

    print(f"Measuring gliner_concurrency={concurrencies} over {len(call_ids)} call(s)...")
    results = asyncio.run(measure(concurrencies, call_ids=call_ids, calls=calls, db_dir=args.db_dir))
    print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
