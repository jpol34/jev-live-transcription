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


def summarize(latencies: list[float], *, n_failed: int = 0) -> dict:
    return {
        "n": len(latencies),
        "n_failed": n_failed,
        "p50": percentile(latencies, 50),
        "p95": percentile(latencies, 95),
        "mean": statistics.fmean(latencies) if latencies else None,
    }


def _fmt(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "n/a"


def print_table(results: dict, *, label: str = "gliner_concurrency") -> None:
    print(f"{label:<20} {'n':<8} {'n_failed':<10} {'p50_ms':<10} {'p95_ms':<10} {'mean_ms':<10}")
    for candidate, stats in results.items():
        if stats is None:
            print(f"{candidate:<20} (candidate errored -- see warning above)")
            continue
        print(
            f"{candidate:<20} {stats['n']:<8} {stats['n_failed']:<10} {_fmt(stats['p50']):<10} "
            f"{_fmt(stats['p95']):<10} {_fmt(stats['mean']):<10}"
        )


def load_call_subset(subset: int | None) -> tuple[dict, list[int]]:
    """Load the corpus (sourcing the Strongbox-backed jev key first) and return
    `(calls, call_ids)`, `call_ids` trimmed to the first `subset` calls when given."""
    secrets.load_typesafe_key()
    calls = corpus.load_all()
    call_ids = sorted(calls)
    if subset is not None:
        call_ids = call_ids[:subset]
    return calls, call_ids


def _delete_db_file(db_path: Path) -> None:
    """Remove `db_path` and its WAL/SHM sidecar files, so each candidate starts from a fresh,
    empty capture DB -- otherwise a re-run against an existing `--db-dir` would collide with a
    prior run's `calls.call_id` primary keys and every insert would fail silently into
    `BatchResult.failed` instead of measuring anything."""
    for path in (db_path, db_path.with_name(db_path.name + "-wal"), db_path.with_name(db_path.name + "-shm")):
        path.unlink(missing_ok=True)


async def _run_one_concurrency(
    concurrency: int, *, call_ids: list[int], calls: dict, db_path: Path, warm_up: bool
) -> dict:
    _delete_db_file(db_path)
    result = await batch_runner.run_batch(
        call_ids,
        db_path=db_path,
        call_concurrency=1,
        gliner_concurrency=concurrency,
        calls=calls,
        warm_up=warm_up,
    )
    if result.failed:
        print(
            f"WARNING: gliner_concurrency={concurrency}: {len(result.failed)} call(s) failed "
            f"({[call_id for call_id, _exc in result.failed]}) -- latency stats below only "
            "reflect the calls that succeeded."
        )
    conn = db_module.connect(db_path)
    try:
        return summarize(load_stage_latencies(conn), n_failed=len(result.failed))
    finally:
        conn.close()


async def measure(
    concurrencies: list[int], *, call_ids: list[int], calls: dict, db_dir: Path
) -> dict[int, dict | None]:
    db_dir.mkdir(parents=True, exist_ok=True)
    results: dict[int, dict | None] = {}
    for i, concurrency in enumerate(concurrencies):
        db_path = db_dir / f"gliner_concurrency_{concurrency}.sqlite3"
        try:
            # Only the first run pays GLiNER's one-time model-load cost -- every candidate shares
            # one warm, already-loaded process, so later candidates' latency numbers aren't
            # inflated by a cost real usage would only ever pay once per process lifetime.
            results[concurrency] = await _run_one_concurrency(
                concurrency, call_ids=call_ids, calls=calls, db_path=db_path, warm_up=(i == 0)
            )
        except Exception as exc:  # noqa: BLE001 -- isolated per candidate, see docstring below
            # A bad later candidate must not discard every already-completed (expensive) prior
            # candidate's results -- record the failure and keep going instead of letting the
            # exception propagate out of the whole measurement loop.
            print(f"WARNING: gliner_concurrency={concurrency} errored: {exc}")
            results[concurrency] = None
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
    calls, call_ids = load_call_subset(args.subset)

    print(f"Measuring gliner_concurrency={concurrencies} over {len(call_ids)} call(s)...")
    results = asyncio.run(measure(concurrencies, call_ids=call_ids, calls=calls, db_dir=args.db_dir))
    print("\nFinal results:")
    print_table(results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
