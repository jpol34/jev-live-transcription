"""Measures gliner_standard-stage latency and recall across candidate zero-shot window sizes.

Runs the corpus (or a subset) once per candidate `config.GLINER_ZERO_SHOT_WINDOW_CHARS` value into
its own capture DB, then prints a p50/p95 latency + per-corpus recall comparison table across the
candidates. Ticket #19's 200-char window was tuned for CPU's latency-scales-with-length behavior;
this script only measures whether a wider window is (near-)free on a different device -- the
actual `config.GLINER_ZERO_SHOT_WINDOW_CHARS` change (if any) is a separate, deliberate step made
from this printed data.

Usage:
    uv run python scripts/measure_window_size.py [--subset N] [--window-sizes 200,400,800]
        [--db-dir output/window_size_measurements]
"""

import argparse
import asyncio
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import score_recall  # noqa: E402 -- must follow the sys.path insert above

from jev_live_transcription import batch_runner, config  # noqa: E402
from jev_live_transcription import db as db_module  # noqa: E402
from measure_gliner_concurrency import (  # noqa: E402
    STAGE,
    _delete_db_file,
    load_call_subset,
    load_stage_latencies,
    print_table,
    summarize,
)

DEFAULT_WINDOW_SIZES = (200, 400, 800)
DEFAULT_DB_DIR = Path("output/window_size_measurements")


def _format_recall(recall: float | None) -> str:
    return f"{recall:.0%}" if recall is not None else "n/a"


def print_recall_table(recall_by_window: dict[int, dict | None]) -> None:
    fields = score_recall.FIELDS
    print(f"\n{'window_chars':<14} " + " ".join(f"{field:<14}" for field in fields))
    for window_chars, stats in recall_by_window.items():
        if stats is None:
            print(f"{window_chars:<14} (candidate errored -- see warning above)")
            continue
        row = " ".join(f"{_format_recall(stats[field]['recall']):<14}" for field in fields)
        print(f"{window_chars:<14} {row}")


async def _run_one_window_size(
    window_chars: int, *, call_ids: list[int], calls: dict, db_path: Path, warm_up: bool
) -> tuple[dict, dict]:
    _delete_db_file(db_path)
    original = config.GLINER_ZERO_SHOT_WINDOW_CHARS
    config.GLINER_ZERO_SHOT_WINDOW_CHARS = window_chars
    try:
        result = await batch_runner.run_batch(
            call_ids,
            db_path=db_path,
            call_concurrency=1,
            gliner_concurrency=1,
            calls=calls,
            warm_up=warm_up,
        )
    finally:
        config.GLINER_ZERO_SHOT_WINDOW_CHARS = original

    if result.failed:
        print(
            f"WARNING: window_chars={window_chars}: {len(result.failed)} call(s) failed "
            f"({[call_id for call_id, _exc in result.failed]}) -- latency/recall stats below "
            "only reflect the calls that succeeded."
        )

    conn = db_module.connect(db_path)
    try:
        latency_stats = summarize(load_stage_latencies(conn, STAGE), n_failed=len(result.failed))
        ground_truths = score_recall.load_ground_truths(conn)
        committed = score_recall.load_final_committed_values(conn, "gliner_jev")
        recall_stats = score_recall.score_recall(ground_truths, committed)
    finally:
        conn.close()
    return latency_stats, recall_stats


async def measure(
    window_sizes: list[int], *, call_ids: list[int], calls: dict, db_dir: Path
) -> tuple[dict[int, dict | None], dict[int, dict | None]]:
    db_dir.mkdir(parents=True, exist_ok=True)
    latency_results: dict[int, dict | None] = {}
    recall_results: dict[int, dict | None] = {}
    for i, window_chars in enumerate(window_sizes):
        db_path = db_dir / f"window_{window_chars}.sqlite3"
        try:
            latency_stats, recall_stats = await _run_one_window_size(
                window_chars, call_ids=call_ids, calls=calls, db_path=db_path, warm_up=(i == 0)
            )
        except Exception as exc:  # noqa: BLE001 -- isolated per candidate, see measure_gliner_concurrency
            print(f"WARNING: window_chars={window_chars} errored: {exc}")
            latency_stats, recall_stats = None, None
        latency_results[window_chars] = latency_stats
        recall_results[window_chars] = recall_stats
    return latency_results, recall_results


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--subset", type=int, default=None, help="Only run the first N calls (by call_id)."
    )
    parser.add_argument(
        "--window-sizes",
        type=str,
        default=",".join(str(w) for w in DEFAULT_WINDOW_SIZES),
        help=f"Comma-separated window-size candidates, in chars (default: {DEFAULT_WINDOW_SIZES}).",
    )
    parser.add_argument("--db-dir", type=Path, default=DEFAULT_DB_DIR)
    args = parser.parse_args()

    window_sizes = [int(w) for w in args.window_sizes.split(",")]
    calls, call_ids = load_call_subset(args.subset)

    print(f"Measuring window sizes={window_sizes} over {len(call_ids)} call(s)...")
    latency_results, recall_results = asyncio.run(
        measure(window_sizes, call_ids=call_ids, calls=calls, db_dir=args.db_dir)
    )
    print("\nFinal results:")
    print_table(latency_results, label="window_chars")
    print_recall_table(recall_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
