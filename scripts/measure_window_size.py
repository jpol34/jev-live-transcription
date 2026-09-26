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

from jev_live_transcription import batch_runner, config, corpus, secrets  # noqa: E402
from jev_live_transcription import db as db_module  # noqa: E402
from measure_gliner_concurrency import STAGE, load_stage_latencies, print_table, summarize  # noqa: E402

DEFAULT_WINDOW_SIZES = (200, 400, 800)
DEFAULT_DB_DIR = Path("output/window_size_measurements")


def _format_recall(recall: float | None) -> str:
    return f"{recall:.0%}" if recall is not None else "n/a"


def print_recall_table(recall_by_window: dict[int, dict]) -> None:
    fields = score_recall.FIELDS
    print(f"\n{'window_chars':<14} " + " ".join(f"{field:<14}" for field in fields))
    for window_chars, stats in recall_by_window.items():
        row = " ".join(f"{_format_recall(stats[field]['recall']):<14}" for field in fields)
        print(f"{window_chars:<14} {row}")


async def _run_one_window_size(
    window_chars: int, *, call_ids: list[int], calls: dict, db_path: Path, warm_up: bool
) -> tuple[dict, dict]:
    original = config.GLINER_ZERO_SHOT_WINDOW_CHARS
    config.GLINER_ZERO_SHOT_WINDOW_CHARS = window_chars
    try:
        await batch_runner.run_batch(
            call_ids,
            db_path=db_path,
            call_concurrency=1,
            gliner_concurrency=1,
            calls=calls,
            warm_up=warm_up,
        )
    finally:
        config.GLINER_ZERO_SHOT_WINDOW_CHARS = original

    conn = db_module.connect(db_path)
    try:
        latency_stats = summarize(load_stage_latencies(conn, STAGE))
        ground_truths = score_recall.load_ground_truths(conn)
        committed = score_recall.load_final_committed_values(conn, "gliner_jev")
        recall_stats = score_recall.score_recall(ground_truths, committed)
    finally:
        conn.close()
    return latency_stats, recall_stats


async def measure(
    window_sizes: list[int], *, call_ids: list[int], calls: dict, db_dir: Path
) -> tuple[dict[int, dict], dict[int, dict]]:
    db_dir.mkdir(parents=True, exist_ok=True)
    latency_results = {}
    recall_results = {}
    for i, window_chars in enumerate(window_sizes):
        db_path = db_dir / f"window_{window_chars}.sqlite3"
        latency_stats, recall_stats = await _run_one_window_size(
            window_chars, call_ids=call_ids, calls=calls, db_path=db_path, warm_up=(i == 0)
        )
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

    secrets.load_typesafe_key()
    calls = corpus.load_all()
    call_ids = sorted(calls)
    if args.subset is not None:
        call_ids = call_ids[: args.subset]

    print(f"Measuring window sizes={window_sizes} over {len(call_ids)} call(s)...")
    latency_results, recall_results = asyncio.run(
        measure(window_sizes, call_ids=call_ids, calls=calls, db_dir=args.db_dir)
    )
    print_table(latency_results)
    print_recall_table(recall_results)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
