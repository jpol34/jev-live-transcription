"""Computes per-field recall for one pipeline in a capture DB against corpus ground truth.

For each of the 11 target fields, recall is the fraction of calls where the corpus's ground
truth has a non-null value for that field *and* the pipeline's final committed value for that
call matches it (case/whitespace-insensitive equality or substring containment either way, since
extracted values are free text rather than guaranteed to match the ground-truth string exactly).

Usage:
    uv run python scripts/score_recall.py <db_path> [--pipeline gliner_jev]
        [--save <snapshot.json>] [--baseline <snapshot.json>] [--tolerance 0.05]

`--save` writes this run's per-field recall to a JSON snapshot file, for a later `--baseline`
comparison (e.g. re-verifying GPU-run recall against a saved pre-migration CPU baseline).
`--baseline` loads a previously saved snapshot and diffs this run against it, exiting non-zero if
any field's recall dropped by more than `--tolerance`.
"""

import argparse
import json
import re
import sqlite3
from pathlib import Path

from jev_live_transcription import db
from jev_live_transcription.llm_baseline import FIELDS

DEFAULT_TOLERANCE = 0.05

_WHITESPACE_RE = re.compile(r"\s+")


def normalize(value: str) -> str:
    return _WHITESPACE_RE.sub(" ", value.strip().lower())


def is_match(extracted: str, truth: str) -> bool:
    """True if `extracted` and `truth` agree closely enough to count as a recall hit.

    Extracted values are free text produced by a model, not guaranteed to match the ground-truth
    string exactly (e.g. "the caller, Tim Barker" vs "Tim Barker") -- normalized equality or
    substring containment either way covers that without requiring an exact match.
    """
    norm_extracted, norm_truth = normalize(extracted), normalize(truth)
    if not norm_extracted or not norm_truth:
        return False
    return norm_extracted == norm_truth or norm_truth in norm_extracted or norm_extracted in norm_truth


def load_ground_truths(conn: sqlite3.Connection) -> dict[int, dict]:
    """Return `{call_id: ground_truth_dict}` for every call in the capture DB."""
    rows = conn.execute("SELECT call_id, ground_truth_json FROM calls").fetchall()
    return {call_id: json.loads(ground_truth_json) for call_id, ground_truth_json in rows}


def load_final_committed_values(conn: sqlite3.Connection, pipeline: str) -> dict[int, dict[str, str]]:
    """Return `{call_id: {field_name: value}}` -- each field's last committed value for `pipeline`.

    "Last" means the highest-`id` `field_extractions` row with `is_committed = 1` for that
    (call_id, field_name) -- `pipeline_core`'s carry-forward state means a field's commit only
    ever gets confirmed or superseded across ticks, never reverted while still committed, so the
    most recent committed row is that call's final answer.
    """
    rows = conn.execute(
        """
        SELECT fe.call_id, fe.field_name, fe.candidate_value
        FROM field_extractions fe
        WHERE fe.pipeline = ?
          AND fe.is_committed = 1
          AND fe.id = (
              SELECT MAX(id) FROM field_extractions
              WHERE pipeline = fe.pipeline
                AND call_id = fe.call_id
                AND field_name = fe.field_name
                AND is_committed = 1
          )
        """,
        (pipeline,),
    ).fetchall()
    committed: dict[int, dict[str, str]] = {}
    for call_id, field_name, candidate_value in rows:
        if candidate_value is not None:
            committed.setdefault(call_id, {})[field_name] = candidate_value
    return committed


def score_recall(
    ground_truths: dict[int, dict],
    committed: dict[int, dict[str, str]],
    fields: tuple[str, ...] = FIELDS,
) -> dict[str, dict]:
    """Return `{field_name: {"n_expected", "n_matched", "recall"}}` across every call.

    `recall` is `None` (rather than a divide-by-zero) for a field no call's ground truth ever
    discloses a value for.
    """
    stats: dict[str, dict] = {}
    for field_name in fields:
        expected_call_ids = [
            call_id
            for call_id, gt in ground_truths.items()
            if gt.get(field_name) not in (None, "")
        ]
        matched = sum(
            1
            for call_id in expected_call_ids
            if is_match(
                committed.get(call_id, {}).get(field_name, "") or "", ground_truths[call_id][field_name]
            )
        )
        n_expected = len(expected_call_ids)
        stats[field_name] = {
            "n_expected": n_expected,
            "n_matched": matched,
            "recall": (matched / n_expected) if n_expected else None,
        }
    return stats


def print_table(stats: dict[str, dict]) -> None:
    print(f"{'field':<22} {'matched/expected':<18} {'recall':<8}")
    for field_name, field_stats in stats.items():
        recall = field_stats["recall"]
        recall_str = f"{recall:.2%}" if recall is not None else "n/a"
        print(
            f"{field_name:<22} "
            f"{field_stats['n_matched']}/{field_stats['n_expected']:<15} "
            f"{recall_str:<8}"
        )


def diff_against_baseline(stats: dict[str, dict], baseline: dict[str, dict], tolerance: float) -> list[str]:
    """Return a list of human-readable regression messages, one per field whose recall dropped by
    more than `tolerance` relative to `baseline`. Empty if there's no regression past tolerance."""
    regressions = []
    for field_name, field_stats in stats.items():
        base_recall = baseline.get(field_name, {}).get("recall")
        recall = field_stats["recall"]
        if base_recall is None or recall is None:
            continue
        drop = base_recall - recall
        if drop > tolerance:
            regressions.append(
                f"{field_name}: recall dropped from {base_recall:.2%} to {recall:.2%} "
                f"(-{drop:.2%}, tolerance {tolerance:.2%})"
            )
    return regressions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path)
    parser.add_argument("--pipeline", default="gliner_jev", help="Pipeline to score (default: gliner_jev).")
    parser.add_argument("--save", type=Path, default=None, help="Write this run's recall table to a JSON snapshot.")
    parser.add_argument(
        "--baseline", type=Path, default=None, help="A previously saved snapshot to diff this run against."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=f"Max allowed per-field recall drop vs. --baseline before exiting non-zero (default: {DEFAULT_TOLERANCE}).",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        parser.error(f"database file not found: {args.db_path}")

    conn = db.connect(args.db_path)
    try:
        ground_truths = load_ground_truths(conn)
        committed = load_final_committed_values(conn, args.pipeline)
    finally:
        conn.close()

    stats = score_recall(ground_truths, committed)
    print_table(stats)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"\nSaved recall snapshot -> {args.save}")

    if args.baseline is not None:
        baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        regressions = diff_against_baseline(stats, baseline, args.tolerance)
        if regressions:
            print(f"\nRecall regressed past tolerance ({args.tolerance:.2%}) vs. {args.baseline}:")
            for message in regressions:
                print(f"  {message}")
            return 1
        print(f"\nNo recall regression past tolerance ({args.tolerance:.2%}) vs. {args.baseline}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
