"""Computes per-field recall, precision, and jev call volume for one pipeline in a capture DB
against corpus ground truth.

For each of the 11 target fields, recall is the fraction of calls where the corpus's ground
truth has a non-null value for that field *and* the pipeline's final committed value for that
call matches it (case/whitespace-insensitive equality or substring containment either way, since
extracted values are free text rather than guaranteed to match the ground-truth string exactly).
A field whose ground truth is a list (e.g. `amenities_requested` can disclose several amenities)
counts as matched if the committed value agrees with any one item in that list.

Precision is the fraction of the pipeline's committed values for a field that are correct. Per
(call, field): a committed value against non-null ground truth that doesn't match is
simultaneously a recall miss and a precision false positive; a committed value against null
ground truth (nothing to have extracted) is always a precision false positive; a list-valued
ground truth counts as a true positive if the committed value matches any item in the list,
mirroring recall's treatment of that case.

Jev call volume snapshots, per field, how much of the "jev" resolver's disambiguation work
(`jev_pipeline.JevFieldResolver`) each field costs across the corpus: the number of jev calls
that were `Choice` (multi-candidate) questions, versus cheaper single-candidate `Noul` questions,
and the average/max size of the distinct-candidate set jev had to disambiguate between. This is
tracked because loosening a field's confidence floor can inflate its candidate sets and trigger
more (and larger) Choice calls without necessarily changing recall/precision at all.

Usage:
    uv run python scripts/score_recall.py <db_path> [--pipeline gliner_jev]
        [--save <snapshot.json>] [--baseline <snapshot.json>] [--tolerance 0.05]

`--save` writes this run's per-field stats (recall, precision, jev call volume) to a JSON
snapshot file, for a later `--baseline` comparison (e.g. re-verifying GPU-run recall against a
saved pre-migration CPU baseline). `--baseline` loads a previously saved snapshot and diffs this
run against it, exiting non-zero if any field's recall or precision dropped by more than
`--tolerance`, or its jev Choice-call volume rose by more than `--tolerance`.
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


def _disclosed_items(value: object) -> list[str]:
    """Return the non-empty string items a ground-truth field value actually discloses.

    A field's ground truth is either a single string or a list of strings (e.g.
    `amenities_requested`, when a call discloses more than one). Corpus generation has no schema
    enforcement on array element types, so a list can contain `None` or empty-string noise --
    filtered out here so it's never treated as a disclosed-but-unmatchable value.
    """
    items = value if isinstance(value, list) else [value]
    return [item for item in items if isinstance(item, str) and item.strip()]


def is_match(extracted: str, truth: str | list[str]) -> bool:
    r"""True if `extracted` and `truth` agree closely enough to count as a recall hit.

    Extracted values are free text produced by a model, not guaranteed to match the ground-truth
    string exactly (e.g. "the caller, Tim Barker" vs "Tim Barker") -- normalized equality, or the
    shorter value appearing as a whole word (or run of words) inside the longer one, covers that
    without requiring an exact match. Plain substring containment (with no boundary check) would
    count "212" as a match for ground truth "12", or "103" for "3" -- a real risk for the short
    numeric/ID-like fields (unit_number, phone digits, prices), where every digit string is a
    substring of many other digit strings that share no actual value.

    The boundary check requires a non-word lookaround only on a side whose own edge character is
    itself a word character. A bare-digit edge (e.g. "212") needs it on both sides to block a
    digit-collision false positive ("212" inside "3212"). A symbol-prefixed edge (e.g. "$950")
    needs no lookaround at all: the symbol itself unambiguously delimits the value regardless of
    what's adjacent to it, so "$950" correctly matches whether it's preceded by whitespace
    ("around $950") or abuts a word with no separator ("was$950 total").

    `truth` can be a list (e.g. `amenities_requested`, where a call's ground truth can disclose
    several amenities) since the pipeline only ever commits a single value per field -- a hit
    against any one disclosed item counts as recall for that call.
    """
    if isinstance(truth, list):
        return any(is_match(extracted, item) for item in _disclosed_items(truth))
    norm_extracted, norm_truth = normalize(extracted), normalize(truth)
    if not norm_extracted or not norm_truth:
        return False
    if norm_extracted == norm_truth:
        return True
    shorter, longer = sorted((norm_extracted, norm_truth), key=len)
    lead = r"(?<!\w)" if re.match(r"\w", shorter[0]) else ""
    trail = r"(?!\w)" if re.match(r"\w", shorter[-1]) else ""
    return re.search(f"{lead}{re.escape(shorter)}{trail}", longer) is not None


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
            call_id for call_id, gt in ground_truths.items() if _disclosed_items(gt.get(field_name))
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


def score_precision(
    ground_truths: dict[int, dict],
    committed: dict[int, dict[str, str]],
    fields: tuple[str, ...] = FIELDS,
) -> dict[str, dict]:
    """Return `{field_name: {"n_committed", "n_correct", "n_false_positives", "precision"}}`.

    Precision is `None` (rather than a divide-by-zero) for a field the pipeline never committed a
    value for. Per the spec in the module docstring: a committed value counts as correct
    (`n_correct`) only when the ground truth for that call discloses a value for the field (non-
    null) *and* `is_match` agrees; a committed value against null ground truth, or against
    non-null ground truth it doesn't match, is a false positive.
    """
    stats: dict[str, dict] = {}
    for field_name in fields:
        n_committed = 0
        n_correct = 0
        for call_id, fields_committed in committed.items():
            value = fields_committed.get(field_name)
            if not value:
                continue
            n_committed += 1
            truth = ground_truths.get(call_id, {}).get(field_name)
            if _disclosed_items(truth) and is_match(value, truth):
                n_correct += 1
        stats[field_name] = {
            "n_committed": n_committed,
            "n_correct": n_correct,
            "n_false_positives": n_committed - n_correct,
            "precision": (n_correct / n_committed) if n_committed else None,
        }
    return stats


def load_jev_calls(conn: sqlite3.Connection, pipeline: str) -> dict[str, list[dict]]:
    """Return `{field_name: [raw_output_dict, ...]}` for every successful jev call for `pipeline`.

    A jev call's per-field identity only exists via its `field_extractions` row (`pipeline_runs`
    itself has no `field_name` column), so this joins the two on `run_id`. A jev call that errored
    out (see `pipeline_core._run_gliner_jev_step`) never gets a `field_extractions` row and so is
    excluded here -- it carries no field identity to attribute it to.
    """
    rows = conn.execute(
        """
        SELECT fe.field_name, pr.raw_output_json
        FROM field_extractions fe
        JOIN pipeline_runs pr ON pr.run_id = fe.run_id
        WHERE pr.pipeline = ? AND pr.stage = 'jev' AND pr.raw_output_json IS NOT NULL
        """,
        (pipeline,),
    ).fetchall()
    calls: dict[str, list[dict]] = {}
    for field_name, raw_output_json in rows:
        calls.setdefault(field_name, []).append(json.loads(raw_output_json))
    return calls


def score_call_volume(jev_calls: dict[str, list[dict]], fields: tuple[str, ...] = FIELDS) -> dict[str, dict]:
    """Return per-field jev call volume: `Choice`-question count and distinct-candidate-set size.

    `Choice` questions (2+ distinct candidates) are jev's expensive path -- their context is every
    distinct candidate's snippet concatenated together, vs. a `Noul` question's single snippet --
    so `n_choice_calls` and the distinct-candidate-set sizes are what a threshold change (inflating
    or shrinking the sets jev dedups against) would move.
    """
    stats: dict[str, dict] = {}
    for field_name in fields:
        calls = jev_calls.get(field_name, [])
        n_choice_calls = sum(1 for call in calls if call.get("question_type") == "choice")
        distinct_sizes = [len(call.get("distinct_candidates") or []) for call in calls]
        stats[field_name] = {
            "n_jev_calls": len(calls),
            "n_choice_calls": n_choice_calls,
            "avg_distinct_candidates": (sum(distinct_sizes) / len(distinct_sizes)) if distinct_sizes else None,
            "max_distinct_candidates": max(distinct_sizes) if distinct_sizes else None,
        }
    return stats


def merge_stats(*stat_dicts: dict[str, dict], fields: tuple[str, ...] = FIELDS) -> dict[str, dict]:
    """Merge several `{field_name: {...}}` stat dicts (recall, precision, call volume) into one
    `{field_name: {**all_their_keys}}` dict, keyed the same way each already is."""
    merged: dict[str, dict] = {field_name: {} for field_name in fields}
    for stat_dict in stat_dicts:
        for field_name, field_stats in stat_dict.items():
            merged.setdefault(field_name, {}).update(field_stats)
    return merged


def print_table(stats: dict[str, dict]) -> None:
    header = (
        f"{'field':<22} {'matched/expected':<18} {'recall':<8} "
        f"{'correct/committed':<20} {'precision':<10} "
        f"{'jev calls':<10} {'choice calls':<13} {'avg distinct':<13}"
    )
    print(header)
    for field_name, field_stats in stats.items():
        recall = field_stats.get("recall")
        recall_str = f"{recall:.2%}" if recall is not None else "n/a"
        ratio_str = f"{field_stats.get('n_matched', 0)}/{field_stats.get('n_expected', 0)}"

        precision = field_stats.get("precision")
        precision_str = f"{precision:.2%}" if precision is not None else "n/a"
        precision_ratio_str = f"{field_stats.get('n_correct', 0)}/{field_stats.get('n_committed', 0)}"

        n_jev_calls = field_stats.get("n_jev_calls", 0)
        n_choice_calls = field_stats.get("n_choice_calls", 0)
        avg_distinct = field_stats.get("avg_distinct_candidates")
        avg_distinct_str = f"{avg_distinct:.2f}" if avg_distinct is not None else "n/a"

        print(
            f"{field_name:<22} {ratio_str:<18} {recall_str:<8} "
            f"{precision_ratio_str:<20} {precision_str:<10} "
            f"{n_jev_calls:<10} {n_choice_calls:<13} {avg_distinct_str:<13}"
        )


def diff_against_baseline(stats: dict[str, dict], baseline: dict[str, dict], tolerance: float) -> list[str]:
    """Return a list of human-readable regression messages vs. `baseline`, empty if none.

    Flags, per field: recall dropped by more than `tolerance`; precision dropped by more than
    `tolerance`; or jev Choice-call volume rose by more than a `tolerance` fraction relative to
    the baseline's count (more Choice calls means more cost/latency, so an *increase* is the
    regression direction here, unlike recall/precision). Any of the three is skipped for a field
    where the relevant metric is missing on either side (e.g. a field neither run ever committed
    a value for has no precision to compare).
    """
    regressions = []
    for field_name, field_stats in stats.items():
        base = baseline.get(field_name, {})

        base_recall = base.get("recall")
        recall = field_stats.get("recall")
        if base_recall is not None and recall is not None:
            drop = base_recall - recall
            if drop > tolerance:
                regressions.append(
                    f"{field_name}: recall dropped from {base_recall:.2%} to {recall:.2%} "
                    f"(-{drop:.2%}, tolerance {tolerance:.2%})"
                )

        base_precision = base.get("precision")
        precision = field_stats.get("precision")
        if base_precision is not None and precision is not None:
            drop = base_precision - precision
            if drop > tolerance:
                regressions.append(
                    f"{field_name}: precision dropped from {base_precision:.2%} to {precision:.2%} "
                    f"(-{drop:.2%}, tolerance {tolerance:.2%})"
                )

        base_choice_calls = base.get("n_choice_calls")
        choice_calls = field_stats.get("n_choice_calls")
        if base_choice_calls is not None and choice_calls is not None:
            allowed = base_choice_calls * (1 + tolerance)
            if choice_calls > allowed:
                regressions.append(
                    f"{field_name}: jev choice-call volume rose from {base_choice_calls} to "
                    f"{choice_calls} (tolerance {tolerance:.2%} vs. baseline)"
                )

    return regressions


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("db_path", type=Path)
    parser.add_argument("--pipeline", default="gliner_jev", help="Pipeline to score (default: gliner_jev).")
    parser.add_argument(
        "--save", type=Path, default=None, help="Write this run's recall/precision/call-volume table to a JSON snapshot."
    )
    parser.add_argument(
        "--baseline", type=Path, default=None, help="A previously saved snapshot to diff this run against."
    )
    parser.add_argument(
        "--tolerance",
        type=float,
        default=DEFAULT_TOLERANCE,
        help=(
            "Max allowed per-field recall/precision drop, or jev Choice-call volume increase, "
            f"vs. --baseline before exiting non-zero (default: {DEFAULT_TOLERANCE})."
        ),
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        parser.error(f"database file not found: {args.db_path}")

    conn = db.connect(args.db_path)
    try:
        ground_truths = load_ground_truths(conn)
        committed = load_final_committed_values(conn, args.pipeline)
        jev_calls = load_jev_calls(conn, args.pipeline)
    finally:
        conn.close()

    stats = merge_stats(
        score_recall(ground_truths, committed),
        score_precision(ground_truths, committed),
        score_call_volume(jev_calls),
    )
    print_table(stats)

    if args.save is not None:
        args.save.parent.mkdir(parents=True, exist_ok=True)
        args.save.write_text(json.dumps(stats, indent=2), encoding="utf-8")
        print(f"\nSaved snapshot -> {args.save}")

    if args.baseline is not None:
        if not args.baseline.exists():
            parser.error(f"baseline snapshot not found: {args.baseline}")
        try:
            baseline = json.loads(args.baseline.read_text(encoding="utf-8"))
        except json.JSONDecodeError as exc:
            parser.error(f"baseline snapshot at {args.baseline} is not valid JSON: {exc}")
        regressions = diff_against_baseline(stats, baseline, args.tolerance)
        if regressions:
            print(f"\nRegressed past tolerance ({args.tolerance:.2%}) vs. {args.baseline}:")
            for message in regressions:
                print(f"  {message}")
            return 1
        print(f"\nNo regression past tolerance ({args.tolerance:.2%}) vs. {args.baseline}.")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
