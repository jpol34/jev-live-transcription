"""Computes per-field `decision_reason` tallies for `gliner_only` in a capture DB.

`gliner_only_resolver.GlinerOnlyResolver` records why it did or didn't commit a value each time it
resolves a field, via a `decision_reason` string (`single_above_floor` / `single_below_floor` /
`margin_winner` / `margin_too_close` / `below_floor`) written into the `gliner_only` pipeline_runs
row's `raw_output_json` for that tick. `score_recall.py`'s printed "jev calls / choice calls / avg
distinct" columns hardcode `WHERE stage = 'jev'`, so they're not meaningful for `gliner_only` --
this script is the `gliner_only` analogue, reporting the real none-of-these-equivalent rate
(`margin_too_close` + `below_floor`, both of which reject every candidate for that decision) instead.

A `gliner_only` `pipeline_runs` row's `raw_output_json` holds `{field_name: {...,
"decision_reason": ...}}` for whichever fields `_run_gliner_only_step` resolved that tick -- joined
against `field_extractions` via `score_recall.load_field_run_rows` (shared with that script's own
`load_jev_calls`) so each persisted (call, field, tick) resolution is counted exactly once.

`below_floor`/`margin_too_close` are the two `is_none_of_these=True` outcomes, both only reachable
once 2+ distinct candidates have settled (`GlinerOnlyResolver._resolve_settled` --
`below_floor` fires when even the top-ranked settled candidate misses its commit floor,
`margin_too_close` when it clears the floor but not its margin over the runner-up). The 1-candidate
case (`single_above_floor`/`single_below_floor`) never reaches settled arbitration at all, so
`single_below_floor` is a distinct "no signal yet" outcome, not a confident rejection, and is
excluded from the none-of-these rate below.

Usage:
    uv run python scripts/score_decision_reasons.py <db_path>
"""

import argparse
import json
import sqlite3
from collections import Counter, defaultdict
from pathlib import Path

from jev_live_transcription import db
from jev_live_transcription.llm_baseline import FIELDS
from score_recall import load_field_run_rows

_NONE_OF_THESE_REASONS = ("margin_too_close", "below_floor")


def load_decision_reasons(conn: sqlite3.Connection) -> dict[str, Counter]:
    """Return `{field_name: Counter({decision_reason: count})}` across every `gliner_only_commit`
    resolution in the DB."""
    counts: dict[str, Counter] = defaultdict(Counter)
    for field_name, raw_output_json in load_field_run_rows(conn, "gliner_only", "gliner_only_commit"):
        entry = json.loads(raw_output_json).get(field_name)
        if entry is None:
            continue
        counts[field_name][entry.get("decision_reason", "(missing)")] += 1
    return counts


def print_table(counts: dict[str, Counter], fields: tuple[str, ...] = FIELDS) -> None:
    all_reasons = sorted({reason for c in counts.values() for reason in c})
    print(f"{'field':<22} {'total':<7} " + " ".join(f"{r:<20}" for r in all_reasons))
    for field_name in fields:
        c = counts.get(field_name, Counter())
        total = sum(c.values())
        print(f"{field_name:<22} {total:<7} " + " ".join(f"{c.get(r, 0):<20}" for r in all_reasons))

    print("\nnone-of-these rate (margin_too_close + below_floor) / total decisions:")
    for field_name in fields:
        c = counts.get(field_name, Counter())
        total = sum(c.values())
        none_of_these = sum(c.get(r, 0) for r in _NONE_OF_THESE_REASONS)
        rate = f"{none_of_these / total:.2%}" if total else "n/a"
        margin_rate = f"{c.get('margin_too_close', 0) / total:.2%}" if total else "n/a"
        print(f"  {field_name:<22} none_of_these={rate:<9} margin_too_close={margin_rate:<9} n={total}")


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("db_path", type=Path)
    args = parser.parse_args()

    if not args.db_path.exists():
        parser.error(f"database file not found: {args.db_path}")

    conn = db.connect(args.db_path)
    try:
        counts = load_decision_reasons(conn)
    finally:
        conn.close()

    print_table(counts)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
