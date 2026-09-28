"""Computes true per-tick total latency (mean/median/p95) and error rate for each pipeline in a
capture DB.

`pipeline_runs` records one row per pipeline invocation, but a tick's real end-to-end latency for
a pipeline isn't always one row -- see `pipeline_core`'s module docstring for the full design:

- `gliner_jev`: a tick's total is its `"gliner_standard"` row's latency plus every `"jev"`-stage
  row's latency for that tick (one `"jev"` row per field jev was asked to resolve that tick).
- `gliner_only`: GLiNER extraction only ever runs once per tick, so its `"gliner_standard"` row is
  recorded once, under `pipeline = "gliner_jev"`, and shared rather than duplicated under
  `pipeline = "gliner_only"`. A tick's true `gliner_only` latency is therefore that shared row's
  latency plus the tick's own `"gliner_only_commit"` row's latency, joined by `tick_id` -- using
  `gliner_only`'s own rows alone would undercount every tick by the shared extraction cost.
- `llm`: its `"llm"`-stage row is already self-contained (one row per tick it ran on, no shared
  rows to fold in).

A tick's latency figure sums every contributing row's `latency_ms`, whether or not that row
errored -- a failed call still spent real wall-clock time, and this is meant to answer "how long
does a tick actually take", not "how long do only the successful ones take". A tick's error rate
counts it as errored if any contributing row recorded a non-null `error`.

Usage:
    uv run python scripts/score_latency.py <db_path> [--pipelines gliner_jev,gliner_only,llm]
"""

import argparse
import sqlite3
import statistics
import sys
from pathlib import Path

from jev_live_transcription import db

DEFAULT_PIPELINES = ("gliner_jev", "gliner_only", "llm")

# Where the shared GLiNER extraction row lives regardless of which resolver(s) consume it -- see
# pipeline_core's module docstring.
_GLINER_STANDARD_PIPELINE = "gliner_jev"
_GLINER_STANDARD_STAGE = "gliner_standard"


def percentile(values: list[float], p: float) -> float | None:
    """Return the `p`th percentile (0-100) of `values` via nearest-rank, or `None` if empty.

    Same nearest-rank approach as `scripts/measure_gliner_concurrency.py`'s `percentile()`."""
    if not values:
        return None
    ordered = sorted(values)
    rank = max(0, min(len(ordered) - 1, round(p / 100 * (len(ordered) - 1))))
    return ordered[rank]


def _load_rows(conn: sqlite3.Connection, pipeline: str, stages: tuple[str, ...]):
    """Return `(tick_id, latency_ms, error)` for every `pipeline_runs` row matching `pipeline` and
    one of `stages`."""
    placeholders = ",".join("?" for _ in stages)
    return conn.execute(
        f"SELECT tick_id, latency_ms, error FROM pipeline_runs "
        f"WHERE pipeline = ? AND stage IN ({placeholders})",
        (pipeline, *stages),
    ).fetchall()


def per_tick_gliner_jev(conn: sqlite3.Connection) -> dict[int, dict]:
    """Return `{tick_id: {"latency_ms", "errored"}}` for `gliner_jev`: its `gliner_standard` row's
    latency plus every `jev`-stage row's latency for that tick, summed."""
    rows = _load_rows(conn, "gliner_jev", ("gliner_standard", "jev"))
    ticks: dict[int, dict] = {}
    for tick_id, latency_ms, error in rows:
        entry = ticks.setdefault(tick_id, {"latency_ms": 0.0, "errored": False})
        if latency_ms is not None:
            entry["latency_ms"] += latency_ms
        if error is not None:
            entry["errored"] = True
    return ticks


def per_tick_gliner_only(conn: sqlite3.Connection) -> dict[int, dict]:
    """Return `{tick_id: {"latency_ms", "errored"}}` for `gliner_only`: each tick's
    `gliner_only_commit` row joined by `tick_id` with the shared `gliner_standard` row recorded
    under `pipeline = "gliner_jev"`.

    A `gliner_only_commit` row whose tick has no matching shared `gliner_standard` row (data the
    real pipeline never produces, since `_run_gliner_only_step` only ever runs on candidates
    `_run_gliner_extraction_step` already extracted for that same tick) is excluded rather than
    silently counted at its own latency alone -- reporting an undercounted number would be worse
    than omitting it, given this join exists specifically to avoid undercounting. Excluded ticks
    are reported to stderr as a warning.
    """
    standard_by_tick = {
        tick_id: (latency_ms, error)
        for tick_id, latency_ms, error in _load_rows(
            conn, _GLINER_STANDARD_PIPELINE, (_GLINER_STANDARD_STAGE,)
        )
    }
    commit_rows = _load_rows(conn, "gliner_only", ("gliner_only_commit",))

    ticks: dict[int, dict] = {}
    unjoined: list[int] = []
    for tick_id, latency_ms, error in commit_rows:
        shared = standard_by_tick.get(tick_id)
        if shared is None:
            unjoined.append(tick_id)
            continue
        shared_latency, shared_error = shared
        total = (shared_latency or 0.0) + (latency_ms or 0.0)
        errored = error is not None or shared_error is not None
        ticks[tick_id] = {"latency_ms": total, "errored": errored}

    if unjoined:
        print(
            f"warning: {len(unjoined)} gliner_only_commit row(s) had no matching shared "
            f"gliner_standard row (tick_id(s): {sorted(unjoined)}) -- excluded from latency stats",
            file=sys.stderr,
        )
    return ticks


def per_tick_llm(conn: sqlite3.Connection) -> dict[int, dict]:
    """Return `{tick_id: {"latency_ms", "errored"}}` for `llm`: its own `llm`-stage row is already
    self-contained, one per tick it ran on."""
    rows = _load_rows(conn, "llm", ("llm",))
    return {
        tick_id: {"latency_ms": latency_ms or 0.0, "errored": error is not None}
        for tick_id, latency_ms, error in rows
    }


_LOADERS = {
    "gliner_jev": per_tick_gliner_jev,
    "gliner_only": per_tick_gliner_only,
    "llm": per_tick_llm,
}


def summarize(ticks: dict[int, dict]) -> dict:
    """Return `{"n_ticks", "mean_ms", "median_ms", "p95_ms", "n_errored", "error_rate"}` across
    every tick's total latency. All stats are `None` (rather than a divide-by-zero) when `ticks`
    is empty."""
    latencies = [entry["latency_ms"] for entry in ticks.values()]
    n_ticks = len(latencies)
    n_errored = sum(1 for entry in ticks.values() if entry["errored"])
    return {
        "n_ticks": n_ticks,
        "mean_ms": statistics.fmean(latencies) if latencies else None,
        "median_ms": statistics.median(latencies) if latencies else None,
        "p95_ms": percentile(latencies, 95),
        "n_errored": n_errored,
        "error_rate": (n_errored / n_ticks) if n_ticks else None,
    }


def score_latency(conn: sqlite3.Connection, pipelines: tuple[str, ...] = DEFAULT_PIPELINES) -> dict[str, dict]:
    """Return `{pipeline: summary}` for each of `pipelines`.

    Raises `ValueError` for a pipeline name outside `_LOADERS` -- `main` validates this up front
    via `argparse.error` instead, so this only fires for a caller that bypasses the CLI.
    """
    stats: dict[str, dict] = {}
    for pipeline in pipelines:
        loader = _LOADERS.get(pipeline)
        if loader is None:
            raise ValueError(f"unknown pipeline: {pipeline!r} (expected one of {sorted(_LOADERS)})")
        stats[pipeline] = summarize(loader(conn))
    return stats


def _fmt_ms(value: float | None) -> str:
    return f"{value:.1f}" if value is not None else "n/a"


def _fmt_pct(value: float | None) -> str:
    return f"{value:.2%}" if value is not None else "n/a"


def print_table(stats: dict[str, dict]) -> None:
    header = (
        f"{'pipeline':<14} {'n_ticks':<9} {'mean_ms':<10} {'median_ms':<10} "
        f"{'p95_ms':<10} {'n_errored':<10} {'error_rate':<10}"
    )
    print(header)
    for pipeline, s in stats.items():
        print(
            f"{pipeline:<14} {s['n_ticks']:<9} {_fmt_ms(s['mean_ms']):<10} "
            f"{_fmt_ms(s['median_ms']):<10} {_fmt_ms(s['p95_ms']):<10} "
            f"{s['n_errored']:<10} {_fmt_pct(s['error_rate']):<10}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("db_path", type=Path)
    parser.add_argument(
        "--pipelines",
        default=",".join(DEFAULT_PIPELINES),
        help=f"Comma-separated pipelines to report (default: {','.join(DEFAULT_PIPELINES)}).",
    )
    args = parser.parse_args()

    if not args.db_path.exists():
        parser.error(f"database file not found: {args.db_path}")

    pipelines = tuple(p.strip() for p in args.pipelines.split(",") if p.strip())
    unknown = [p for p in pipelines if p not in _LOADERS]
    if unknown:
        parser.error(f"unknown pipeline(s): {', '.join(unknown)} (expected one of {sorted(_LOADERS)})")

    conn = db.connect(args.db_path)
    try:
        stats = score_latency(conn, pipelines)
    finally:
        conn.close()

    print_table(stats)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
