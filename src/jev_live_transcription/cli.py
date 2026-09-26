"""Command-line entry point for the jev-live-transcription benchmark harness.

Installed as the `jlt` console script (see `pyproject.toml`); also runnable as
`uv run python -m jev_live_transcription.cli`.
"""

from __future__ import annotations

import argparse
from pathlib import Path


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="jlt", description="jev-live-transcription CLI")
    subparsers = parser.add_subparsers(dest="command", required=True)

    tui_parser = subparsers.add_parser(
        "tui", help="Watch one call replay live, tick by tick, in a terminal UI"
    )
    tui_parser.add_argument(
        "call_id", type=int, help="Call id to replay (see output/metadata/ for available ids)"
    )
    tui_parser.add_argument(
        "--db-path",
        type=Path,
        default=None,
        help=(
            "Capture DB path to write the run to (defaults to "
            "output/tui_captures/call_<id>.db)"
        ),
    )

    return parser


def main(argv: list[str] | None = None) -> None:
    parser = _build_parser()
    args = parser.parse_args(argv)

    if args.command == "tui":
        from . import tui

        tui.run(args.call_id, db_path=args.db_path)


if __name__ == "__main__":
    main()
