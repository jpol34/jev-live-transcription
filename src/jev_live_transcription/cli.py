"""Command-line entry point for the jev-live-transcription benchmark harness.

Installed as the `jlt` console script (see `pyproject.toml`); also runnable as
`uv run python -m jev_live_transcription.cli`.
"""

from __future__ import annotations

import argparse
import asyncio
import logging
from pathlib import Path

from . import batch_runner, config, corpus, secrets

DEFAULT_DB_PATH = Path("data/benchmark.sqlite3")


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
    tui_parser.add_argument(
        "--enable-llm-baseline",
        action="store_true",
        default=False,
        help=(
            "Also show the GPT-5.1 comparison column (default: off). This calls the real OpenAI "
            "API on every config.LLM_CADENCE_TICKS-th grown tick and costs real money -- only "
            "pass this with explicit approval, never by default."
        ),
    )

    batch_parser = subparsers.add_parser(
        "batch", help="Drive the call corpus through the pipeline in batch (non-realtime) mode."
    )
    batch_parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Only run the first N calls (by call_id) instead of the full corpus.",
    )
    batch_parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Capture DB path (default: {DEFAULT_DB_PATH}).",
    )
    batch_parser.add_argument(
        "--call-concurrency",
        type=int,
        default=1,
        help=(
            "Max concurrent calls (default: 1, fully sequential). Raising this makes calls "
            "contend for the shared GLiNER model, which inflates recorded latency with queueing "
            f"delay a real single call would never see -- use config.CALL_CONCURRENCY "
            f"({config.CALL_CONCURRENCY}) only for a throughput run whose latency numbers won't "
            "be used, never for real benchmark data collection."
        ),
    )
    batch_parser.add_argument(
        "--gliner-concurrency",
        type=int,
        default=1,
        help=(
            "Max concurrent GLiNER inferences across all calls (default: 1). Same caveat as "
            f"--call-concurrency -- config.GLINER_CONCURRENCY ({config.GLINER_CONCURRENCY}) is a "
            "throughput setting, not a benchmark-data-collection one."
        ),
    )
    batch_parser.add_argument(
        "--enable-llm-baseline",
        action="store_true",
        default=False,
        help=(
            "Also run the GPT-5.1 comparison arm (default: off). This calls the real OpenAI API "
            "on every config.LLM_CADENCE_TICKS-th grown tick and costs real money -- only pass "
            "this for an explicitly approved final benchmark comparison run, never for routine "
            "GLiNER+jev data collection."
        ),
    )

    gpu_run_parser = subparsers.add_parser(
        "gpu-run", help="Run the benchmark on a real RunPod GPU pod (creates and tears down the pod)."
    )
    gpu_run_parser.add_argument(
        "--subset",
        type=int,
        default=None,
        help="Only run the first N calls (by call_id) instead of the full corpus.",
    )
    gpu_run_parser.add_argument(
        "--db-path",
        type=Path,
        default=DEFAULT_DB_PATH,
        help=f"Local path to copy the pod's capture DB back to (default: {DEFAULT_DB_PATH}).",
    )
    gpu_run_parser.add_argument("--call-concurrency", type=int, default=1)
    gpu_run_parser.add_argument("--gliner-concurrency", type=int, default=1)
    gpu_run_parser.add_argument(
        "--enable-llm-baseline",
        action="store_true",
        default=False,
        help="Also run the GPT-5.1 comparison arm -- same real-money caveat as `jlt batch`.",
    )
    gpu_run_parser.add_argument(
        "--pod-state-path",
        type=Path,
        default=Path(".jlt_gpu_state.json"),
        help="Where to persist the pod id so a re-run resumes the same pod instead of creating a new one.",
    )
    gpu_run_parser.add_argument(
        "--ssh-key",
        required=True,
        help=(
            "Path to the SSH private key matching a key registered on the RunPod account. "
            "The matching `<ssh-key>.pub` file's contents are injected into the pod as "
            "`PUBLIC_KEY`, which the pod image's own startup script requires to start sshd at "
            "all -- required, not optional, since the pod is otherwise never SSH-reachable."
        ),
    )
    gpu_run_parser.add_argument(
        "--keep-pod",
        action="store_true",
        default=False,
        help="Don't terminate the pod on exit (debugging only -- the pod keeps billing).",
    )

    return parser


def _run_batch(args: argparse.Namespace) -> int:
    if args.enable_llm_baseline:
        secrets.load_openai_key()
    secrets.load_typesafe_key()

    calls = corpus.load_all()
    call_ids = sorted(calls)
    if args.subset is not None:
        call_ids = call_ids[: args.subset]

    args.db_path.parent.mkdir(parents=True, exist_ok=True)

    print(
        f"Running {len(call_ids)} call(s) -> {args.db_path} "
        f"(call_concurrency={args.call_concurrency}, gliner_concurrency={args.gliner_concurrency}, "
        f"enable_llm_baseline={args.enable_llm_baseline})"
    )

    result = asyncio.run(
        batch_runner.run_batch(
            call_ids,
            db_path=args.db_path,
            call_concurrency=args.call_concurrency,
            gliner_concurrency=args.gliner_concurrency,
            calls=calls,
            enable_llm_baseline=args.enable_llm_baseline,
        )
    )

    print(f"Done: {len(result.succeeded)} succeeded, {len(result.failed)} failed.")
    for call_id, exc in result.failed:
        print(f"  call_id={call_id} failed: {exc}")
    return 1 if result.failed else 0


def _run_tui(args: argparse.Namespace) -> int:
    from . import tui

    tui.run(args.call_id, db_path=args.db_path, enable_llm_baseline=args.enable_llm_baseline)
    return 0


def _run_gpu_run(args: argparse.Namespace) -> int:
    from . import gpu_run

    return gpu_run.run_gpu(
        subset=args.subset,
        db_path=args.db_path,
        call_concurrency=args.call_concurrency,
        gliner_concurrency=args.gliner_concurrency,
        enable_llm_baseline=args.enable_llm_baseline,
        pod_state_path=args.pod_state_path,
        ssh_key=args.ssh_key,
        keep_pod=args.keep_pod,
    )


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = _build_parser()
    args = parser.parse_args(argv)
    if args.command == "batch":
        return _run_batch(args)
    if args.command == "tui":
        return _run_tui(args)
    if args.command == "gpu-run":
        return _run_gpu_run(args)
    parser.error(f"unknown command: {args.command!r}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
