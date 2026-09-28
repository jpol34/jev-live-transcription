"""Stands up a RunPod A100 pod, installs this repo's own package (not a third-party stack), starts
`jlt serve`, and runs a closed-loop load test against it -- either a concurrency sweep at one fixed
tuning config (`bench` mode) or a batch-tuning sweep at one fixed concurrency (`tune` mode, restarts
the server between configs). Either way, `results.json` gets one entry per point, rewritten after
each completes so a crash partway through the sweep doesn't lose already-collected data. Always
terminates the pod on exit, success or failure.

Mirrors `benchmarks/gliner_serve/pod_bench.py`'s hangar-based lifecycle (create/resume via
`hangar`, poll for `RUNNING` + `ssh.direct`, retry real SSH connectability rather than trusting
metadata presence alone) and reuses `gpu_run._cuda_preflight`/`_read_public_key`/`_wait_for_pod_ready`
directly, but installs this project fresh from source (`uv pip install -e .`) instead of a
third-party package -- there is no pre-built, up-to-date Docker image of this repo's latest commit
to pull instead (the `gpu-test` image `gpu_run.py` uses is built and pushed manually, not on every
commit, so it can't be trusted to carry code this session just merged).

Two details of this base image are load-bearing and easy to drop by accident (same as
`gliner_serve/pod_bench.py`, which shares this image):

- `PUBLIC_KEY` must be injected via `extra_env` -- the pod image's own startup script only starts
  sshd when that env var is present. Without it every SSH attempt looks identical to a slow boot
  for the entire wait window.
- `pip install` on this image needs `--break-system-packages` -- it's PEP 668
  externally-managed (Debian base), same as the image this repo's own `Dockerfile` builds from.

Usage:
    python pod_bench.py smoke --ssh-key ~/.ssh/id_ed25519
    python pod_bench.py bench --ssh-key ~/.ssh/id_ed25519 --concurrencies 50,200 --duration-s 25
    python pod_bench.py tune --ssh-key ~/.ssh/id_ed25519 --concurrency 200 \
        --configs 16:20,32:20,64:20,16:10,32:10,64:10
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import shlex
import subprocess
import sys
import time
from pathlib import Path

import hangar

from jev_live_transcription import config, secrets
from jev_live_transcription.gpu_run import _cuda_preflight, _read_public_key, _wait_for_pod_ready

# `load_test.py` lives alongside this file and defines the one real request shape this benchmark
# uses -- imported rather than duplicated so the readiness check here exercises the exact same
# payload shape as the real load-tested traffic. `_pod_ssh` is the SSH/scp helper module shared
# with the sibling `gliner_serve/pod_bench.py`, one level up.
_THIS_DIR = Path(__file__).resolve().parent
for _extra_path in (_THIS_DIR, _THIS_DIR.parent):
    if str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))
import load_test  # noqa: E402
from _pod_ssh import run_ssh as _run_ssh  # noqa: E402
from _pod_ssh import scp_up as _scp_up  # noqa: E402
from _pod_ssh import wait_for_ssh_connectable as _wait_for_ssh_connectable  # noqa: E402

_LOGGER = logging.getLogger(__name__)

_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
_POD_NAME = "gliner-batch-service-bench"
_DISK_GB = 20
_SERVE_PORT = 8000
_REMOTE_DIR = "/root/gliner_batch_service_bench"
_REMOTE_LOG_PATH = f"{_REMOTE_DIR}/jlt-serve.log"
_REMOTE_PID_PATH = f"{_REMOTE_DIR}/jlt-serve.pid"
_REMOTE_FIXTURE_PATH = f"{_REMOTE_DIR}/fixtures/sample_windows.json"

# Generous: covers a cold HuggingFace Hub download of the checkpoint on top of the model's own
# load time -- this pod installs the project fresh, so unlike a pre-baked Docker image, the
# checkpoint is never already cached on first run.
_SERVER_READY_TIMEOUT_S = 600.0
# Per-request cap for load_test.py's aiohttp client (see load_test.py's run_load_test) -- keeps one
# stuck request from making the whole load_test.py process run far longer than --duration-s, which
# would in turn blow past this script's own SSH call timeout below.
_LOAD_TEST_REQUEST_TIMEOUT_S = 30.0
# Margin added to --duration-s for the SSH call wrapping each load_test.py invocation: bounds a
# single request to _LOAD_TEST_REQUEST_TIMEOUT_S, so the worst case is one straggler request still
# in flight when the run's duration elapses, not an unbounded hang.
_LOAD_TEST_SSH_MARGIN_S = _LOAD_TEST_REQUEST_TIMEOUT_S + 30.0


def _install_deps(ssh_direct: dict, ssh_key: str) -> None:
    """Uploads this project's source and installs it fresh via `uv pip install -e .` -- the exact
    commit under test, not whatever a separately-built Docker image happens to carry.
    """
    repo_root = _THIS_DIR.parent.parent
    _run_ssh(ssh_direct, ssh_key, f"mkdir -p {_REMOTE_DIR}")
    _scp_up(ssh_direct, ssh_key, repo_root / "pyproject.toml", f"{_REMOTE_DIR}/pyproject.toml")
    _scp_up(ssh_direct, ssh_key, repo_root / "src", f"{_REMOTE_DIR}/src", recursive=True, timeout_s=120.0)

    # This project's own pyproject.toml depends on `hangar @ git+https://...` -- pip needs the
    # `git` binary on PATH to resolve that VCS URL, unrelated to whether hangar is ever used by
    # the served app itself. Installed unconditionally rather than probed first: `apt-get install`
    # on an already-present package is a fast no-op, cheaper than a separate `which git` round trip.
    result = _run_ssh(
        ssh_direct,
        ssh_key,
        f"apt-get update -qq && apt-get install -y -qq git && "
        f"pip install --no-cache-dir --break-system-packages uv && "
        f"cd {_REMOTE_DIR} && uv pip install --system --break-system-packages --no-cache -e .",
        timeout_s=600.0,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to install this project on the pod: {result.stderr}")
    print("Project installed on pod.")

    _run_ssh(ssh_direct, ssh_key, f"mkdir -p {_REMOTE_DIR}/fixtures")
    _scp_up(
        ssh_direct,
        ssh_key,
        _THIS_DIR.parent / "gliner_serve" / "fixtures" / "sample_windows.json",
        _REMOTE_FIXTURE_PATH,
    )
    _scp_up(ssh_direct, ssh_key, _THIS_DIR / "load_test.py", f"{_REMOTE_DIR}/load_test.py")


def _start_server(
    ssh_direct: dict, ssh_key: str, *, max_batch_size: int | None, batch_wait_timeout_ms: float | None
) -> None:
    """Launches `jlt serve` detached and blocks until `/healthz` reports ready -- `create_app`'s
    lifespan loads the checkpoint and starts GlinerBatchEngine's worker synchronously (offloaded to
    a thread, but awaited) before uvicorn ever reports "startup complete", so a plain healthz poll
    is a real readiness signal here, unlike a service whose model loads lazily on first request.
    """
    cmd = f"jlt serve --host 0.0.0.0 --port {_SERVE_PORT}"
    if max_batch_size is not None:
        cmd += f" --max-batch-size {max_batch_size}"
    if batch_wait_timeout_ms is not None:
        cmd += f" --batch-wait-timeout-ms {batch_wait_timeout_ms}"
    # `nohup` alone does not background anything -- it only makes the process immune to SIGHUP.
    # The `&` must sit directly after the single command being backgrounded, not after a `&&`
    # chain: `A && nohup B &` backgrounds the *whole* `A && B` compound as one job, so `nohup B`
    # still runs synchronously inside it and the SSH channel blocks until B exits (forever, for a
    # server). `jlt serve` needs no cwd-relative files (unlike `_run_bench`'s `load_test.py`
    # invocation), so there's no `cd` to work around this trap with -- `mkdir -p` uses `;`, not
    # `&&`, and every path below is already absolute.
    remote_command = (
        f"mkdir -p {_REMOTE_DIR}; nohup {cmd} < /dev/null > {_REMOTE_LOG_PATH} 2>&1 & "
        f"echo $! > {_REMOTE_PID_PATH}; disown; echo STARTED"
    )
    result = _run_ssh(ssh_direct, ssh_key, remote_command)
    if "STARTED" not in result.stdout:
        raise RuntimeError(f"failed to launch jlt serve: {result.stderr}")

    deadline = time.monotonic() + _SERVER_READY_TIMEOUT_S
    last_result: subprocess.CompletedProcess | None = None
    while time.monotonic() < deadline:
        remaining = max(5.0, deadline - time.monotonic())
        try:
            last_result = _run_ssh(
                ssh_direct,
                ssh_key,
                f"curl -sSf http://localhost:{_SERVE_PORT}/healthz",
                timeout_s=min(30.0, remaining),
            )
        except subprocess.TimeoutExpired:
            continue
        if last_result.returncode == 0 and '"ok"' in last_result.stdout:
            return
        time.sleep(3.0)
    log_tail = _run_ssh(ssh_direct, ssh_key, f"tail -n 60 {_REMOTE_LOG_PATH} 2>&1")
    last_stderr = last_result.stderr if last_result else "(no attempt made)"
    raise TimeoutError(
        f"jlt serve did not report healthy within {_SERVER_READY_TIMEOUT_S}s "
        f"(last curl stderr: {last_stderr}) -- log tail:\n{log_tail.stdout}"
    )


def _stop_server(ssh_direct: dict, ssh_key: str) -> None:
    _run_ssh(
        ssh_direct,
        ssh_key,
        f"kill $(cat {_REMOTE_PID_PATH} 2>/dev/null) 2>/dev/null; pkill -f 'jlt serve' 2>/dev/null; true",
    )


def _smoke_test(ssh_direct: dict, ssh_key: str) -> None:
    """Starts the default-config server, confirms it answers `/extract` with the expected shape,
    then tears it down -- a human sanity check before this script is trusted for a real sweep."""
    _start_server(ssh_direct, ssh_key, max_batch_size=None, batch_wait_timeout_ms=None)
    try:
        payload = json.dumps({"text": "Hi, my name is Jane Doe and I live in unit 204."})
        result = _run_ssh(
            ssh_direct,
            ssh_key,
            f"curl -sSf -X POST http://localhost:{_SERVE_PORT}{load_test.ROUTE_PREFIX} "
            f"-H 'Content-Type: application/json' -d {shlex.quote(payload)}",
        )
        if result.returncode != 0:
            raise RuntimeError(f"smoke-test /extract request failed: {result.stderr}")
        try:
            parsed = json.loads(result.stdout)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"smoke-test response isn't valid JSON: {result.stdout!r}") from exc
        if not isinstance(parsed, dict) or not isinstance(parsed.get("entities"), list):
            raise RuntimeError(f"smoke-test response doesn't have the expected shape: {parsed!r}")
        print(f"Smoke-test response:\n{result.stdout}")
    finally:
        _stop_server(ssh_direct, ssh_key)


def _fmt(value: float | None) -> str:
    """Format a stat that's `None` when `summarize()` saw zero of that kind (e.g. `p50_ms` with no
    successful requests) without crashing on `None.__format__`.
    """
    return f"{value:.1f}" if value is not None else "n/a"


def _write_results(results_path: Path, results: list[dict]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))


def _record_and_report(results: list[dict], entry_extra: dict, stats: dict, results_path: Path) -> None:
    """Appends one sweep point's result, rewrites `results_path`, and prints its summary line --
    shared by `_run_bench` and `_run_tune` so both sweeps stay crash-safe (results persisted after
    every point) and report results the same way.
    """
    results.append({**entry_extra, **stats})
    _write_results(results_path, results)
    print(
        f"  -> p50={_fmt(stats['p50_ms'])}ms p95={_fmt(stats['p95_ms'])}ms "
        f"throughput={_fmt(stats['throughput_req_s'])}req/s"
    )


def _run_bench(
    ssh_direct: dict,
    ssh_key: str,
    *,
    concurrencies: list[int],
    duration_s: float,
    max_batch_size: int | None,
    batch_wait_timeout_ms: float | None,
    results_path: Path,
) -> None:
    """Starts one server (given `max_batch_size`/`batch_wait_timeout_ms`) and runs `load_test.py`
    against it once per concurrency point, rewriting `results_path` after each so a crash partway
    through the sweep doesn't lose already-collected data.
    """
    _start_server(ssh_direct, ssh_key, max_batch_size=max_batch_size, batch_wait_timeout_ms=batch_wait_timeout_ms)
    results: list[dict] = []
    try:
        for concurrency in concurrencies:
            print(f"Running load test at concurrency={concurrency}, duration={duration_s}s...")
            stats = _run_load_test_once(
                ssh_direct,
                ssh_key,
                concurrency=concurrency,
                duration_s=duration_s,
                context=f" at concurrency={concurrency}",
            )
            entry_extra = {
                "concurrency": concurrency,
                "duration_s": duration_s,
                "max_batch_size": max_batch_size,
                "batch_wait_timeout_ms": batch_wait_timeout_ms,
            }
            _record_and_report(results, entry_extra, stats, results_path)
    finally:
        _stop_server(ssh_direct, ssh_key)
    print(f"Bench complete -- wrote {len(results)} result(s) to {results_path}")


def _run_load_test_once(
    ssh_direct: dict, ssh_key: str, *, concurrency: int, duration_s: float, context: str = ""
) -> dict:
    remote_command = (
        f"cd {_REMOTE_DIR} && python load_test.py --concurrency {concurrency} "
        f"--duration-s {duration_s} --url http://localhost:{_SERVE_PORT} "
        f"--fixture-path {_REMOTE_FIXTURE_PATH} "
        f"--request-timeout-s {_LOAD_TEST_REQUEST_TIMEOUT_S}"
    )
    run_result = _run_ssh(
        ssh_direct, ssh_key, remote_command, timeout_s=duration_s + _LOAD_TEST_SSH_MARGIN_S
    )
    if run_result.returncode != 0:
        raise RuntimeError(f"load_test.py failed{context}: {run_result.stderr}")
    return json.loads(run_result.stdout.strip().splitlines()[-1])


def _run_tune(
    ssh_direct: dict,
    ssh_key: str,
    *,
    configs: list[tuple[int, float]],
    concurrency: int,
    duration_s: float,
    results_path: Path,
) -> None:
    """Sweeps `configs` (max_batch_size, batch_wait_timeout_ms) pairs at one fixed `concurrency`,
    restarting `jlt serve` between configs so each point measures its own tuning in isolation --
    mirrors `gliner_serve/run_matrix.py`'s restart-per-config pattern. Rewrites `results_path`
    after each config completes, same crash-safety discipline as `_run_bench`.
    """
    results: list[dict] = []
    for max_batch_size, batch_wait_timeout_ms in configs:
        print(
            f"Config max_batch_size={max_batch_size} batch_wait_timeout_ms={batch_wait_timeout_ms}: "
            f"starting server..."
        )
        _start_server(
            ssh_direct, ssh_key, max_batch_size=max_batch_size, batch_wait_timeout_ms=batch_wait_timeout_ms
        )
        try:
            print(f"  running load test at concurrency={concurrency}, duration={duration_s}s...")
            stats = _run_load_test_once(
                ssh_direct,
                ssh_key,
                concurrency=concurrency,
                duration_s=duration_s,
                context=f" for config max_batch_size={max_batch_size} batch_wait_timeout_ms={batch_wait_timeout_ms}",
            )
        finally:
            _stop_server(ssh_direct, ssh_key)
        entry_extra = {
            "concurrency": concurrency,
            "duration_s": duration_s,
            "max_batch_size": max_batch_size,
            "batch_wait_timeout_ms": batch_wait_timeout_ms,
        }
        _record_and_report(results, entry_extra, stats, results_path)
    print(f"Tune sweep complete -- wrote {len(results)} result(s) to {results_path}")


def run(
    *,
    mode: str,
    ssh_key: str,
    concurrencies: list[int],
    duration_s: float,
    max_batch_size: int | None,
    batch_wait_timeout_ms: float | None,
    results_path: Path,
    keep_pod: bool,
    tune_configs: list[tuple[int, float]] | None = None,
    tune_concurrency: int = 200,
) -> int:
    secrets.load_runpod_key()
    hangar.init(os.environ[secrets.RUNPOD_ENV_VAR])

    spec = hangar.PodSpec(
        name=_POD_NAME,
        image=_IMAGE,
        gpu_type_id=config.RUNPOD_GPU_TYPE_ID,
        disk_gb=_DISK_GB,
        ports=["22/tcp"],
        device_env_key="GLINER_DEVICE",
        device_env_value="cuda",
        extra_env={"PUBLIC_KEY": _read_public_key(ssh_key)},
    )

    pod_id = hangar.start_pod(spec)
    print(f"Pod {pod_id} starting ({config.RUNPOD_GPU_TYPE_ID})...")

    try:
        ssh_direct = _wait_for_pod_ready(pod_id)
        print(f"Pod {pod_id} ready, SSH at {ssh_direct['host']}:{ssh_direct['port']}")

        _wait_for_ssh_connectable(ssh_direct, ssh_key)
        _cuda_preflight(ssh_direct, ssh_key)
        _install_deps(ssh_direct, ssh_key)

        if mode == "smoke":
            _smoke_test(ssh_direct, ssh_key)
            print("Smoke test passed.")
        elif mode == "tune":
            _run_tune(
                ssh_direct,
                ssh_key,
                configs=tune_configs or [],
                concurrency=tune_concurrency,
                duration_s=duration_s,
                results_path=results_path,
            )
        else:
            _run_bench(
                ssh_direct,
                ssh_key,
                concurrencies=concurrencies,
                duration_s=duration_s,
                max_batch_size=max_batch_size,
                batch_wait_timeout_ms=batch_wait_timeout_ms,
                results_path=results_path,
            )

        return 0
    finally:
        if keep_pod:
            print(f"--keep-pod set: leaving pod {pod_id} running.")
        else:
            try:
                hangar.pod_action(pod_id, "terminate")
                print(f"Pod {pod_id} terminated.")
            except Exception:
                _LOGGER.exception("failed to terminate pod %s -- terminate it manually", pod_id)


def main(argv: list[str] | None = None) -> int:
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "mode",
        choices=["smoke", "bench", "tune"],
        help=(
            "smoke: one quick default-config check. bench: the concurrency sweep at one fixed "
            "tuning config. tune: a batch-tuning sweep (--configs) at one fixed concurrency."
        ),
    )
    parser.add_argument(
        "--ssh-key", required=True, help="Path to the SSH private key matching a key registered on the RunPod account."
    )
    parser.add_argument(
        "--concurrencies",
        type=str,
        default="50,200",
        help=(
            "Comma-separated concurrency points to sweep (bench mode only). Default 50,200: 50 for "
            "direct comparability with gliner_serve's own measured numbers, 200 as the low end of "
            "this project's real 200-500 concurrent-call target -- nothing measured before this "
            "suite existed has tested that shape."
        ),
    )
    parser.add_argument("--duration-s", type=float, default=25.0, help="Wall-clock duration per concurrency point.")
    parser.add_argument("--max-batch-size", type=int, default=None, help="Override GLINER_BATCH_MAX_SIZE for this run.")
    parser.add_argument(
        "--batch-wait-timeout-ms", type=float, default=None, help="Override GLINER_BATCH_WAIT_TIMEOUT_MS for this run."
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help=(
            "Local path to write results to (bench and tune modes). Defaults to "
            "benchmarks/gliner_batch_service/results.json for bench mode or tuning_sweep.json for "
            "tune mode, so running both modes with no override doesn't overwrite one sweep's "
            "results with the other's."
        ),
    )
    parser.add_argument(
        "--keep-pod", action="store_true", default=False, help="Don't terminate the pod on exit (debugging only -- the pod keeps billing)."
    )
    parser.add_argument(
        "--configs",
        type=str,
        default=None,
        help=(
            "Comma-separated max_batch_size:batch_wait_timeout_ms pairs to sweep (tune mode only, "
            "required for it), e.g. '16:20,32:20,64:20,16:10,32:10,64:10'."
        ),
    )
    parser.add_argument(
        "--concurrency",
        type=int,
        default=200,
        help=(
            "Single fixed concurrency to hold while sweeping --configs (tune mode only). Default "
            "200: the low end of this project's real target range, and the point the concurrency "
            "sweep found missing its latency budget at the placeholder tuning."
        ),
    )
    args = parser.parse_args(argv)
    if args.results_path is None:
        args.results_path = (
            Path("benchmarks/gliner_batch_service/tuning_sweep.json")
            if args.mode == "tune"
            else Path("benchmarks/gliner_batch_service/results.json")
        )
    concurrencies = [int(c.strip()) for c in args.concurrencies.split(",") if c.strip()]
    if not concurrencies or any(c < 1 for c in concurrencies):
        parser.error(f"--concurrencies must be a comma-separated list of positive integers, got {args.concurrencies!r}")
    if args.duration_s <= 0:
        parser.error(f"--duration-s must be positive, got {args.duration_s}")
    tune_configs: list[tuple[int, float]] = []
    if args.mode == "tune":
        if not args.configs:
            parser.error("tune mode requires --configs")
        for pair in args.configs.split(","):
            pair = pair.strip()
            if not pair:
                continue
            try:
                batch_size_str, wait_ms_str = pair.split(":")
                tune_configs.append((int(batch_size_str), float(wait_ms_str)))
            except ValueError:
                parser.error(f"--configs entries must be 'max_batch_size:batch_wait_timeout_ms', got {pair!r}")
        if any(size < 1 for size, _ in tune_configs) or any(wait < 0 for _, wait in tune_configs):
            parser.error("--configs values must be positive max_batch_size and non-negative batch_wait_timeout_ms")
        if args.concurrency < 1:
            parser.error(f"--concurrency must be positive, got {args.concurrency}")
    return run(
        mode=args.mode,
        ssh_key=args.ssh_key,
        concurrencies=concurrencies,
        duration_s=args.duration_s,
        max_batch_size=args.max_batch_size,
        batch_wait_timeout_ms=args.batch_wait_timeout_ms,
        results_path=args.results_path,
        keep_pod=args.keep_pod,
        tune_configs=tune_configs,
        tune_concurrency=args.concurrency,
    )


if __name__ == "__main__":
    raise SystemExit(main())
