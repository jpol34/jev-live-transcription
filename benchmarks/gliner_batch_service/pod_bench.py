"""Stands up a RunPod A100 pod, installs this repo's own package (not a third-party stack), starts
`jlt serve`, and runs a closed-loop load-test sweep against it across `--concurrencies` -- retrieving
`results.json` with one entry per concurrency point, rewritten after each completes so a crash
partway through the sweep doesn't lose already-collected data. Always terminates the pod on exit,
success or failure.

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
# payload shape as the real load-tested traffic.
_THIS_DIR = Path(__file__).resolve().parent
if str(_THIS_DIR) not in sys.path:
    sys.path.insert(0, str(_THIS_DIR))
import load_test  # noqa: E402

_LOGGER = logging.getLogger(__name__)

_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
_POD_NAME = "gliner-batch-service-bench"
_DISK_GB = 20
_SERVE_PORT = 8000
_REMOTE_DIR = "/root/gliner_batch_service_bench"
_REMOTE_LOG_PATH = f"{_REMOTE_DIR}/jlt-serve.log"
_REMOTE_PID_PATH = f"{_REMOTE_DIR}/jlt-serve.pid"
_REMOTE_FIXTURE_PATH = f"{_REMOTE_DIR}/fixtures/sample_windows.json"

_SSH_CONNECT_TIMEOUT_S = 300.0
_SSH_CONNECT_POLL_S = 5.0
# Generous: covers a cold HuggingFace Hub download of the checkpoint on top of the model's own
# load time -- this pod installs the project fresh, so unlike a pre-baked Docker image, the
# checkpoint is never already cached on first run.
_SERVER_READY_TIMEOUT_S = 600.0


def _ssh_target(ssh_key: str) -> list[str]:
    # `-n` redirects the local ssh client's own stdin from /dev/null. Without it, a non-interactive
    # `ssh host 'cmd &'` can hang past the backgrounded command finishing: the client keeps the
    # channel open waiting on local stdin activity that never comes, regardless of the remote
    # command's own stdout/stderr redirection.
    return ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-n", "-i", ssh_key]


def _run_ssh(
    ssh_direct: dict, ssh_key: str, command: str, *, timeout_s: float = 30.0
) -> subprocess.CompletedProcess:
    target = f"{ssh_direct['username']}@{ssh_direct['host']}"
    return subprocess.run(
        ["ssh", *_ssh_target(ssh_key), "-p", str(ssh_direct["port"]), target, command],
        capture_output=True,
        # Explicit UTF-8 rather than `text=True`'s platform-default decoding: on Windows that
        # default is cp1252, which crashes decoding remote output containing multi-byte UTF-8
        # sequences (observed directly in the gliner_serve suite's own pip install progress bar).
        # `errors="replace"` keeps a decode hiccup from crashing the whole SSH call outright.
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )


def _wait_for_ssh_connectable(ssh_direct: dict, ssh_key: str) -> None:
    deadline = time.monotonic() + _SSH_CONNECT_TIMEOUT_S
    last_result: subprocess.CompletedProcess | None = None
    while time.monotonic() < deadline:
        last_result = _run_ssh(ssh_direct, ssh_key, "true", timeout_s=10.0)
        if last_result.returncode == 0:
            return
        time.sleep(_SSH_CONNECT_POLL_S)
    stderr = last_result.stderr if last_result else "(no attempt made)"
    raise TimeoutError(f"SSH never became connectable within {_SSH_CONNECT_TIMEOUT_S}s: {stderr}")


def _scp_up(ssh_direct: dict, ssh_key: str, local_path: Path, remote_path: str, *, recursive: bool = False) -> None:
    target = f"{ssh_direct['username']}@{ssh_direct['host']}:{remote_path}"
    args = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-i", ssh_key, "-P", str(ssh_direct["port"])]
    if recursive:
        args.append("-r")
    result = subprocess.run(
        ["scp", *args, str(local_path), target], capture_output=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to upload {local_path} -> {remote_path}: {result.stderr}")


def _install_deps(ssh_direct: dict, ssh_key: str) -> None:
    """Uploads this project's source and installs it fresh via `uv pip install -e .` -- the exact
    commit under test, not whatever a separately-built Docker image happens to carry.
    """
    repo_root = _THIS_DIR.parent.parent
    _run_ssh(ssh_direct, ssh_key, f"mkdir -p {_REMOTE_DIR}")
    _scp_up(ssh_direct, ssh_key, repo_root / "pyproject.toml", f"{_REMOTE_DIR}/pyproject.toml")
    _scp_up(ssh_direct, ssh_key, repo_root / "src", f"{_REMOTE_DIR}/src", recursive=True)

    result = _run_ssh(
        ssh_direct,
        ssh_key,
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
    # server).
    remote_command = (
        f"cd {_REMOTE_DIR} && nohup {cmd} < /dev/null > {_REMOTE_LOG_PATH} 2>&1 & "
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
    _stop_server(ssh_direct, ssh_key)


def _write_results(results_path: Path, results: list[dict]) -> None:
    results_path.parent.mkdir(parents=True, exist_ok=True)
    results_path.write_text(json.dumps(results, indent=2))


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
            remote_command = (
                f"cd {_REMOTE_DIR} && python load_test.py --concurrency {concurrency} "
                f"--duration-s {duration_s} --url http://localhost:{_SERVE_PORT} "
                f"--fixture-path {_REMOTE_FIXTURE_PATH}"
            )
            run_result = _run_ssh(
                ssh_direct, ssh_key, remote_command, timeout_s=duration_s + 60.0
            )
            if run_result.returncode != 0:
                raise RuntimeError(
                    f"load_test.py failed at concurrency={concurrency}: {run_result.stderr}"
                )
            stats = json.loads(run_result.stdout.strip().splitlines()[-1])
            entry = {
                "concurrency": concurrency,
                "duration_s": duration_s,
                "max_batch_size": max_batch_size,
                "batch_wait_timeout_ms": batch_wait_timeout_ms,
                **stats,
            }
            results.append(entry)
            _write_results(results_path, results)
            print(f"  -> p50={stats['p50_ms']:.1f}ms p95={stats['p95_ms']:.1f}ms throughput={stats['throughput_req_s']:.1f}req/s")
    finally:
        _stop_server(ssh_direct, ssh_key)
    print(f"Bench complete -- wrote {len(results)} result(s) to {results_path}")


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
        "mode", choices=["smoke", "bench"], help="smoke: one quick default-config check. bench: the concurrency sweep."
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
        default=Path("benchmarks/gliner_batch_service/results.json"),
        help="Local path to write results.json to (bench mode only).",
    )
    parser.add_argument(
        "--keep-pod", action="store_true", default=False, help="Don't terminate the pod on exit (debugging only -- the pod keeps billing)."
    )
    args = parser.parse_args(argv)
    concurrencies = [int(c.strip()) for c in args.concurrencies.split(",") if c.strip()]
    return run(
        mode=args.mode,
        ssh_key=args.ssh_key,
        concurrencies=concurrencies,
        duration_s=args.duration_s,
        max_batch_size=args.max_batch_size,
        batch_wait_timeout_ms=args.batch_wait_timeout_ms,
        results_path=args.results_path,
        keep_pod=args.keep_pod,
    )


if __name__ == "__main__":
    raise SystemExit(main())
