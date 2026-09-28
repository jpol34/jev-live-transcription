"""Stands up a RunPod A100 pod, installs `gliner[serve]`, and runs one of three modes against it:
a quick smoke test, the full batch-window/quantization benchmark matrix (`run_matrix.py`), or the
blocking-handler/batch-churn latency diagnosis (`diagnose.py`) -- retrieving `results.json`/
`diagnosis.json` for the matrix/diagnose modes. Always terminates the pod on exit, success or
failure.

Mirrors `src/jev_live_transcription/gpu_run.py`'s pod-lifecycle pattern (create/resume via
`hangar`, poll for `RUNNING` + `ssh.direct`, retry real SSH connectability rather than trusting
metadata presence alone), trimmed to what this benchmark needs -- no capture DB, no
call/gliner-concurrency args, no LLM baseline key. This script and everything it drives is a
prototype for a hypothetical future production GLiNER-serving service, not a change to this
repo's own extraction pipeline: it only reads `config.RUNPOD_GPU_TYPE_ID` and
`secrets.load_runpod_key()` from that package, never `gliner_pipeline`/`pipeline_core`.

Two details of this base image are load-bearing and easy to drop by accident:

- `PUBLIC_KEY` must be injected via `extra_env` -- the pod image's own startup script only starts
  sshd when that env var is present. Without it every SSH attempt looks identical to a slow boot
  for the entire wait window.
- `pip install` on this image needs `--break-system-packages` -- it's PEP 668
  externally-managed (Debian base), same as the image this repo's own `Dockerfile` builds from.
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
# uses (confirmed against the installed gliner[serve] package, not guessed) -- imported rather than
# duplicated so the warmup/readiness check here exercises the exact same payload shape as the real
# load-tested traffic in `run_matrix.py`. `_pod_ssh` is the SSH/scp helper module shared with the
# sibling `gliner_batch_service/pod_bench.py`, one level up.
_THIS_DIR = Path(__file__).resolve().parent
for _extra_path in (_THIS_DIR, _THIS_DIR.parent):
    if str(_extra_path) not in sys.path:
        sys.path.insert(0, str(_extra_path))
import load_test  # noqa: E402
from _pod_ssh import run_ssh as _run_ssh  # noqa: E402
from _pod_ssh import ssh_target as _ssh_target  # noqa: E402
from _pod_ssh import wait_for_ssh_connectable as _wait_for_ssh_connectable  # noqa: E402

_LOGGER = logging.getLogger(__name__)

_IMAGE = "runpod/pytorch:1.0.2-cu1281-torch280-ubuntu2404"
_POD_NAME = "gliner-serve-bench"
_DISK_GB = 20
_MODEL_NAME = "urchade/gliner_medium-v2.1"
_SERVE_PORT = 8000
_REMOTE_DIR = "/root/gliner_serve_bench"
_REMOTE_LOG_PATH = f"{_REMOTE_DIR}/gliner-serve.log"
_REMOTE_PID_PATH = f"{_REMOTE_DIR}/gliner-serve.pid"
_REMOTE_RESULTS_PATH = f"{_REMOTE_DIR}/results.json"
_REMOTE_DIAGNOSIS_PATH = f"{_REMOTE_DIR}/diagnosis.json"

# Generous: the first real request against a freshly-started replica pays a one-time torch.compile
# warmup cost observed directly to exceed 4 minutes on this model/GPU (dynamo hit its recompile
# limit during warmup) -- a real finding about gliner.serve's default config, not a bug to paper
# over by shrinking this number.
_SERVER_READY_TIMEOUT_S = 600.0


def _install_deps(ssh_direct: dict, ssh_key: str) -> None:
    result = _run_ssh(
        ssh_direct,
        ssh_key,
        f"mkdir -p {_REMOTE_DIR} && pip install --no-cache-dir --break-system-packages 'gliner[serve]' aiohttp",
        timeout_s=600.0,
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to install gliner[serve]/aiohttp: {result.stderr}")
    print("gliner[serve] + aiohttp installed.")


def _start_server(
    ssh_direct: dict,
    ssh_key: str,
    *,
    batch_wait_ms: int,
    dtype: str | None = None,
    quantization: str | None = None,
) -> str:
    """Launches gliner.serve and blocks until it answers a real warmup request, returning that
    request's response body (the readiness check IS a real inference call, so its response is
    reused by callers that just need proof the server actually works, rather than firing a second,
    redundant request)."""
    cmd = f"python -m gliner.serve --model {_MODEL_NAME} --batch-wait-timeout-ms {batch_wait_ms}"
    if dtype:
        cmd += f" --dtype {dtype}"
    if quantization:
        cmd += f" --quantization {quantization}"
    # `nohup` alone does not background anything -- it only makes the process immune to SIGHUP.
    # The `&` must sit directly after the single command being backgrounded, not after a `&&`
    # chain: `A && nohup B &` backgrounds the *whole* `A && B` compound as one job, so `nohup B`
    # still runs synchronously inside it and the SSH channel blocks until B exits (i.e. forever,
    # for a server). All paths here are already absolute, so `cd` is skipped rather than working
    # around the same trap with `cd DIR && nohup ... &`.
    remote_command = (
        f"mkdir -p {_REMOTE_DIR}; nohup {cmd} < /dev/null > {_REMOTE_LOG_PATH} 2>&1 & "
        f"echo $! > {_REMOTE_PID_PATH}; disown; echo STARTED"
    )
    result = _run_ssh(ssh_direct, ssh_key, remote_command)
    if "STARTED" not in result.stdout:
        raise RuntimeError(f"failed to launch gliner.serve: {result.stderr}")

    # Confirmed ready by making a real inference request against `/gliner`, not a cheap probe of
    # `/` -- Ray Serve's HTTP proxy answers `/` as soon as it's up, well before the port is even
    # bound (Ray's own startup) or the GLiNERDeployment replica has finished loading the model and
    # its first-request torch.compile warmup (observed directly to take over a minute on this
    # model/GPU). Retried rather than a single long-timeout attempt: early attempts fail fast with
    # connection-refused (Ray hasn't bound the port yet) rather than hanging. `-f` makes curl treat
    # an HTTP error status as a failure (not just a non-empty body) and `-S` keeps its error message
    # visible despite `-s`. Each attempt gets a generous per-try timeout (covering an attempt that
    # lands mid torch.compile and genuinely takes a while) but `subprocess.TimeoutExpired` on any
    # single attempt is treated as "not ready yet" rather than fatal, so a slow-but-progressing
    # attempt doesn't abort the whole readiness wait. Uses `load_test.build_request` against a real
    # fixture window so this warmup request exercises the exact same payload shape as the real
    # load-tested traffic, not a hand-rolled shape that could silently diverge from it. The payload
    # is `shlex.quote`d, not hand-wrapped in single quotes -- real transcript text routinely
    # contains apostrophes (contractions like "I'm"), which would otherwise break out of a naively
    # single-quoted shell argument and fail with a shell syntax error, not an HTTP error.
    labels, windows = load_test.load_fixture()
    payload = json.dumps(load_test.build_request(windows[0], labels))
    quoted_payload = shlex.quote(payload)
    deadline = time.monotonic() + _SERVER_READY_TIMEOUT_S
    last_result: subprocess.CompletedProcess | None = None
    while time.monotonic() < deadline:
        remaining = max(5.0, deadline - time.monotonic())
        try:
            last_result = _run_ssh(
                ssh_direct,
                ssh_key,
                f"curl -sSf -X POST http://localhost:{_SERVE_PORT}{load_test.ROUTE_PREFIX} "
                f"-H 'Content-Type: application/json' -d {quoted_payload}",
                timeout_s=min(60.0, remaining),
            )
        except subprocess.TimeoutExpired:
            continue
        if last_result.returncode == 0 and last_result.stdout.strip():
            return last_result.stdout
        time.sleep(3.0)
    log_tail = _run_ssh(ssh_direct, ssh_key, f"tail -n 60 {_REMOTE_LOG_PATH} 2>&1")
    last_stderr = last_result.stderr if last_result else "(no attempt made)"
    raise TimeoutError(
        f"gliner.serve did not answer a real /gliner request within {_SERVER_READY_TIMEOUT_S}s "
        f"(last curl stderr: {last_stderr}) -- log tail:\n{log_tail.stdout}"
    )


def _stop_server(ssh_direct: dict, ssh_key: str) -> None:
    _run_ssh(
        ssh_direct,
        ssh_key,
        f"kill $(cat {_REMOTE_PID_PATH} 2>/dev/null) 2>/dev/null; pkill -f gliner.serve 2>/dev/null; true",
    )


def _smoke_test(ssh_direct: dict, ssh_key: str) -> None:
    """Starts the default-config server, confirms the request schema by inspecting the installed
    package (not guessed from `--help`, which only documents server startup flags), and prints the
    real warmup response `_start_server` already had to make to confirm readiness -- a human sanity
    check before this script is ever trusted for the real matrix run."""
    schema_probe = _run_ssh(
        ssh_direct,
        ssh_key,
        "python -c \"import gliner.serve.server as s, inspect; print(inspect.getsourcefile(s))\" 2>&1",
    )
    print(f"gliner[serve] server module location:\n{schema_probe.stdout}{schema_probe.stderr}")

    help_probe = _run_ssh(ssh_direct, ssh_key, "python -m gliner.serve --help 2>&1")
    print(f"gliner.serve --help:\n{help_probe.stdout}{help_probe.stderr}")

    response_body = _start_server(ssh_direct, ssh_key, batch_wait_ms=10)
    try:
        print(f"Smoke-test response (from the readiness-confirming request):\n{response_body}")
        try:
            parsed = json.loads(response_body)
        except json.JSONDecodeError as exc:
            raise RuntimeError(f"smoke-test response isn't valid JSON: {response_body!r}") from exc
        if not isinstance(parsed, dict) or not isinstance(parsed.get("entities"), list):
            raise RuntimeError(
                f"smoke-test response doesn't have the expected {{'entities': [...]}} shape: "
                f"{parsed!r} -- check the route/schema against the installed package before "
                "trusting the real matrix run"
            )
    finally:
        _stop_server(ssh_direct, ssh_key)


def _run_matrix(ssh_direct: dict, ssh_key: str, local_results_path: Path) -> None:
    scp_up_args = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-i", ssh_key, "-P", str(ssh_direct["port"])]
    target = f"{ssh_direct['username']}@{ssh_direct['host']}"
    # (local path, remote path relative to _REMOTE_DIR) -- the fixture keeps its `fixtures/`
    # subdirectory since `load_test.load_fixture` resolves it relative to the script's own
    # location, not the remote dir's top level.
    local_files = [
        (Path(__file__).parent / "fixtures" / "sample_windows.json", "fixtures/sample_windows.json"),
        (Path(__file__).parent / "load_test.py", "load_test.py"),
        (Path(__file__).parent / "run_matrix.py", "run_matrix.py"),
    ]
    missing = [local_path for local_path, _ in local_files if not local_path.exists()]
    if missing:
        raise FileNotFoundError(f"matrix mode needs these files built first: {missing}")

    _run_ssh(ssh_direct, ssh_key, f"mkdir -p {_REMOTE_DIR}/fixtures")
    for local_path, remote_rel_path in local_files:
        result = subprocess.run(
            ["scp", *scp_up_args, str(local_path), f"{target}:{_REMOTE_DIR}/{remote_rel_path}"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"failed to upload {remote_rel_path}: {result.stderr}")

    # Streamed rather than captured (unlike every other `_run_ssh` call in this file): this
    # invocation runs the full multi-config sweep and can take the better part of an hour, and
    # `_run_ssh`'s `capture_output=True` would buffer all of that output silently until the whole
    # thing finishes -- leaving no way to notice a stalled config short of waiting out the full
    # timeout. `stdout=None`/`stderr=None` let the child inherit this process's own streams so
    # `run_matrix.py`'s own per-config progress prints appear live.
    run_result = subprocess.run(
        [
            "ssh",
            *_ssh_target(ssh_key),
            "-p",
            str(ssh_direct["port"]),
            target,
            f"cd {_REMOTE_DIR} && python run_matrix.py --results-path {_REMOTE_RESULTS_PATH}",
        ],
        timeout=3600.0,
    )
    if run_result.returncode != 0:
        raise RuntimeError(f"run_matrix.py failed on the pod (exit code {run_result.returncode})")

    local_results_path.parent.mkdir(parents=True, exist_ok=True)
    scp_down_args = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-i", ssh_key, "-P", str(ssh_direct["port"])]
    result = subprocess.run(
        ["scp", *scp_down_args, f"{target}:{_REMOTE_RESULTS_PATH}", str(local_results_path)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to retrieve results.json: {result.stderr}")
    print(f"Results retrieved to {local_results_path}")


def _run_diagnose(ssh_direct: dict, ssh_key: str, local_results_path: Path) -> None:
    """Upload diagnose.py + its dependencies, run it in one blocking SSH call, and retrieve
    diagnosis.json -- mirrors `_run_matrix`'s upload/execute/retrieve pattern exactly."""
    scp_up_args = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-i", ssh_key, "-P", str(ssh_direct["port"])]
    target = f"{ssh_direct['username']}@{ssh_direct['host']}"
    local_files = [
        (Path(__file__).parent / "fixtures" / "sample_windows.json", "fixtures/sample_windows.json"),
        (Path(__file__).parent / "load_test.py", "load_test.py"),
        (Path(__file__).parent / "diagnose.py", "diagnose.py"),
    ]
    missing = [local_path for local_path, _ in local_files if not local_path.exists()]
    if missing:
        raise FileNotFoundError(f"diagnose mode needs these files built first: {missing}")

    _run_ssh(ssh_direct, ssh_key, f"mkdir -p {_REMOTE_DIR}/fixtures")
    for local_path, remote_rel_path in local_files:
        result = subprocess.run(
            ["scp", *scp_up_args, str(local_path), f"{target}:{_REMOTE_DIR}/{remote_rel_path}"],
            capture_output=True,
            encoding="utf-8",
            errors="replace",
        )
        if result.returncode != 0:
            raise RuntimeError(f"failed to upload {remote_rel_path}: {result.stderr}")

    # Streamed rather than captured, same reasoning as `_run_matrix`: this invocation runs the
    # full two-phase diagnosis plus the int8 retry and can take a while, and `_run_ssh`'s
    # `capture_output=True` would buffer all of that silently until the whole thing finishes.
    run_result = subprocess.run(
        [
            "ssh",
            *_ssh_target(ssh_key),
            "-p",
            str(ssh_direct["port"]),
            target,
            f"cd {_REMOTE_DIR} && python diagnose.py --results-path {_REMOTE_DIAGNOSIS_PATH}",
        ],
        timeout=3600.0,
    )
    if run_result.returncode != 0:
        raise RuntimeError(f"diagnose.py failed on the pod (exit code {run_result.returncode})")

    local_results_path.parent.mkdir(parents=True, exist_ok=True)
    scp_down_args = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-i", ssh_key, "-P", str(ssh_direct["port"])]
    result = subprocess.run(
        ["scp", *scp_down_args, f"{target}:{_REMOTE_DIAGNOSIS_PATH}", str(local_results_path)],
        capture_output=True,
        encoding="utf-8",
        errors="replace",
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to retrieve diagnosis.json: {result.stderr}")
    print(f"Diagnosis retrieved to {local_results_path}")


def run(*, mode: str, ssh_key: str, results_path: Path, keep_pod: bool) -> int:
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
        elif mode == "matrix":
            _run_matrix(ssh_direct, ssh_key, results_path)
        else:
            _run_diagnose(ssh_direct, ssh_key, results_path)

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
        choices=["smoke", "matrix", "diagnose"],
        help="smoke: one quick default-config check. matrix: the full sweep. diagnose: the latency-pathology diagnosis.",
    )
    parser.add_argument(
        "--ssh-key",
        required=True,
        help="Path to the SSH private key matching a key registered on the RunPod account.",
    )
    parser.add_argument(
        "--results-path",
        type=Path,
        default=None,
        help=(
            "Local path to copy the pod's results.json/diagnosis.json back to (matrix/diagnose "
            "modes only). Defaults to benchmarks/gliner_serve/results.json for matrix and "
            "benchmarks/gliner_serve/diagnosis.json for diagnose -- kept mode-specific so running "
            "one mode without overriding this flag can't silently overwrite the other's output."
        ),
    )
    parser.add_argument(
        "--keep-pod",
        action="store_true",
        default=False,
        help="Don't terminate the pod on exit (debugging only -- the pod keeps billing).",
    )
    args = parser.parse_args(argv)
    results_path = args.results_path
    if results_path is None:
        default_name = "diagnosis.json" if args.mode == "diagnose" else "results.json"
        results_path = Path(f"benchmarks/gliner_serve/{default_name}")
    return run(mode=args.mode, ssh_key=args.ssh_key, results_path=results_path, keep_pod=args.keep_pod)


if __name__ == "__main__":
    raise SystemExit(main())
