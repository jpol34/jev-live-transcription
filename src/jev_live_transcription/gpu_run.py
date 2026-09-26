"""`jlt gpu-run`: runs the benchmark on a real RunPod GPU pod instead of this machine's CPU.

Creates (or resumes) a pod via `hangar`, waits for it to be SSH-reachable, runs a fast CUDA
preflight, then launches `jlt batch` on the pod detached over SSH (not held open for the run's
full duration -- a dropped local network connection must not kill a real, paid, potentially
hours-long remote run) and polls for completion. The resulting capture DB is copied back over scp.
The pod is always terminated on the way out, success or failure, unless `--keep-pod` -- this is a
dev-benchmarking tool, so there's no data-center pinning, network volume, or GPU-type fallback
list here: `config.RUNPOD_GPU_TYPE_ID` is a single best-effort choice, and a capacity failure is
just an error to retry, not something this code works around.
"""

from __future__ import annotations

import json
import logging
import os
import subprocess
import time
from pathlib import Path

import hangar

from . import config, secrets

_LOGGER = logging.getLogger(__name__)

_IMAGE = "ghcr.io/jpol34/jev-live-transcription:gpu-test"
_POD_NAME = "jev-live-transcription-gpu-run"
_REMOTE_DB_PATH = "/root/benchmark.sqlite3"
_REMOTE_LOG_PATH = "/root/gpu-run.log"
_REMOTE_EXIT_MARKER = "/root/gpu-run.exit"

_POD_READY_TIMEOUT_S = 600.0
_POD_READY_POLL_S = 5.0
_COMPLETION_POLL_S = 15.0


def _load_state(state_path: Path) -> dict:
    if not state_path.exists():
        return {}
    return json.loads(state_path.read_text())


def _save_state(state_path: Path, state: dict) -> None:
    state_path.parent.mkdir(parents=True, exist_ok=True)
    state_path.write_text(json.dumps(state, indent=2))


def _ssh_target(ssh_direct: dict, ssh_key: str | None) -> list[str]:
    base = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes"]
    if ssh_key:
        base += ["-i", ssh_key]
    return base


def _run_ssh(ssh_direct: dict, ssh_key: str | None, command: str, *, timeout_s: float = 30.0) -> subprocess.CompletedProcess:
    target = f"{ssh_direct['username']}@{ssh_direct['host']}"
    return subprocess.run(
        ["ssh", *_ssh_target(ssh_direct, ssh_key), "-p", str(ssh_direct["port"]), target, command],
        capture_output=True,
        text=True,
        timeout=timeout_s,
    )


def _wait_for_pod_ready(pod_id: str) -> dict:
    """Polls `get_pod` until the pod reports `RUNNING` with live runtime info and a direct SSH
    target. The proxy SSH method (`ssh.proxy`) RunPod also returns requires an interactive PTY and
    can't run a scripted, non-interactive command -- only `ssh.direct` (the pod's own public
    IP/port, populated once its container has actually booted) works for that."""
    deadline = time.monotonic() + _POD_READY_TIMEOUT_S
    while time.monotonic() < deadline:
        pod = hangar.get_pod(pod_id)
        if pod is None:
            raise RuntimeError(f"pod {pod_id} disappeared while waiting for it to become ready")
        ssh_direct = (pod.get("ssh") or {}).get("direct")
        if pod.get("status") == "RUNNING" and pod.get("runtime") and ssh_direct:
            return ssh_direct
        time.sleep(_POD_READY_POLL_S)
    raise TimeoutError(f"pod {pod_id} did not become SSH-ready within {_POD_READY_TIMEOUT_S}s")


def _cuda_preflight(ssh_direct: dict, ssh_key: str | None) -> None:
    result = _run_ssh(ssh_direct, ssh_key, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if result.returncode != 0:
        raise RuntimeError(f"CUDA preflight failed (nvidia-smi rc={result.returncode}): {result.stderr}")
    _LOGGER.info("CUDA preflight ok: %s", result.stdout.strip())


def _launch_batch_detached(
    ssh_direct: dict,
    ssh_key: str | None,
    *,
    subset: int | None,
    call_concurrency: int,
    gliner_concurrency: int,
    enable_llm_baseline: bool,
) -> None:
    batch_cmd = (
        f"jlt batch --db-path {_REMOTE_DB_PATH} "
        f"--call-concurrency {call_concurrency} --gliner-concurrency {gliner_concurrency}"
    )
    if subset is not None:
        batch_cmd += f" --subset {subset}"
    if enable_llm_baseline:
        batch_cmd += " --enable-llm-baseline"

    remote_command = (
        f"nohup sh -c '{batch_cmd}; echo $? > {_REMOTE_EXIT_MARKER}' "
        f"> {_REMOTE_LOG_PATH} 2>&1 & disown; echo LAUNCHED"
    )
    result = _run_ssh(ssh_direct, ssh_key, remote_command)
    if "LAUNCHED" not in result.stdout:
        raise RuntimeError(f"failed to launch jlt batch on the pod: {result.stderr}")


def _wait_for_completion(ssh_direct: dict, ssh_key: str | None) -> int:
    """Polls for `_REMOTE_EXIT_MARKER` by reconnecting periodically rather than holding one SSH
    session open for the run's full (potentially hours-long) duration -- a transient local network
    blip must not be mistaken for the remote run failing."""
    while True:
        result = _run_ssh(
            ssh_direct, ssh_key, f"cat {_REMOTE_EXIT_MARKER} 2>/dev/null || echo NOT_DONE"
        )
        output = result.stdout.strip()
        if output != "NOT_DONE" and output:
            try:
                return int(output)
            except ValueError:
                raise RuntimeError(f"unexpected exit marker contents: {output!r}") from None
        time.sleep(_COMPLETION_POLL_S)


def _retrieve_db(ssh_direct: dict, ssh_key: str | None, local_db_path: Path) -> None:
    local_db_path.parent.mkdir(parents=True, exist_ok=True)
    scp_args = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-P", str(ssh_direct["port"])]
    if ssh_key:
        scp_args += ["-i", ssh_key]
    source = f"{ssh_direct['username']}@{ssh_direct['host']}:{_REMOTE_DB_PATH}"
    result = subprocess.run(["scp", *scp_args, source, str(local_db_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"failed to retrieve capture DB from pod: {result.stderr}")


def run_gpu(
    *,
    subset: int | None,
    db_path: Path,
    call_concurrency: int,
    gliner_concurrency: int,
    enable_llm_baseline: bool,
    pod_state_path: Path,
    ssh_key: str | None,
    keep_pod: bool,
) -> int:
    secrets.load_typesafe_key()
    if enable_llm_baseline:
        secrets.load_openai_key()
    secrets.load_runpod_key()

    hangar.init(os.environ[secrets.RUNPOD_ENV_VAR])

    state = _load_state(pod_state_path)
    extra_env = {"TYPESAFE_API_KEY": os.environ[secrets.TYPESAFE_ENV_VAR]}
    if enable_llm_baseline:
        extra_env["OPENAI_API_KEY"] = os.environ[secrets.OPENAI_ENV_VAR]

    spec = hangar.PodSpec(
        name=_POD_NAME,
        image=_IMAGE,
        gpu_type_id=config.RUNPOD_GPU_TYPE_ID,
        disk_gb=30,
        ports=["22/tcp"],
        device_env_key="GLINER_DEVICE",
        device_env_value="cuda",
        pod_id=state.get("pod_id"),
        extra_env=extra_env,
    )

    pod_id = hangar.start_pod(spec)
    _save_state(pod_state_path, {"pod_id": pod_id, "gpu_type_id": config.RUNPOD_GPU_TYPE_ID})
    print(f"Pod {pod_id} starting ({config.RUNPOD_GPU_TYPE_ID})...")

    try:
        ssh_direct = _wait_for_pod_ready(pod_id)
        print(f"Pod {pod_id} ready, SSH at {ssh_direct['host']}:{ssh_direct['port']}")

        _cuda_preflight(ssh_direct, ssh_key)

        _launch_batch_detached(
            ssh_direct,
            ssh_key,
            subset=subset,
            call_concurrency=call_concurrency,
            gliner_concurrency=gliner_concurrency,
            enable_llm_baseline=enable_llm_baseline,
        )
        print("jlt batch launched on the pod, polling for completion...")

        exit_code = _wait_for_completion(ssh_direct, ssh_key)
        print(f"Remote jlt batch finished with exit code {exit_code}")

        _retrieve_db(ssh_direct, ssh_key, db_path)
        print(f"Capture DB retrieved to {db_path}")

        return exit_code
    finally:
        if keep_pod:
            print(f"--keep-pod set: leaving pod {pod_id} running.")
        else:
            try:
                hangar.pod_action(pod_id, "terminate")
                print(f"Pod {pod_id} terminated.")
            except Exception:
                _LOGGER.exception("failed to terminate pod %s -- terminate it manually", pod_id)
