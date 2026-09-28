"""`jlt gpu-run`: runs the benchmark on a real RunPod GPU pod instead of this machine's CPU.

Creates (or resumes) a pod via `hangar`, waits for it to be SSH-reachable, runs a fast CUDA
preflight, writes this run's secrets to a file on the pod for `jlt batch` to source (sshd doesn't
pass the container's own environment into an SSH session, so they aren't otherwise visible there),
then launches `jlt batch` on the pod detached over SSH (not held open for the run's full duration
-- a dropped local network connection must not kill a real, paid, potentially hours-long remote
run) and polls for completion. The resulting capture DB is copied back over scp.
The pod is always terminated on the way out, success or failure, unless `--keep-pod` -- this is a
dev-benchmarking tool, so there's no data-center pinning, network volume, or GPU-type fallback
list here: `config.RUNPOD_GPU_TYPE_ID` is a single best-effort choice, and a capacity failure is
just an error to retry, not something this code works around.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shlex
import sqlite3
import subprocess
import time
from pathlib import Path

import hangar

from . import config, secrets
from . import db as db_module

_LOGGER = logging.getLogger(__name__)

_IMAGE = "ghcr.io/jpol34/jev-live-transcription:gpu-test"
_POD_NAME = "jev-live-transcription-gpu-run"
_REMOTE_DB_PATH = "/root/benchmark.sqlite3"
_REMOTE_LOG_PATH = "/root/gpu-run.log"
_REMOTE_EXIT_MARKER = "/root/gpu-run.exit"
_REMOTE_ENV_PATH = "/root/.jlt_env"

_POD_READY_TIMEOUT_S = 600.0
_POD_READY_POLL_S = 5.0
_COMPLETION_POLL_S = 15.0
# Generous headroom for the pod image's own startup script to generate SSH host keys and start
# sshd after `ssh.direct` first appears in the pod's metadata -- that field is populated once
# RunPod's control plane assigns the port mapping, which can briefly precede the container's own
# startup script actually finishing. A `PUBLIC_KEY` env var missing from the pod entirely (see
# `_read_public_key`) produces the same symptom -- connection refused for the whole window -- so a
# timeout here does not by itself mean this is a slow-startup race; check the pod's own container
# logs before assuming a longer timeout is the fix.
_SSH_CONNECT_TIMEOUT_S = 300.0
_SSH_CONNECT_POLL_S = 5.0


def _read_public_key(ssh_key: str | None) -> str:
    """Return the contents of `<ssh_key>.pub`, the counterpart to the private key path this
    project's SSH/scp calls already use.

    The pod image's own startup script only creates `~/.ssh/authorized_keys`, generates SSH host
    keys, and starts sshd at all when a `PUBLIC_KEY` environment variable is present in the pod's
    environment. RunPod's REST API does not inject this on its own the way pod creation through the
    web console does, so this project must supply it explicitly on every pod creation -- without
    it, the pod comes up with no sshd running at all, which looks identical from the outside to a
    slow-starting sshd (both are a connection refused for the entire readiness wait).
    """
    if not ssh_key:
        raise ValueError(
            "--ssh-key is required: its matching <ssh-key>.pub file's contents are injected into "
            "the pod as PUBLIC_KEY, which the pod image's startup script requires to start sshd "
            "at all -- without it the pod comes up with no sshd running and every SSH attempt "
            "gets a connection refused for the entire wait window, indistinguishable from a slow "
            "startup."
        )
    pub_key_path = Path(f"{ssh_key}.pub")
    if not pub_key_path.exists():
        raise FileNotFoundError(f"no public key found at {pub_key_path} (expected alongside --ssh-key)")
    return pub_key_path.read_text().strip()


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


def _run_ssh(
    ssh_direct: dict,
    ssh_key: str | None,
    command: str,
    *,
    timeout_s: float = 30.0,
    input: str | None = None,
) -> subprocess.CompletedProcess:
    target = f"{ssh_direct['username']}@{ssh_direct['host']}"
    return subprocess.run(
        ["ssh", *_ssh_target(ssh_direct, ssh_key), "-p", str(ssh_direct["port"]), target, command],
        input=input,
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


def _wait_for_ssh_connectable(ssh_direct: dict, ssh_key: str | None) -> None:
    """Retries a trivial SSH command until it succeeds. `ssh.direct` in the pod's metadata reflects
    RunPod's port mapping, not whether sshd inside the container is actually up yet, so readiness
    has to be confirmed by an actual successful connection rather than the field's mere presence in
    the API response. A connection refused for the entire timeout window most likely means sshd
    never started at all (see `_read_public_key`), not that this loop needs more time."""
    deadline = time.monotonic() + _SSH_CONNECT_TIMEOUT_S
    last_result: subprocess.CompletedProcess | None = None
    while time.monotonic() < deadline:
        last_result = _run_ssh(ssh_direct, ssh_key, "true", timeout_s=10.0)
        if last_result.returncode == 0:
            return
        time.sleep(_SSH_CONNECT_POLL_S)
    stderr = last_result.stderr if last_result else "(no attempt made)"
    raise TimeoutError(f"SSH never became connectable within {_SSH_CONNECT_TIMEOUT_S}s: {stderr}")


def _cuda_preflight(ssh_direct: dict, ssh_key: str | None) -> None:
    result = _run_ssh(ssh_direct, ssh_key, "nvidia-smi --query-gpu=name,memory.total --format=csv,noheader")
    if result.returncode != 0:
        raise RuntimeError(f"CUDA preflight failed (nvidia-smi rc={result.returncode}): {result.stderr}")
    _LOGGER.info("CUDA preflight ok: %s", result.stdout.strip())


def _write_remote_env(ssh_direct: dict, ssh_key: str | None, env: dict[str, str]) -> None:
    """Writes `env` to `_REMOTE_ENV_PATH` on the pod as shell `export` lines, for `jlt batch` to
    source before running.

    sshd only passes through the client's `LANG`/`LC_*` environment into a session by default
    (`AcceptEnv` in `sshd_config`) -- a container's own `ENV` values (this project's secrets among
    them) are visible to the container's PID 1 and its own child processes, but not to a fresh SSH
    session, which gets a minimal environment of its own. The env content is sent over stdin rather
    than as a shell argument so a secret value is never visible in the pod's own process listing.
    """
    lines = "\n".join(f"export {key}={shlex.quote(value)}" for key, value in env.items())
    # `umask 077` before creating the file, rather than a separate `chmod 600` afterward, so there
    # is never a window where the file exists at the shell's default (often world-readable)
    # permissions.
    result = _run_ssh(ssh_direct, ssh_key, f"umask 077 && cat > {_REMOTE_ENV_PATH}", input=lines)
    if result.returncode != 0:
        raise RuntimeError(f"failed to write remote env file: {result.stderr}")


def _launch_batch_detached(
    ssh_direct: dict,
    ssh_key: str | None,
    *,
    subset: int | None,
    call_concurrency: int,
    gliner_concurrency: int,
    enable_llm_baseline: bool,
    enable_gliner_only: bool = False,
    enable_jev: bool = True,
) -> None:
    batch_cmd = (
        f"jlt batch --db-path {_REMOTE_DB_PATH} "
        f"--call-concurrency {call_concurrency} --gliner-concurrency {gliner_concurrency}"
    )
    if subset is not None:
        batch_cmd += f" --subset {subset}"
    if enable_llm_baseline:
        batch_cmd += " --enable-llm-baseline"
    if enable_gliner_only:
        batch_cmd += " --enable-gliner-only"
    if not enable_jev:
        batch_cmd += " --disable-jev"

    remote_command = (
        f"nohup sh -c '. {_REMOTE_ENV_PATH} && {batch_cmd}; echo $? > {_REMOTE_EXIT_MARKER}' "
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


def _verify_db(db_path: Path, expected_calls: int) -> str | None:
    """Return `None` if `db_path` looks like a complete, uncorrupted capture DB for
    `expected_calls` calls, else a human-readable reason it doesn't.

    A capture DB can scp over successfully (exit code 0) while still being unusable in two
    different ways: structurally corrupted (`PRAGMA integrity_check` catches this -- the pod's
    disk durability under a high `--call-concurrency` write rate isn't guaranteed the way a local
    SSD's would be), or structurally fine but silently incomplete (a well-formed but empty or
    partial DB, e.g. from a scp that copied nothing, passes `integrity_check` trivially, so the
    `calls` row count is checked against what this run actually asked for). Silently handing back
    either is worse than failing loud here, immediately after retrieval, rather than as a
    confusing `DatabaseError` or an inexplicably empty results table several steps later during
    scoring.
    """
    try:
        conn = db_module.connect(db_path)
        try:
            if conn.execute("PRAGMA integrity_check").fetchall() != [("ok",)]:
                return "failed PRAGMA integrity_check"
            (n_calls,) = conn.execute("SELECT COUNT(*) FROM calls").fetchone()
        finally:
            conn.close()
    except sqlite3.DatabaseError:
        return "is not a readable SQLite database"
    if n_calls != expected_calls:
        return f"has {n_calls} calls, expected {expected_calls}"
    return None


_REMOTE_RUNNING_LINE_RE = re.compile(r"^Running (\d+) call\(s\)", re.MULTILINE)


def _read_remote_expected_calls(ssh_direct: dict, ssh_key: str | None) -> int:
    """Return the call count `jlt batch` itself reported running, parsed from its own
    `"Running N call(s) -> ..."` log line (`cli.py`'s `_run_batch`) on the pod.

    Deriving this from the remote side's own report -- rather than recomputing it locally from
    `corpus.load_all()`/`--subset` -- means it's always exactly what that run actually did,
    regardless of whether the pod's baked-in corpus (`_IMAGE`) happens to match the local
    checkout's `output/transcripts`/`output/metadata` at the moment `run_gpu` is invoked.
    """
    result = _run_ssh(ssh_direct, ssh_key, f"cat {_REMOTE_LOG_PATH}")
    if result.returncode != 0:
        raise RuntimeError(f"failed to read remote log at {_REMOTE_LOG_PATH}: {result.stderr}")
    match = _REMOTE_RUNNING_LINE_RE.search(result.stdout)
    if not match:
        raise RuntimeError(
            f"could not find jlt batch's own 'Running N call(s)' line in the remote log at "
            f"{_REMOTE_LOG_PATH} -- can't verify the retrieved DB's completeness."
        )
    return int(match.group(1))


def _scp_db(ssh_direct: dict, ssh_key: str | None, local_db_path: Path) -> None:
    local_db_path.parent.mkdir(parents=True, exist_ok=True)
    scp_args = ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-P", str(ssh_direct["port"])]
    if ssh_key:
        scp_args += ["-i", ssh_key]
    source = f"{ssh_direct['username']}@{ssh_direct['host']}:{_REMOTE_DB_PATH}"
    result = subprocess.run(["scp", *scp_args, source, str(local_db_path)], capture_output=True, text=True)
    if result.returncode != 0:
        raise RuntimeError(f"failed to retrieve capture DB from pod: {result.stderr}")


def _retrieve_db(
    ssh_direct: dict, ssh_key: str | None, local_db_path: Path, expected_calls: int | None
) -> None:
    """Copy the pod's capture DB to `local_db_path`.

    `expected_calls` is `None` when the remote `jlt batch` itself reported a failure -- the DB is
    then retrieved best-effort, for debugging, on a single attempt with no verification, since a
    failed run's DB is expected to be incomplete and shouldn't be held to the same bar (or block
    retrieval entirely) the way a DB from a *successful* run should. When given, it's the number
    of `calls` rows a successful, complete DB must have (see `_read_remote_expected_calls`), and a
    mismatch is retried once before raising -- see `_verify_db`.
    """
    attempts = 2 if expected_calls is not None else 1
    problem: str | None = None
    for attempt in range(1, attempts + 1):
        _scp_db(ssh_direct, ssh_key, local_db_path)
        if expected_calls is None:
            return
        problem = _verify_db(local_db_path, expected_calls)
        if problem is None:
            return
        _LOGGER.warning(
            "retrieved capture DB at %s %s (attempt %d/%d)%s",
            local_db_path,
            problem,
            attempt,
            attempts,
            "" if attempt < attempts else " -- giving up",
        )

    raise RuntimeError(
        f"capture DB retrieved from the pod at {local_db_path} {problem} twice in a row -- likely "
        "a real problem on the pod itself rather than a transient transfer issue. Re-run with "
        "--keep-pod to inspect the pod's own copy before it's torn down."
    )


def run_gpu(
    *,
    subset: int | None,
    db_path: Path,
    call_concurrency: int,
    gliner_concurrency: int,
    enable_llm_baseline: bool,
    enable_gliner_only: bool = False,
    enable_jev: bool = True,
    pod_state_path: Path,
    ssh_key: str | None,
    keep_pod: bool,
) -> int:
    if enable_jev:
        secrets.load_typesafe_key()
    if enable_llm_baseline:
        secrets.load_openai_key()
    secrets.load_runpod_key()

    hangar.init(os.environ[secrets.RUNPOD_ENV_VAR])

    state = _load_state(pod_state_path)
    extra_env = {"PUBLIC_KEY": _read_public_key(ssh_key)}
    if enable_jev:
        extra_env["TYPESAFE_API_KEY"] = os.environ[secrets.TYPESAFE_ENV_VAR]
    if enable_llm_baseline:
        extra_env["OPENAI_API_KEY"] = os.environ[secrets.OPENAI_ENV_VAR]

    spec = hangar.PodSpec(
        name=_POD_NAME,
        image=_IMAGE,
        gpu_type_id=config.RUNPOD_GPU_TYPE_ID,
        # The image itself is ~38GB (RunPod's base CUDA/torch layers plus two baked-in GLiNER
        # checkpoints) -- disk_gb is the pod's container disk, which must hold the pulled+
        # extracted image with room to spare, not just the app's own runtime footprint. 30GB was
        # too small and left the pod stuck mid-pull with uptime never leaving 0 (confirmed live).
        disk_gb=60,
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

        _wait_for_ssh_connectable(ssh_direct, ssh_key)
        _cuda_preflight(ssh_direct, ssh_key)

        # PUBLIC_KEY is only relevant to the pod's own sshd startup, not to `jlt batch` -- excluded
        # here so the remote secrets file holds only what the batch process actually consumes.
        _write_remote_env(ssh_direct, ssh_key, {k: v for k, v in extra_env.items() if k != "PUBLIC_KEY"})
        _launch_batch_detached(
            ssh_direct,
            ssh_key,
            subset=subset,
            call_concurrency=call_concurrency,
            gliner_concurrency=gliner_concurrency,
            enable_llm_baseline=enable_llm_baseline,
            enable_gliner_only=enable_gliner_only,
            enable_jev=enable_jev,
        )
        print("jlt batch launched on the pod, polling for completion...")

        exit_code = _wait_for_completion(ssh_direct, ssh_key)
        print(f"Remote jlt batch finished with exit code {exit_code}")

        expected_calls = _read_remote_expected_calls(ssh_direct, ssh_key) if exit_code == 0 else None
        _retrieve_db(ssh_direct, ssh_key, db_path, expected_calls)
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
