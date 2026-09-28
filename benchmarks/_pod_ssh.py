"""Shared SSH/scp helpers for this repo's `pod_bench.py` scripts (`benchmarks/gliner_serve/`,
`benchmarks/gliner_batch_service/`) -- extracted rather than left duplicated across both, the same
reasoning that pulled RunPod pod-lifecycle code out into `hangar` (see `~/.claude/CLAUDE.md`'s
RunPod GPU pods section): a fix to SSH/scp handling (encoding, timeouts) made in one copy but not
the other silently drifts the two out of sync.

Standalone (no `jev_live_transcription` import) so a `pod_bench.py` can add this file's directory
to `sys.path` and import it the same way it already imports its own sibling `load_test.py`.
"""

from __future__ import annotations

import subprocess
import time

SSH_CONNECT_TIMEOUT_S = 300.0
SSH_CONNECT_POLL_S = 5.0


def ssh_target(ssh_key: str) -> list[str]:
    # `-n` redirects the local ssh client's own stdin from /dev/null. Without it, a non-interactive
    # `ssh host 'cmd &'` can hang past the backgrounded command finishing: the client keeps the
    # channel open waiting on local stdin activity that never comes, regardless of the remote
    # command's own stdout/stderr redirection.
    return ["-o", "StrictHostKeyChecking=accept-new", "-o", "BatchMode=yes", "-n", "-i", ssh_key]


def run_ssh(
    ssh_direct: dict, ssh_key: str, command: str, *, timeout_s: float = 30.0
) -> subprocess.CompletedProcess:
    target = f"{ssh_direct['username']}@{ssh_direct['host']}"
    return subprocess.run(
        ["ssh", *ssh_target(ssh_key), "-p", str(ssh_direct["port"]), target, command],
        capture_output=True,
        # Explicit UTF-8 rather than `text=True`'s platform-default decoding: on Windows that
        # default is cp1252, which crashes decoding remote output containing multi-byte UTF-8
        # sequences (observed directly -- pip's install progress bar during dependency installs).
        # `errors="replace"` keeps a decode hiccup from crashing the whole SSH call outright.
        encoding="utf-8",
        errors="replace",
        timeout=timeout_s,
    )


def wait_for_ssh_connectable(
    ssh_direct: dict,
    ssh_key: str,
    *,
    timeout_s: float = SSH_CONNECT_TIMEOUT_S,
    poll_s: float = SSH_CONNECT_POLL_S,
) -> None:
    deadline = time.monotonic() + timeout_s
    last_result: subprocess.CompletedProcess | None = None
    while time.monotonic() < deadline:
        last_result = run_ssh(ssh_direct, ssh_key, "true", timeout_s=10.0)
        if last_result.returncode == 0:
            return
        time.sleep(poll_s)
    stderr = last_result.stderr if last_result else "(no attempt made)"
    raise TimeoutError(f"SSH never became connectable within {timeout_s}s: {stderr}")


def scp_up(
    ssh_direct: dict, ssh_key: str, local_path, remote_path: str, *, recursive: bool = False
) -> None:
    target = f"{ssh_direct['username']}@{ssh_direct['host']}:{remote_path}"
    args = [
        "-o",
        "StrictHostKeyChecking=accept-new",
        "-o",
        "BatchMode=yes",
        "-i",
        ssh_key,
        "-P",
        str(ssh_direct["port"]),
    ]
    if recursive:
        args.append("-r")
    result = subprocess.run(
        ["scp", *args, str(local_path), target], capture_output=True, encoding="utf-8", errors="replace"
    )
    if result.returncode != 0:
        raise RuntimeError(f"failed to upload {local_path} -> {remote_path}: {result.stderr}")
