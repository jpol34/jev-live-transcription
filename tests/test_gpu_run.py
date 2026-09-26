"""Mocked tests for gpu_run.py -- no real RunPod pod, no real SSH, no real secrets.

`hangar` is mocked at the module level (the object `gpu_run` imported), and `subprocess.run` is
mocked for the SSH/scp-shelling-out functions. Focus is teardown-on-failure (the pod must always
be terminated, success or failure, unless `--keep-pod`) and the pod-readiness/completion polling
loops, since those are the parts a real pod can't be used to test cheaply.
"""

from unittest.mock import Mock

import pytest

from jev_live_transcription import gpu_run


def _patch_secrets(monkeypatch):
    monkeypatch.setattr(gpu_run.secrets, "load_typesafe_key", Mock())
    monkeypatch.setattr(gpu_run.secrets, "load_openai_key", Mock())
    monkeypatch.setattr(gpu_run.secrets, "load_runpod_key", Mock())
    monkeypatch.setenv(gpu_run.secrets.TYPESAFE_ENV_VAR, "typesafe-test-key")
    monkeypatch.setenv(gpu_run.secrets.OPENAI_ENV_VAR, "openai-test-key")
    monkeypatch.setenv(gpu_run.secrets.RUNPOD_ENV_VAR, "runpod-test-key")


def _patch_hangar(monkeypatch, *, pod_id="pod-123"):
    ready_pod = {
        "status": "RUNNING",
        "runtime": {"uptime": 10},
        "ssh": {"direct": {"host": "1.2.3.4", "port": 2222, "username": "root"}},
    }
    monkeypatch.setattr(gpu_run.hangar, "init", Mock())
    monkeypatch.setattr(gpu_run.hangar, "start_pod", Mock(return_value=pod_id))
    monkeypatch.setattr(gpu_run.hangar, "get_pod", Mock(return_value=ready_pod))
    monkeypatch.setattr(gpu_run.hangar, "pod_action", Mock())
    return ready_pod


def _patch_ssh_flow(monkeypatch, *, exit_code=0):
    """Makes `_run_ssh` succeed for preflight/launch and immediately report completion."""

    def fake_run(cmd, **kwargs):
        joined = " ".join(cmd)
        if cmd[0] == "scp":
            return Mock(returncode=0, stdout="", stderr="")
        if joined.endswith(" true"):
            return Mock(returncode=0, stdout="", stderr="")
        if "nvidia-smi" in joined:
            return Mock(returncode=0, stdout="NVIDIA L40S, 49140 MiB", stderr="")
        if "nohup" in joined:
            return Mock(returncode=0, stdout="LAUNCHED\n", stderr="")
        if "cat" in joined:
            return Mock(returncode=0, stdout=f"{exit_code}\n", stderr="")
        raise AssertionError(f"unexpected command: {joined!r}")

    monkeypatch.setattr(gpu_run.subprocess, "run", Mock(side_effect=fake_run))


def test_run_gpu_happy_path_terminates_pod_and_returns_remote_exit_code(monkeypatch, tmp_path):
    _patch_secrets(monkeypatch)
    _patch_hangar(monkeypatch)
    _patch_ssh_flow(monkeypatch, exit_code=0)

    result = gpu_run.run_gpu(
        subset=None,
        db_path=tmp_path / "out.sqlite3",
        call_concurrency=1,
        gliner_concurrency=1,
        enable_llm_baseline=False,
        pod_state_path=tmp_path / "state.json",
        ssh_key=None,
        keep_pod=False,
    )

    assert result == 0
    gpu_run.hangar.pod_action.assert_called_once_with("pod-123", "terminate")


def test_run_gpu_terminates_pod_even_when_preflight_fails(monkeypatch, tmp_path):
    _patch_secrets(monkeypatch)
    _patch_hangar(monkeypatch)

    def fake_run(cmd, **kwargs):
        # The SSH-connectable check ("true") must succeed so the flow reaches the real CUDA
        # preflight, which is what this test actually exercises failing.
        if " ".join(cmd).endswith(" true"):
            return Mock(returncode=0, stdout="", stderr="")
        return Mock(returncode=1, stdout="", stderr="no CUDA device")

    monkeypatch.setattr(gpu_run.subprocess, "run", Mock(side_effect=fake_run))

    with pytest.raises(RuntimeError, match="CUDA preflight failed"):
        gpu_run.run_gpu(
            subset=None,
            db_path=tmp_path / "out.sqlite3",
            call_concurrency=1,
            gliner_concurrency=1,
            enable_llm_baseline=False,
            pod_state_path=tmp_path / "state.json",
            ssh_key=None,
            keep_pod=False,
        )

    gpu_run.hangar.pod_action.assert_called_once_with("pod-123", "terminate")


def test_run_gpu_keep_pod_skips_termination(monkeypatch, tmp_path):
    _patch_secrets(monkeypatch)
    _patch_hangar(monkeypatch)
    _patch_ssh_flow(monkeypatch, exit_code=0)

    gpu_run.run_gpu(
        subset=None,
        db_path=tmp_path / "out.sqlite3",
        call_concurrency=1,
        gliner_concurrency=1,
        enable_llm_baseline=False,
        pod_state_path=tmp_path / "state.json",
        ssh_key=None,
        keep_pod=True,
    )

    gpu_run.hangar.pod_action.assert_not_called()


def test_run_gpu_persists_pod_id_to_state_file(monkeypatch, tmp_path):
    _patch_secrets(monkeypatch)
    _patch_hangar(monkeypatch, pod_id="pod-456")
    _patch_ssh_flow(monkeypatch)
    state_path = tmp_path / "state.json"

    gpu_run.run_gpu(
        subset=None,
        db_path=tmp_path / "out.sqlite3",
        call_concurrency=1,
        gliner_concurrency=1,
        enable_llm_baseline=False,
        pod_state_path=state_path,
        ssh_key=None,
        keep_pod=False,
    )

    assert '"pod_id": "pod-456"' in state_path.read_text()


def test_run_gpu_resumes_pod_id_from_existing_state_file(monkeypatch, tmp_path):
    _patch_secrets(monkeypatch)
    _patch_hangar(monkeypatch)
    _patch_ssh_flow(monkeypatch)
    state_path = tmp_path / "state.json"
    state_path.write_text('{"pod_id": "pod-existing"}')

    gpu_run.run_gpu(
        subset=None,
        db_path=tmp_path / "out.sqlite3",
        call_concurrency=1,
        gliner_concurrency=1,
        enable_llm_baseline=False,
        pod_state_path=state_path,
        ssh_key=None,
        keep_pod=False,
    )

    spec = gpu_run.hangar.start_pod.call_args[0][0]
    assert spec.pod_id == "pod-existing"


def test_run_gpu_only_forwards_openai_key_when_llm_baseline_enabled(monkeypatch, tmp_path):
    _patch_secrets(monkeypatch)
    _patch_hangar(monkeypatch)
    _patch_ssh_flow(monkeypatch)

    gpu_run.run_gpu(
        subset=None,
        db_path=tmp_path / "out.sqlite3",
        call_concurrency=1,
        gliner_concurrency=1,
        enable_llm_baseline=True,
        pod_state_path=tmp_path / "state.json",
        ssh_key=None,
        keep_pod=False,
    )

    spec = gpu_run.hangar.start_pod.call_args[0][0]
    assert spec.extra_env["OPENAI_API_KEY"] == "openai-test-key"
    assert spec.extra_env["TYPESAFE_API_KEY"] == "typesafe-test-key"


def test_wait_for_pod_ready_polls_until_ssh_direct_available(monkeypatch):
    not_ready = {"status": "RUNNING", "runtime": None, "ssh": {"direct": None}}
    ready = {
        "status": "RUNNING",
        "runtime": {"uptime": 5},
        "ssh": {"direct": {"host": "1.2.3.4", "port": 2222, "username": "root"}},
    }
    get_pod_mock = Mock(side_effect=[not_ready, not_ready, ready])
    monkeypatch.setattr(gpu_run.hangar, "get_pod", get_pod_mock)
    monkeypatch.setattr(gpu_run.time, "sleep", Mock())

    result = gpu_run._wait_for_pod_ready("pod-123")

    assert result == {"host": "1.2.3.4", "port": 2222, "username": "root"}
    assert get_pod_mock.call_count == 3


def test_wait_for_pod_ready_raises_on_timeout(monkeypatch):
    not_ready = {"status": "RUNNING", "runtime": None, "ssh": {"direct": None}}
    monkeypatch.setattr(gpu_run.hangar, "get_pod", Mock(return_value=not_ready))
    monkeypatch.setattr(gpu_run.time, "sleep", Mock())
    # A zero timeout means the deadline is already past on the loop's first real-time check.
    monkeypatch.setattr(gpu_run, "_POD_READY_TIMEOUT_S", 0.0)

    with pytest.raises(TimeoutError):
        gpu_run._wait_for_pod_ready("pod-123")


def test_wait_for_pod_ready_raises_if_pod_disappears(monkeypatch):
    monkeypatch.setattr(gpu_run.hangar, "get_pod", Mock(return_value=None))
    monkeypatch.setattr(gpu_run.time, "sleep", Mock())

    with pytest.raises(RuntimeError, match="disappeared"):
        gpu_run._wait_for_pod_ready("pod-123")


def test_wait_for_ssh_connectable_retries_past_connection_refused(monkeypatch):
    ssh_direct = {"host": "1.2.3.4", "port": 2222, "username": "root"}
    refused = Mock(returncode=255, stdout="", stderr="ssh: connect to host 1.2.3.4 port 2222: Connection refused")
    ok = Mock(returncode=0, stdout="", stderr="")
    run_mock = Mock(side_effect=[refused, refused, ok])
    monkeypatch.setattr(gpu_run.subprocess, "run", run_mock)
    monkeypatch.setattr(gpu_run.time, "sleep", Mock())

    gpu_run._wait_for_ssh_connectable(ssh_direct, None)  # must not raise

    assert run_mock.call_count == 3


def test_wait_for_ssh_connectable_raises_on_timeout(monkeypatch):
    ssh_direct = {"host": "1.2.3.4", "port": 2222, "username": "root"}
    refused = Mock(returncode=255, stdout="", stderr="Connection refused")
    monkeypatch.setattr(gpu_run.subprocess, "run", Mock(return_value=refused))
    monkeypatch.setattr(gpu_run.time, "sleep", Mock())
    # Deterministic timeout: deadline = 100.0 + _SSH_CONNECT_TIMEOUT_S (default 120.0) = 220.0.
    # First while-check (100.0) passes, runs one attempt; second while-check (300.0) exceeds the
    # deadline and exits the loop. Controlling time.monotonic() directly avoids a real-clock race
    # that a near-zero timeout would have.
    monkeypatch.setattr(gpu_run.time, "monotonic", Mock(side_effect=[100.0, 100.0, 300.0]))

    with pytest.raises(TimeoutError, match="Connection refused"):
        gpu_run._wait_for_ssh_connectable(ssh_direct, None)
