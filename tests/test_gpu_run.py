"""Mocked tests for gpu_run.py -- no real RunPod pod, no real SSH, no real secrets.

`hangar` is mocked at the module level (the object `gpu_run` imported), and `subprocess.run` is
mocked for the SSH/scp-shelling-out functions. Focus is teardown-on-failure (the pod must always
be terminated, success or failure, unless `--keep-pod`) and the pod-readiness/completion polling
loops, since those are the parts a real pod can't be used to test cheaply.
"""

from pathlib import Path
from unittest.mock import Mock

import pytest

from jev_live_transcription import gpu_run


def _fake_ssh_key(tmp_path) -> str:
    """Writes a fake `<key>.pub` file and returns the private-key path `run_gpu` expects -- it
    reads the public key's contents to inject as the pod's `PUBLIC_KEY` env var."""
    key_path = tmp_path / "test_key"
    (tmp_path / "test_key.pub").write_text("ssh-ed25519 AAAAtestkey test@example.com\n")
    return str(key_path)


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


_FAKE_CALL_COUNT = 7


def _patch_ssh_flow(monkeypatch, *, exit_code=0, remote_call_count=_FAKE_CALL_COUNT):
    """Makes `_run_ssh` succeed for preflight/launch and immediately report completion.

    On the `scp` step, writes a `calls` table sized to `remote_call_count`, and the remote-log
    `cat` (distinct from the exit-marker `cat`) reports the same count via a fake
    `"Running N call(s)"` line, matching what `_read_remote_expected_calls` parses from a real
    `jlt batch` log -- `_retrieve_db` verifies the scp'd row count against that on a successful
    (`exit_code=0`) run, so the two must agree for a test to represent a correct run.
    """

    def fake_run(cmd, **kwargs):
        joined = " ".join(cmd)
        if cmd[0] == "scp":
            dest = Path(cmd[-1])
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.unlink(missing_ok=True)
            conn = gpu_run.db_module.connect(dest)
            try:
                conn.execute("CREATE TABLE calls (call_id INTEGER PRIMARY KEY)")
                conn.executemany(
                    "INSERT INTO calls (call_id) VALUES (?)", [(i,) for i in range(remote_call_count)]
                )
                conn.commit()
            finally:
                conn.close()
            return Mock(returncode=0, stdout="", stderr="")
        if joined.endswith(" true"):
            return Mock(returncode=0, stdout="", stderr="")
        if "nvidia-smi" in joined:
            return Mock(returncode=0, stdout="NVIDIA L40S, 49140 MiB", stderr="")
        if gpu_run._REMOTE_ENV_PATH in joined and "umask" in joined:
            return Mock(returncode=0, stdout="", stderr="")
        if "nohup" in joined:
            return Mock(returncode=0, stdout="LAUNCHED\n", stderr="")
        if gpu_run._REMOTE_LOG_PATH in joined:
            return Mock(
                returncode=0,
                stdout=f"Running {remote_call_count} call(s) -> /root/benchmark.sqlite3 (call_concurrency=1)\n",
                stderr="",
            )
        if gpu_run._REMOTE_EXIT_MARKER in joined:
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
        ssh_key=_fake_ssh_key(tmp_path),
        keep_pod=False,
    )

    assert result == 0
    gpu_run.hangar.pod_action.assert_called_once_with("pod-123", "terminate")


def test_run_gpu_subset_larger_than_corpus_does_not_fail_verification(monkeypatch, tmp_path):
    # cli.py's own `_run_batch` slices `call_ids[:subset]`, which silently caps to the corpus size
    # rather than erroring -- a `--subset` larger than the corpus (e.g. "run everything" expressed
    # as a large number) completes correctly with fewer calls than `subset` asked for. Verification
    # is unaffected either way, since `expected_calls` comes from the remote run's own reported
    # count (`_read_remote_expected_calls`), never from `subset` or a local corpus computation.
    _patch_secrets(monkeypatch)
    _patch_hangar(monkeypatch)
    _patch_ssh_flow(monkeypatch, exit_code=0)

    result = gpu_run.run_gpu(
        subset=1_000_000,
        db_path=tmp_path / "out.sqlite3",
        call_concurrency=1,
        gliner_concurrency=1,
        enable_llm_baseline=False,
        pod_state_path=tmp_path / "state.json",
        ssh_key=_fake_ssh_key(tmp_path),
        keep_pod=False,
    )

    assert result == 0


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
            ssh_key=_fake_ssh_key(tmp_path),
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
        ssh_key=_fake_ssh_key(tmp_path),
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
        ssh_key=_fake_ssh_key(tmp_path),
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
        ssh_key=_fake_ssh_key(tmp_path),
        keep_pod=False,
    )

    spec = gpu_run.hangar.start_pod.call_args[0][0]
    assert spec.pod_id == "pod-existing"


def test_run_gpu_skips_typesafe_key_when_jev_disabled(monkeypatch, tmp_path):
    # --disable-jev must mean TYPESAFE_API_KEY is never loaded or validated on the GPU-run path
    # either, or a fully local --enable-gliner-only --disable-jev run can't start without a
    # typesafe.ai credential.
    _patch_secrets(monkeypatch)
    _patch_hangar(monkeypatch)
    _patch_ssh_flow(monkeypatch)
    load_typesafe_key_mock = Mock()
    monkeypatch.setattr(gpu_run.secrets, "load_typesafe_key", load_typesafe_key_mock)

    result = gpu_run.run_gpu(
        subset=None,
        db_path=tmp_path / "out.sqlite3",
        call_concurrency=1,
        gliner_concurrency=1,
        enable_llm_baseline=False,
        enable_gliner_only=True,
        enable_jev=False,
        pod_state_path=tmp_path / "state.json",
        ssh_key=_fake_ssh_key(tmp_path),
        keep_pod=False,
    )

    assert result == 0
    load_typesafe_key_mock.assert_not_called()
    spec = gpu_run.hangar.start_pod.call_args[0][0]
    assert "TYPESAFE_API_KEY" not in spec.extra_env


def test_run_gpu_loads_typesafe_key_when_jev_enabled(monkeypatch, tmp_path):
    _patch_secrets(monkeypatch)
    _patch_hangar(monkeypatch)
    _patch_ssh_flow(monkeypatch)
    load_typesafe_key_mock = Mock()
    monkeypatch.setattr(gpu_run.secrets, "load_typesafe_key", load_typesafe_key_mock)

    gpu_run.run_gpu(
        subset=None,
        db_path=tmp_path / "out.sqlite3",
        call_concurrency=1,
        gliner_concurrency=1,
        enable_llm_baseline=False,
        pod_state_path=tmp_path / "state.json",
        ssh_key=_fake_ssh_key(tmp_path),
        keep_pod=False,
    )

    load_typesafe_key_mock.assert_called_once()
    spec = gpu_run.hangar.start_pod.call_args[0][0]
    assert spec.extra_env["TYPESAFE_API_KEY"] == "typesafe-test-key"


def test_run_gpu_forwards_enable_gliner_only_and_disable_jev_to_remote_batch_command(monkeypatch, tmp_path):
    _patch_secrets(monkeypatch)
    _patch_hangar(monkeypatch)
    _patch_ssh_flow(monkeypatch)
    launch_batch_detached_mock = Mock(wraps=gpu_run._launch_batch_detached)
    monkeypatch.setattr(gpu_run, "_launch_batch_detached", launch_batch_detached_mock)

    gpu_run.run_gpu(
        subset=None,
        db_path=tmp_path / "out.sqlite3",
        call_concurrency=1,
        gliner_concurrency=1,
        enable_llm_baseline=False,
        enable_gliner_only=True,
        enable_jev=False,
        pod_state_path=tmp_path / "state.json",
        ssh_key=_fake_ssh_key(tmp_path),
        keep_pod=False,
    )

    _, kwargs = launch_batch_detached_mock.call_args
    assert kwargs["enable_gliner_only"] is True
    assert kwargs["enable_jev"] is False


def test_launch_batch_detached_includes_enable_gliner_only_flag(monkeypatch):
    ssh_direct = {"host": "1.2.3.4", "port": 2222, "username": "root"}
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = " ".join(cmd)
        return Mock(returncode=0, stdout="LAUNCHED\n", stderr="")

    monkeypatch.setattr(gpu_run.subprocess, "run", Mock(side_effect=fake_run))

    gpu_run._launch_batch_detached(
        ssh_direct,
        None,
        subset=None,
        call_concurrency=1,
        gliner_concurrency=1,
        enable_llm_baseline=False,
        enable_gliner_only=True,
        enable_jev=True,
    )

    assert "--enable-gliner-only" in captured["cmd"]
    assert "--disable-jev" not in captured["cmd"]


def test_launch_batch_detached_includes_disable_jev_flag_when_jev_disabled(monkeypatch):
    ssh_direct = {"host": "1.2.3.4", "port": 2222, "username": "root"}
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = " ".join(cmd)
        return Mock(returncode=0, stdout="LAUNCHED\n", stderr="")

    monkeypatch.setattr(gpu_run.subprocess, "run", Mock(side_effect=fake_run))

    gpu_run._launch_batch_detached(
        ssh_direct,
        None,
        subset=None,
        call_concurrency=1,
        gliner_concurrency=1,
        enable_llm_baseline=False,
        enable_gliner_only=False,
        enable_jev=False,
    )

    assert "--disable-jev" in captured["cmd"]
    assert "--enable-gliner-only" not in captured["cmd"]


def test_launch_batch_detached_omits_both_flags_by_default(monkeypatch):
    ssh_direct = {"host": "1.2.3.4", "port": 2222, "username": "root"}
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = " ".join(cmd)
        return Mock(returncode=0, stdout="LAUNCHED\n", stderr="")

    monkeypatch.setattr(gpu_run.subprocess, "run", Mock(side_effect=fake_run))

    gpu_run._launch_batch_detached(
        ssh_direct,
        None,
        subset=None,
        call_concurrency=1,
        gliner_concurrency=1,
        enable_llm_baseline=False,
    )

    assert "--enable-gliner-only" not in captured["cmd"]
    assert "--disable-jev" not in captured["cmd"]


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
        ssh_key=_fake_ssh_key(tmp_path),
        keep_pod=False,
    )

    spec = gpu_run.hangar.start_pod.call_args[0][0]
    assert spec.extra_env["OPENAI_API_KEY"] == "openai-test-key"
    assert spec.extra_env["TYPESAFE_API_KEY"] == "typesafe-test-key"


def test_run_gpu_injects_public_key_from_ssh_key_pub_file(monkeypatch, tmp_path):
    _patch_secrets(monkeypatch)
    _patch_hangar(monkeypatch)
    _patch_ssh_flow(monkeypatch)

    gpu_run.run_gpu(
        subset=None,
        db_path=tmp_path / "out.sqlite3",
        call_concurrency=1,
        gliner_concurrency=1,
        enable_llm_baseline=False,
        pod_state_path=tmp_path / "state.json",
        ssh_key=_fake_ssh_key(tmp_path),
        keep_pod=False,
    )

    spec = gpu_run.hangar.start_pod.call_args[0][0]
    assert spec.extra_env["PUBLIC_KEY"] == "ssh-ed25519 AAAAtestkey test@example.com"


def test_run_gpu_excludes_public_key_from_remote_env_file(monkeypatch, tmp_path):
    _patch_secrets(monkeypatch)
    _patch_hangar(monkeypatch)

    write_remote_env_mock = Mock()
    monkeypatch.setattr(gpu_run, "_write_remote_env", write_remote_env_mock)
    _patch_ssh_flow(monkeypatch)

    gpu_run.run_gpu(
        subset=None,
        db_path=tmp_path / "out.sqlite3",
        call_concurrency=1,
        gliner_concurrency=1,
        enable_llm_baseline=False,
        pod_state_path=tmp_path / "state.json",
        ssh_key=_fake_ssh_key(tmp_path),
        keep_pod=False,
    )

    _, _, written_env = write_remote_env_mock.call_args[0]
    assert "PUBLIC_KEY" not in written_env
    assert written_env["TYPESAFE_API_KEY"] == "typesafe-test-key"


def test_run_gpu_raises_clear_error_without_ssh_key(monkeypatch, tmp_path):
    _patch_secrets(monkeypatch)
    _patch_hangar(monkeypatch)

    with pytest.raises(ValueError, match="--ssh-key is required"):
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

    gpu_run.hangar.start_pod.assert_not_called()


def test_run_gpu_raises_clear_error_when_pub_file_missing(monkeypatch, tmp_path):
    _patch_secrets(monkeypatch)
    _patch_hangar(monkeypatch)

    with pytest.raises(FileNotFoundError, match="no public key found"):
        gpu_run.run_gpu(
            subset=None,
            db_path=tmp_path / "out.sqlite3",
            call_concurrency=1,
            gliner_concurrency=1,
            enable_llm_baseline=False,
            pod_state_path=tmp_path / "state.json",
            ssh_key=str(tmp_path / "nonexistent_key"),
            keep_pod=False,
        )

    gpu_run.hangar.start_pod.assert_not_called()


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
    monkeypatch.setattr(gpu_run, "_SSH_CONNECT_TIMEOUT_S", 120.0)
    # Deterministic timeout: deadline = 100.0 + 120.0 = 220.0. First while-check (100.0) passes,
    # runs one attempt; second while-check (300.0) exceeds the deadline and exits the loop.
    # Controlling time.monotonic() directly avoids a real-clock race a near-zero timeout would have.
    monkeypatch.setattr(gpu_run.time, "monotonic", Mock(side_effect=[100.0, 100.0, 300.0]))

    with pytest.raises(TimeoutError, match="Connection refused"):
        gpu_run._wait_for_ssh_connectable(ssh_direct, None)


def test_write_remote_env_sends_secrets_via_stdin_not_argv(monkeypatch):
    """A secret value must never appear as a subprocess argument (visible in the pod's own `ps`
    output) -- it has to travel as piped stdin content instead."""
    ssh_direct = {"host": "1.2.3.4", "port": 2222, "username": "root"}
    captured = {}

    def fake_run(cmd, **kwargs):
        captured["cmd"] = cmd
        captured["input"] = kwargs.get("input")
        return Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gpu_run.subprocess, "run", Mock(side_effect=fake_run))

    gpu_run._write_remote_env(ssh_direct, None, {"TYPESAFE_API_KEY": "super-secret-value"})

    assert not any("super-secret-value" in part for part in captured["cmd"])
    assert "export TYPESAFE_API_KEY=super-secret-value" in captured["input"]
    assert gpu_run._REMOTE_ENV_PATH in " ".join(captured["cmd"])
    assert "umask 077" in " ".join(captured["cmd"])


def test_write_remote_env_raises_on_failure(monkeypatch):
    ssh_direct = {"host": "1.2.3.4", "port": 2222, "username": "root"}
    monkeypatch.setattr(
        gpu_run.subprocess, "run", Mock(return_value=Mock(returncode=1, stdout="", stderr="disk full"))
    )

    with pytest.raises(RuntimeError, match="failed to write remote env file"):
        gpu_run._write_remote_env(ssh_direct, None, {"FOO": "bar"})


# --- remote-reported expected call count --------------------------------------------------------


def test_read_remote_expected_calls_parses_the_running_line(monkeypatch):
    ssh_direct = {"host": "1.2.3.4", "port": 2222, "username": "root"}
    log = (
        "some preflight output\n"
        "Running 42 call(s) -> /root/benchmark.sqlite3 (call_concurrency=1, gliner_concurrency=1, "
        "enable_llm_baseline=False)\n"
        "more output after\n"
    )
    monkeypatch.setattr(
        gpu_run.subprocess, "run", Mock(return_value=Mock(returncode=0, stdout=log, stderr=""))
    )

    assert gpu_run._read_remote_expected_calls(ssh_direct, None) == 42


def test_read_remote_expected_calls_raises_when_running_line_is_absent(monkeypatch):
    ssh_direct = {"host": "1.2.3.4", "port": 2222, "username": "root"}
    monkeypatch.setattr(
        gpu_run.subprocess, "run", Mock(return_value=Mock(returncode=0, stdout="no such line here\n", stderr=""))
    )

    with pytest.raises(RuntimeError, match="could not find jlt batch's own"):
        gpu_run._read_remote_expected_calls(ssh_direct, None)


def test_read_remote_expected_calls_raises_ssh_error_not_missing_line_error(monkeypatch):
    # An SSH-level failure (dropped connection, permission denied, ...) must surface as that
    # failure, not be misread as "the log doesn't have the line yet" -- those are different
    # problems with different fixes, and conflating them sends whoever's debugging down the wrong
    # path entirely.
    ssh_direct = {"host": "1.2.3.4", "port": 2222, "username": "root"}
    monkeypatch.setattr(
        gpu_run.subprocess,
        "run",
        Mock(return_value=Mock(returncode=255, stdout="", stderr="ssh: connect to host: Connection refused")),
    )

    with pytest.raises(RuntimeError, match="failed to read remote log.*Connection refused"):
        gpu_run._read_remote_expected_calls(ssh_direct, None)


# --- retrieved-DB verification ----------------------------------------------------------------


def _write_db_with_calls(path, n_calls):
    # scp overwrites the destination wholesale, so this helper does too -- a stale file left at
    # `path` from a prior "attempt" would otherwise make sqlite3 error opening it, rather than
    # exercising the retry path each test actually means to simulate.
    path.unlink(missing_ok=True)
    conn = gpu_run.db_module.connect(path)
    try:
        conn.execute("CREATE TABLE calls (call_id INTEGER PRIMARY KEY)")
        conn.executemany("INSERT INTO calls (call_id) VALUES (?)", [(i,) for i in range(n_calls)])
        conn.commit()
    finally:
        conn.close()


def _write_corrupt_db(path):
    # Not a real SQLite file at all -- the cheapest reliable way to make `integrity_check` fail
    # without depending on reproducing a specific on-disk corruption pattern.
    path.unlink(missing_ok=True)
    path.write_bytes(b"not a sqlite database")


def test_verify_db_none_for_complete_matching_db(tmp_path):
    db_path = tmp_path / "valid.sqlite3"
    _write_db_with_calls(db_path, 5)

    assert gpu_run._verify_db(db_path, expected_calls=5) is None


def test_verify_db_reports_integrity_failure_for_corrupt_file(tmp_path):
    db_path = tmp_path / "corrupt.sqlite3"
    _write_corrupt_db(db_path)

    problem = gpu_run._verify_db(db_path, expected_calls=5)

    assert problem is not None and "not a readable" in problem


def test_verify_db_reports_call_count_mismatch_for_empty_db(tmp_path):
    # Structurally valid (passes integrity_check trivially) but empty -- the scenario an
    # integrity-check-only verification would miss.
    db_path = tmp_path / "empty.sqlite3"
    _write_db_with_calls(db_path, 0)

    problem = gpu_run._verify_db(db_path, expected_calls=5)

    assert problem is not None and "0 calls, expected 5" in problem


def test_verify_db_reports_call_count_mismatch_for_partial_db(tmp_path):
    db_path = tmp_path / "partial.sqlite3"
    _write_db_with_calls(db_path, 3)

    problem = gpu_run._verify_db(db_path, expected_calls=5)

    assert problem is not None and "3 calls, expected 5" in problem


def test_retrieve_db_succeeds_when_scp_output_verifies_clean(monkeypatch, tmp_path):
    ssh_direct = {"host": "1.2.3.4", "port": 2222, "username": "root"}
    local_db_path = tmp_path / "out.sqlite3"

    def fake_scp(cmd, **kwargs):
        _write_db_with_calls(local_db_path, 5)
        return Mock(returncode=0, stdout="", stderr="")

    run_mock = Mock(side_effect=fake_scp)
    monkeypatch.setattr(gpu_run.subprocess, "run", run_mock)

    gpu_run._retrieve_db(ssh_direct, None, local_db_path, expected_calls=5)

    assert run_mock.call_count == 1


def test_retrieve_db_retries_once_then_succeeds_after_a_bad_first_copy(monkeypatch, tmp_path):
    ssh_direct = {"host": "1.2.3.4", "port": 2222, "username": "root"}
    local_db_path = tmp_path / "out.sqlite3"
    attempts = []

    def fake_scp(cmd, **kwargs):
        attempts.append(1)
        if len(attempts) == 1:
            _write_corrupt_db(local_db_path)
        else:
            _write_db_with_calls(local_db_path, 5)
        return Mock(returncode=0, stdout="", stderr="")

    monkeypatch.setattr(gpu_run.subprocess, "run", Mock(side_effect=fake_scp))

    gpu_run._retrieve_db(ssh_direct, None, local_db_path, expected_calls=5)

    assert len(attempts) == 2
    assert gpu_run._verify_db(local_db_path, expected_calls=5) is None


def test_retrieve_db_raises_after_two_consecutive_verification_failures(monkeypatch, tmp_path):
    ssh_direct = {"host": "1.2.3.4", "port": 2222, "username": "root"}
    local_db_path = tmp_path / "out.sqlite3"

    def fake_scp(cmd, **kwargs):
        _write_corrupt_db(local_db_path)
        return Mock(returncode=0, stdout="", stderr="")

    run_mock = Mock(side_effect=fake_scp)
    monkeypatch.setattr(gpu_run.subprocess, "run", run_mock)

    with pytest.raises(RuntimeError, match="not a readable SQLite database.*twice"):
        gpu_run._retrieve_db(ssh_direct, None, local_db_path, expected_calls=5)

    assert run_mock.call_count == 2


def test_retrieve_db_still_raises_on_scp_failure_before_any_verification(monkeypatch, tmp_path):
    ssh_direct = {"host": "1.2.3.4", "port": 2222, "username": "root"}
    local_db_path = tmp_path / "out.sqlite3"
    monkeypatch.setattr(
        gpu_run.subprocess, "run", Mock(return_value=Mock(returncode=1, stdout="", stderr="connection lost"))
    )

    with pytest.raises(RuntimeError, match="failed to retrieve capture DB"):
        gpu_run._retrieve_db(ssh_direct, None, local_db_path, expected_calls=5)


def test_retrieve_db_skips_verification_when_expected_calls_is_none(monkeypatch, tmp_path):
    # expected_calls=None signals the remote jlt batch itself already failed -- retrieval is
    # best-effort, for debugging, on a single attempt, with no row-count or integrity bar to clear.
    ssh_direct = {"host": "1.2.3.4", "port": 2222, "username": "root"}
    local_db_path = tmp_path / "out.sqlite3"

    def fake_scp(cmd, **kwargs):
        _write_corrupt_db(local_db_path)
        return Mock(returncode=0, stdout="", stderr="")

    run_mock = Mock(side_effect=fake_scp)
    monkeypatch.setattr(gpu_run.subprocess, "run", run_mock)

    gpu_run._retrieve_db(ssh_direct, None, local_db_path, expected_calls=None)

    assert run_mock.call_count == 1
    assert local_db_path.exists()
