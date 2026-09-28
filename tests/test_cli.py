"""Mocked tests for cli.py -- no real batch run, no real secrets, no real corpus load.

Exercises the CLI argument-to-`batch_runner.run_batch`-call wiring, since a flag defined in
argparse but never forwarded (e.g. a silently-dead `--enable-llm-baseline`) would otherwise slip
through undetected.
"""

from unittest.mock import AsyncMock, Mock

from jev_live_transcription import batch_runner, cli


def _patch_common(monkeypatch, fake_calls=None):
    monkeypatch.setattr(cli.secrets, "load_openai_key", Mock())
    monkeypatch.setattr(cli.secrets, "load_typesafe_key", Mock())
    monkeypatch.setattr(cli.corpus, "load_all", lambda: fake_calls or {1: {}, 2: {}, 3: {}})


def test_batch_forwards_enable_llm_baseline_flag_when_passed(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    run_batch_mock = AsyncMock(return_value=batch_runner.BatchResult())
    monkeypatch.setattr(cli.batch_runner, "run_batch", run_batch_mock)

    db_path = tmp_path / "out.sqlite3"
    exit_code = cli.main(["batch", "--db-path", str(db_path), "--enable-llm-baseline"])

    assert exit_code == 0
    run_batch_mock.assert_awaited_once()
    _, kwargs = run_batch_mock.await_args
    assert kwargs["enable_llm_baseline"] is True


def test_batch_defaults_enable_llm_baseline_to_false(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    run_batch_mock = AsyncMock(return_value=batch_runner.BatchResult())
    monkeypatch.setattr(cli.batch_runner, "run_batch", run_batch_mock)

    db_path = tmp_path / "out.sqlite3"
    exit_code = cli.main(["batch", "--db-path", str(db_path)])

    assert exit_code == 0
    _, kwargs = run_batch_mock.await_args
    assert kwargs["enable_llm_baseline"] is False


def test_batch_defaults_gliner_concurrency_to_config_value_not_one(monkeypatch, tmp_path):
    # Regression test: --gliner-concurrency's argparse default must track config.GLINER_CONCURRENCY
    # rather than a hardcoded 1, or `jlt batch` with no flags silently overrides run_batch's own
    # config.GLINER_CONCURRENCY-based default back down to 1, defeating GlinerBatchEngine entirely.
    _patch_common(monkeypatch)
    run_batch_mock = AsyncMock(return_value=batch_runner.BatchResult())
    monkeypatch.setattr(cli.batch_runner, "run_batch", run_batch_mock)

    db_path = tmp_path / "out.sqlite3"
    cli.main(["batch", "--db-path", str(db_path)])

    _, kwargs = run_batch_mock.await_args
    assert kwargs["gliner_concurrency"] == cli.config.GLINER_CONCURRENCY


def test_batch_only_loads_openai_key_when_llm_baseline_enabled(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    monkeypatch.setattr(
        cli.batch_runner, "run_batch", AsyncMock(return_value=batch_runner.BatchResult())
    )
    openai_key_mock = Mock()
    monkeypatch.setattr(cli.secrets, "load_openai_key", openai_key_mock)

    db_path = tmp_path / "out.sqlite3"
    cli.main(["batch", "--db-path", str(db_path)])
    openai_key_mock.assert_not_called()

    cli.main(["batch", "--db-path", str(db_path), "--enable-llm-baseline"])
    openai_key_mock.assert_called_once()


def test_batch_forwards_subset_and_concurrency_flags(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    run_batch_mock = AsyncMock(return_value=batch_runner.BatchResult())
    monkeypatch.setattr(cli.batch_runner, "run_batch", run_batch_mock)

    db_path = tmp_path / "out.sqlite3"
    cli.main(
        [
            "batch",
            "--db-path",
            str(db_path),
            "--subset",
            "2",
            "--call-concurrency",
            "4",
            "--gliner-concurrency",
            "3",
        ]
    )

    args, kwargs = run_batch_mock.await_args
    call_ids = args[0]
    assert call_ids == [1, 2]  # first 2 call_ids of the 3 fake calls, sorted
    assert kwargs["call_concurrency"] == 4
    assert kwargs["gliner_concurrency"] == 3
    assert kwargs["db_path"] == db_path


def test_batch_reports_failures_via_exit_code(monkeypatch, tmp_path):
    _patch_common(monkeypatch)
    failed_result = batch_runner.BatchResult(succeeded=[1], failed=[(2, RuntimeError("boom"))])
    monkeypatch.setattr(cli.batch_runner, "run_batch", AsyncMock(return_value=failed_result))

    db_path = tmp_path / "out.sqlite3"
    exit_code = cli.main(["batch", "--db-path", str(db_path)])

    assert exit_code == 1


def test_gpu_run_forwards_flags(monkeypatch, tmp_path):
    # Unlike `tui` (patched via a fake sys.modules entry, since nothing else imports it and it's
    # heavy to import for real), `gpu_run` is already genuinely imported by test_gpu_run.py at
    # collection time -- `from . import gpu_run` inside `_run_gpu_run` resolves via the
    # `jev_live_transcription` package's own cached attribute, not sys.modules, so a sys.modules
    # fake would silently be ignored and the real hangar/RunPod calls would run instead. Patching
    # the real module's `run_gpu` attribute directly is what actually intercepts the call.
    from jev_live_transcription import gpu_run as real_gpu_run

    run_gpu_mock = Mock(return_value=0)
    monkeypatch.setattr(real_gpu_run, "run_gpu", run_gpu_mock)

    db_path = tmp_path / "out.sqlite3"
    state_path = tmp_path / "state.json"
    exit_code = cli.main(
        [
            "gpu-run",
            "--subset",
            "5",
            "--db-path",
            str(db_path),
            "--call-concurrency",
            "2",
            "--gliner-concurrency",
            "3",
            "--enable-llm-baseline",
            "--pod-state-path",
            str(state_path),
            "--ssh-key",
            "/home/me/.ssh/id_ed25519",
            "--keep-pod",
        ]
    )

    assert exit_code == 0
    run_gpu_mock.assert_called_once_with(
        subset=5,
        db_path=db_path,
        call_concurrency=2,
        gliner_concurrency=3,
        enable_llm_baseline=True,
        pod_state_path=state_path,
        ssh_key="/home/me/.ssh/id_ed25519",
        keep_pod=True,
    )


def test_gpu_run_defaults(monkeypatch, tmp_path):
    from jev_live_transcription import gpu_run as real_gpu_run

    run_gpu_mock = Mock(return_value=0)
    monkeypatch.setattr(real_gpu_run, "run_gpu", run_gpu_mock)

    cli.main(["gpu-run", "--ssh-key", "/path/to/key"])

    _, kwargs = run_gpu_mock.call_args
    assert kwargs["subset"] is None
    assert kwargs["call_concurrency"] == 1
    assert kwargs["gliner_concurrency"] == cli.config.GLINER_CONCURRENCY
    assert kwargs["enable_llm_baseline"] is False
    assert kwargs["ssh_key"] == "/path/to/key"
    assert kwargs["keep_pod"] is False


def test_tui_defaults_enable_llm_baseline_to_false(monkeypatch):
    # `_run_tui` imports `tui` lazily inside the function -- patch the module in sys.modules so
    # that import resolves to a fake instead of the real (model-loading) tui module.
    import sys
    from types import ModuleType

    tui_run_mock = Mock()
    fake_tui = ModuleType("jev_live_transcription.tui")
    fake_tui.run = tui_run_mock
    monkeypatch.setitem(sys.modules, "jev_live_transcription.tui", fake_tui)

    cli.main(["tui", "42"])

    tui_run_mock.assert_called_once_with(42, db_path=None, enable_llm_baseline=False)


def test_tui_forwards_enable_llm_baseline_when_passed(monkeypatch):
    import sys
    from types import ModuleType

    tui_run_mock = Mock()
    fake_tui = ModuleType("jev_live_transcription.tui")
    fake_tui.run = tui_run_mock
    monkeypatch.setitem(sys.modules, "jev_live_transcription.tui", fake_tui)

    cli.main(["tui", "42", "--enable-llm-baseline"])

    tui_run_mock.assert_called_once_with(42, db_path=None, enable_llm_baseline=True)


def test_serve_runs_uvicorn_against_a_freshly_created_app(monkeypatch):
    import sys

    from jev_live_transcription.serving import app as real_serving_app

    fake_app = object()
    create_app_mock = Mock(return_value=fake_app)
    monkeypatch.setattr(real_serving_app, "create_app", create_app_mock)
    uvicorn_run_mock = Mock()
    monkeypatch.setitem(sys.modules, "uvicorn", Mock(run=uvicorn_run_mock))

    exit_code = cli.main(["serve", "--host", "127.0.0.1", "--port", "9000"])

    assert exit_code == 0
    create_app_mock.assert_called_once_with(max_batch_size=None, batch_wait_timeout_ms=None)
    uvicorn_run_mock.assert_called_once_with(fake_app, host="127.0.0.1", port=9000)


def test_serve_defaults_host_and_port(monkeypatch):
    import sys

    uvicorn_run_mock = Mock()
    monkeypatch.setitem(sys.modules, "uvicorn", Mock(run=uvicorn_run_mock))

    cli.main(["serve"])

    _, kwargs = uvicorn_run_mock.call_args
    assert kwargs["host"] == "0.0.0.0"
    assert kwargs["port"] == 8000


def test_serve_forwards_batch_tuning_overrides_to_create_app(monkeypatch):
    import sys

    from jev_live_transcription.serving import app as real_serving_app

    create_app_mock = Mock(return_value=object())
    monkeypatch.setattr(real_serving_app, "create_app", create_app_mock)
    monkeypatch.setitem(sys.modules, "uvicorn", Mock(run=Mock()))

    cli.main(["serve", "--max-batch-size", "32", "--batch-wait-timeout-ms", "5"])

    create_app_mock.assert_called_once_with(max_batch_size=32, batch_wait_timeout_ms=5.0)


def test_serve_rejects_non_positive_max_batch_size(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "uvicorn", Mock(run=Mock()))

    exit_code = cli.main(["serve", "--max-batch-size", "0"])

    assert exit_code == 2


def test_serve_rejects_negative_batch_wait_timeout(monkeypatch):
    import sys

    monkeypatch.setitem(sys.modules, "uvicorn", Mock(run=Mock()))

    exit_code = cli.main(["serve", "--batch-wait-timeout-ms", "-1"])

    assert exit_code == 2
