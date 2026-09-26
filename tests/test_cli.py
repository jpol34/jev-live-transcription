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
