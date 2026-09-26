import subprocess
from unittest.mock import patch

import pytest

from jev_live_transcription import secrets


def test_load_secret_noop_if_env_var_already_set(monkeypatch):
    monkeypatch.setenv("SOME_KEY", "already-set")
    with patch("subprocess.run") as mock_run:
        secrets.load_secret("SOME_KEY")
    mock_run.assert_not_called()


def test_load_secret_sets_env_var_from_strongbox_stdout(monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="\nsk-fake-value\n", stderr="")
    with patch("subprocess.run", return_value=fake_result):
        secrets.load_secret("SOME_KEY", secret_name="SOME_KEY")
    assert __import__("os").environ["SOME_KEY"] == "sk-fake-value"


def test_load_secret_takes_last_nonblank_line(monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    fake_result = subprocess.CompletedProcess(
        args=[], returncode=0, stdout="WARNING: some module noise\n\nsk-real-value\n", stderr=""
    )
    with patch("subprocess.run", return_value=fake_result):
        secrets.load_secret("SOME_KEY", secret_name="SOME_KEY")
    assert __import__("os").environ["SOME_KEY"] == "sk-real-value"


def test_load_secret_raises_on_empty_output(monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    fake_result = subprocess.CompletedProcess(args=[], returncode=0, stdout="", stderr="")
    with patch("subprocess.run", return_value=fake_result):
        with pytest.raises(RuntimeError):
            secrets.load_secret("SOME_KEY", secret_name="SOME_KEY")


def test_load_secret_raises_on_nonzero_exit(monkeypatch):
    monkeypatch.delenv("SOME_KEY", raising=False)
    fake_result = subprocess.CompletedProcess(args=[], returncode=1, stdout="", stderr="not found")
    with patch("subprocess.run", return_value=fake_result):
        with pytest.raises(RuntimeError):
            secrets.load_secret("SOME_KEY", secret_name="SOME_KEY")
