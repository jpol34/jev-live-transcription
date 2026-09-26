"""Strongbox-backed secret loading.

Sources API keys from the Strongbox PowerShell module rather than plaintext
.env files, hardcoded values, or asking for a secret to be pasted in. Never
prints or logs a secret value.
"""

import os
import subprocess

OPENAI_ENV_VAR = "OPENAI_API_KEY"
TYPESAFE_ENV_VAR = "TYPESAFE_API_KEY"
RUNPOD_ENV_VAR = "RUNPOD_API_KEY"


def load_secret(env_var: str, secret_name: str | None = None) -> None:
    """Load one secret from Strongbox into the process environment.

    No-op if `env_var` is already set. `secret_name` defaults to `env_var`
    when the Strongbox entry uses the same name.
    """
    if os.environ.get(env_var):
        return
    name = secret_name or env_var
    # Escape for PowerShell's single-quoted string literal: a literal quote
    # is written as two quotes in a row.
    escaped_name = name.replace("'", "''")
    result = subprocess.run(
        [
            "pwsh", "-NoProfile", "-Command",
            "$WarningPreference = 'SilentlyContinue'; "
            "Import-Module Strongbox -WarningAction SilentlyContinue; "
            f"Get-Secret -Name '{escaped_name}' -Vault Strongbox -AsPlainText",
        ],
        capture_output=True,
        text=True,
    )
    # Defensive: only take the last non-blank line, in case any module output
    # still leaks onto stdout ahead of the secret.
    lines = [ln for ln in result.stdout.splitlines() if ln.strip()]
    value = lines[-1].strip() if lines else ""
    if result.returncode != 0 or not value:
        raise RuntimeError(
            f"Could not retrieve {name} from Strongbox: "
            f"rc={result.returncode} stderr={result.stderr.strip()[:200]}"
        )
    os.environ[env_var] = value


def load_openai_key() -> None:
    load_secret(OPENAI_ENV_VAR)


def load_runpod_key() -> None:
    load_secret(RUNPOD_ENV_VAR)


def load_typesafe_key() -> None:
    # typesafe.ai keys are provisioned per-project on their side, so this pulls the
    # project-scoped Strongbox entry rather than a generic one (same convention as trellis's and
    # laya-bench's scripts/configure-typesafe-secret.ps1).
    load_secret(TYPESAFE_ENV_VAR, secret_name="TYPESAFE_API_KEY::jpol34/jev-live-transcription")
