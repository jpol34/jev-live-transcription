<#
.SYNOPSIS
    Sets TYPESAFE_API_KEY on the jev-live-transcription webapp's Railway service, sourced from
    Strongbox -- run this yourself; the secret value never passes through Claude.

.DESCRIPTION
    No OPENAI_API_KEY is set here: the webapp never runs the GPT-5.1 baseline arm
    (pipeline_core.run_call's enable_llm_baseline defaults to False and the webapp never opts in),
    so it has no use for that key.

    Requires the Strongbox PowerShell module and the railway CLI, both already linked to this
    project (`railway link`) before running.

.PARAMETER ServiceName
    The Railway service name to set the variable on. Defaults to "jev-live-transcription-webapp"
    -- adjust if the actual service was created under a different name.
#>
param(
    [string]$ServiceName = "jev-live-transcription-webapp"
)

$ErrorActionPreference = "Stop"

Import-Module Strongbox -WarningAction SilentlyContinue
$typesafeKey = Get-Secret -Name "TYPESAFE_API_KEY::jpol34/jev-live-transcription" -Vault Strongbox -AsPlainText
if (-not $typesafeKey) {
    throw "Could not retrieve TYPESAFE_API_KEY from Strongbox."
}

railway variables --service $ServiceName --set "TYPESAFE_API_KEY=$typesafeKey"

Write-Host "Set TYPESAFE_API_KEY on Railway service '$ServiceName'."
