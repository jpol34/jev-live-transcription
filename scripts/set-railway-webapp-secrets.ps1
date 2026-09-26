<#
.SYNOPSIS
    Sets TYPESAFE_API_KEY on the jev-live-transcription webapp's Railway service, sourced from
    Strongbox -- run this yourself; the secret value never passes through Claude.

.DESCRIPTION
    No OPENAI_API_KEY is set here: the webapp never runs the GPT-5.1 baseline arm
    (pipeline_core.run_call's enable_llm_baseline defaults to False and the webapp never opts in),
    so it has no use for that key.

    Requires the Strongbox PowerShell module and the railway CLI. Run `railway link --project
    94d4b2ae-f830-4250-b596-1569c141df54` first (project "jev-live-transcription-webapp") if this
    shell isn't already linked to it.

.PARAMETER ServiceName
    The Railway service name to set the variable on. Defaults to "webapp", matching the service
    created in the jev-live-transcription-webapp project.
#>
param(
    [string]$ServiceName = "webapp"
)

$ErrorActionPreference = "Stop"

Import-Module Strongbox -WarningAction SilentlyContinue
$typesafeKey = Get-Secret -Name "TYPESAFE_API_KEY::jpol34/jev-live-transcription" -Vault Strongbox -AsPlainText
if (-not $typesafeKey) {
    throw "Could not retrieve TYPESAFE_API_KEY from Strongbox."
}

$typesafeKey | railway variable set TYPESAFE_API_KEY --stdin --service $ServiceName

Write-Host "Set TYPESAFE_API_KEY on Railway service '$ServiceName'."
