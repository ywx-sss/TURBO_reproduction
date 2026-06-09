# SFT eval launcher — always uses project venv + hf-mirror
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

$env:HF_ENDPOINT = "https://hf-mirror.com"
$env:PYTHONUNBUFFERED = "1"

$Python = Join-Path (Get-Location) ".venv-step05\Scripts\python.exe"
if (-not (Test-Path $Python)) {
    Write-Host "[ERROR] Missing $Python — run .\scripts\install_vl_env.ps1 first" -ForegroundColor Red
    exit 1
}

Write-Host "Using: $Python"
& $Python --version
Write-Host ""

& $Python -u scripts\eval_sft.py @args
exit $LASTEXITCODE
