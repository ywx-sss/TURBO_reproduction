# Download TabMWP from OpenXLab (credentials in data.env)

$ErrorActionPreference = "Stop"
$Root = Split-Path $PSScriptRoot -Parent
$Target = Join-Path $Root "data\TabMWP"
New-Item -ItemType Directory -Force -Path $Target | Out-Null

foreach ($name in @("data.env", ".env.local")) {
    $EnvFile = Join-Path $Root $name
    if (-not (Test-Path $EnvFile)) { continue }
    Write-Host "Loading credentials from $name"
    Get-Content $EnvFile | ForEach-Object {
        if ($_ -match '^\s*([^#][^=]+)=(.*)$') {
            $k = $matches[1].Trim()
            $val = $matches[2].Trim().Trim('"').Trim("'")
            Set-Item -Path "env:$k" -Value $val
        }
    }
    break
}

if (-not $env:OPENXLAB_AK -or -not $env:OPENXLAB_SK) {
    Write-Host "[ERROR] Missing OPENXLAB_AK or OPENXLAB_SK in data.env"
    exit 1
}

$Ox = "$env:APPDATA\Python\Python314\Scripts\openxlab.exe"
if (-not (Test-Path $Ox)) {
    $cmd = Get-Command openxlab -ErrorAction SilentlyContinue
    if ($cmd) { $Ox = $cmd.Source } else {
        Write-Host "[ERROR] openxlab not found. Run: pip install -U openxlab"
        exit 1
    }
}

Write-Host "[1/3] Login OpenXLab..."
python -c @"
from openxlab.xlab.handler.user_login import login
login('$($env:OPENXLAB_AK)', '$($env:OPENXLAB_SK)')
print('Login OK')
"@

Write-Host "[2/3] Dataset info..."
& $Ox dataset info --dataset-repo OpenDataLab/TabMWP

Write-Host "[3/3] Download -> $Target"
& $Ox dataset get --dataset-repo OpenDataLab/TabMWP --target-path $Target

Write-Host "Done. Unzip tables.zip under raw/ if needed."
