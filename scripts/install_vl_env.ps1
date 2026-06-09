# Install PyTorch (CUDA) + Qwen3-VL inference dependencies
# Requires Python 3.10–3.12

$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot\..

function Resolve-Python312 {
    if (Get-Command py -ErrorAction SilentlyContinue) {
        try {
            $exe = & py -3.12 -c "import sys; print(sys.executable)" 2>$null
            if ($exe -and (Test-Path $exe.Trim())) { return $exe.Trim() }
        } catch {}
    }
    $candidates = @(
        "$env:LOCALAPPDATA\Programs\Python\Python312\python.exe",
        "$env:ProgramFiles\Python312\python.exe"
    )
    foreach ($p in $candidates) {
        if (Test-Path $p) {
            $ver = & $p -c "import sys; print(f'{sys.version_info.major}.{sys.version_info.minor}')" 2>$null
            if ($ver -eq "3.12") { return $p }
        }
    }
    throw "Python 3.12 not found. Install: winget install Python.Python.3.12"
}

function Test-VenvPip($PythonPath) {
    $null = & $PythonPath -m pip --version 2>&1
    return $LASTEXITCODE -eq 0
}

function Ensure-Venv($BasePython, [string]$VenvDir) {
    $venvPython = Join-Path (Join-Path (Get-Location) $VenvDir) "Scripts\python.exe"
    if (-not (Test-Path $venvPython)) {
        Write-Host "Creating $VenvDir ..."
        & $BasePython -m venv $VenvDir
    }
    if (Test-VenvPip $venvPython) { return @{ Dir = $VenvDir; Python = $venvPython } }

    Write-Host "[WARN] Bootstrapping pip ..."
    & $venvPython -m ensurepip --upgrade
    if (Test-VenvPip $venvPython) { return @{ Dir = $VenvDir; Python = $venvPython } }

    try { & $BasePython -m venv --clear $VenvDir } catch {}
    if (Test-VenvPip $venvPython) { return @{ Dir = $VenvDir; Python = $venvPython } }

    $altDir = ".venv-vl-new"
    Write-Host "Creating $altDir ..."
    if (Test-Path $altDir) { try { Remove-Item -Recurse -Force $altDir } catch {} }
    & $BasePython -m venv $altDir
    $venvPython = Join-Path (Join-Path (Get-Location) $altDir) "Scripts\python.exe"
    & $venvPython -m ensurepip --upgrade
    if (-not (Test-VenvPip $venvPython)) { throw "Cannot bootstrap pip. Close terminals and retry." }
    return @{ Dir = $altDir; Python = $venvPython }
}

$PythonExe = Resolve-Python312
Write-Host "Using: $PythonExe"
& $PythonExe --version

if ($env:VIRTUAL_ENV) { deactivate 2>$null }

$venvDir = ".venv-vl"
if (Test-Path $venvDir) {
    try { Remove-Item -Recurse -Force $venvDir -ErrorAction Stop }
    catch { Write-Host "[WARN] Reusing locked venv $venvDir" -ForegroundColor Yellow }
}

$venv = Ensure-Venv $PythonExe $venvDir
$venvPython = $venv.Python
$venvDir = $venv.Dir

& $venvPython -m pip install -U pip
& $venvPython -m pip install -r requirements.txt
& $venvPython -m pip cache purge 2>$null
& $venvPython -m pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
& $venvPython -m pip install -r requirements-vl.txt

Write-Host ""
& $venvPython -c "import torch; print('cuda:', torch.cuda.is_available())"
Write-Host ""
Write-Host "Activate: .\.$venvDir\Scripts\Activate.ps1"
Write-Host "Run:      python scripts/compare_st_vs_image.py --limit 2"
