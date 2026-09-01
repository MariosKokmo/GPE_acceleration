<#
.SYNOPSIS
    Build pipeline for the GPE Simulation Studio GUI.

.DESCRIPTION
    Runs the test suite, then builds the standalone GUI executable with
    PyInstaller into dist/. The GUI is decoupled from the solver, so the exe
    bundles only the UI (PySide6); at runtime it launches the solver via a
    chosen interpreter / the installed `baqs` command.

.PARAMETER SkipTests
    Skip the test suite and build directly.

.PARAMETER Run
    Launch the built exe after a successful build.

.PARAMETER Clean
    Remove previous build/ and dist/ output before building.

.EXAMPLE
    ./build_gui.ps1                 # test, then build
    ./build_gui.ps1 -SkipTests      # build only
    ./build_gui.ps1 -Clean -Run     # clean build, then launch the exe
#>
[CmdletBinding()]
param(
    [switch]$SkipTests,
    [switch]$Run,
    [switch]$Clean
)

$ErrorActionPreference = 'Stop'
$root = $PSScriptRoot
Set-Location $root

$AppName = 'GPE Simulation Studio'
$ExePath = Join-Path $root "dist/$AppName/$AppName.exe"

# --- locate the interpreter (prefer the project venv) -----------------------
$py = Join-Path $root '.venv/Scripts/python.exe'
if (-not (Test-Path $py)) {
    Write-Host "venv python not found; falling back to 'python' on PATH." -ForegroundColor Yellow
    $py = 'python'
}
Write-Host "Interpreter: $py" -ForegroundColor DarkGray

function Invoke-Step {
    param([string]$Desc, [scriptblock]$Action)
    Write-Host ""
    Write-Host "==> $Desc" -ForegroundColor Cyan
    & $Action
    if ($LASTEXITCODE -ne 0) {
        Write-Host "FAILED: $Desc (exit $LASTEXITCODE)" -ForegroundColor Red
        exit $LASTEXITCODE
    }
}

# --- ensure PyInstaller is available ----------------------------------------
$savedPref = $ErrorActionPreference
$ErrorActionPreference = 'Continue'
& $py -m pip show pyinstaller 2>$null | Out-Null
$hasPyInstaller = ($LASTEXITCODE -eq 0)
$ErrorActionPreference = $savedPref
if (-not $hasPyInstaller) {
    Invoke-Step "Installing PyInstaller" { & $py -m pip install pyinstaller }
}

# --- 1/2: tests -------------------------------------------------------------
if (-not $SkipTests) {
    Invoke-Step "[1/2] Running tests" {
        & $py -m unittest discover -s tests -p "test_*.py"
    }
} else {
    Write-Host ""
    Write-Host "==> [1/2] Skipping tests (-SkipTests)" -ForegroundColor Yellow
}

# --- optional clean ---------------------------------------------------------
if ($Clean) {
    Write-Host ""
    Write-Host "==> Cleaning build/ and dist/" -ForegroundColor Cyan
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $root 'build')
    Remove-Item -Recurse -Force -ErrorAction SilentlyContinue (Join-Path $root 'dist')
}

# --- 2/2: build -------------------------------------------------------------
Invoke-Step "[2/2] Building GUI exe" {
    & $py -m PyInstaller --noconfirm --clean --windowed --name $AppName --paths gui gui/run_gui.py
}

if (-not (Test-Path $ExePath)) {
    Write-Host "Build reported success but exe not found at:`n  $ExePath" -ForegroundColor Red
    exit 1
}

Write-Host ""
Write-Host "Build complete:" -ForegroundColor Green
Write-Host "  $ExePath"
Write-Host "  (distribute the whole 'dist/$AppName' folder, not just the .exe)" -ForegroundColor DarkGray

if ($Run) {
    Write-Host ""
    Write-Host "==> Launching $AppName" -ForegroundColor Cyan
    Start-Process -FilePath $ExePath
}
