<#
.SYNOPSIS
    Clones and installs pymsio in one step (Windows).
.DESCRIPTION
    1. Clones the pymsio repository from GitHub.
    2. Runs the pymsio install script (license agreement, DLL download, pip install).
    3. Returns to the original directory.
.PARAMETER InstallDir
    Directory where pymsio will be cloned. Defaults to a 'pymsio' folder
    next to this script.
#>
param(
    [string]$InstallDir = ""
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$OriginalDir = Get-Location

if ([string]::IsNullOrWhiteSpace($InstallDir)) {
    $InstallDir = Join-Path $ScriptDir "pymsio"
}

$PYMSIO_REPO = "https://github.com/bertis-informatics/pymsio.git"

# ── Clone pymsio ─────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  pymsio Installer" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""

if (Test-Path (Join-Path $InstallDir ".git")) {
    Write-Host "[*] pymsio repository already exists at $InstallDir" -ForegroundColor Yellow
    Write-Host "    Pulling latest changes..."
    Push-Location $InstallDir
    git pull
    Pop-Location
} else {
    Write-Host "[*] Cloning pymsio from $PYMSIO_REPO ..." -ForegroundColor Green
    git clone $PYMSIO_REPO $InstallDir
}

# ── Run pymsio install script ────────────────────────────────────────────────
Write-Host ""
Write-Host "[*] Running pymsio install script ..." -ForegroundColor Green

$installScript = Join-Path $InstallDir "install.ps1"
if (-not (Test-Path $installScript)) {
    Write-Host "ERROR: install.ps1 not found at $installScript" -ForegroundColor Red
    exit 1
}

Push-Location $InstallDir
& $installScript
Pop-Location

# ── Done ─────────────────────────────────────────────────────────────────────
Set-Location $OriginalDir
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  pymsio installation complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
