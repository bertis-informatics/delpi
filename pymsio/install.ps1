<#
.SYNOPSIS
    Downloads Thermo RawFileReader DLLs into the installed pymsio package.
.DESCRIPTION
    1. Displays the Thermo RawFileReader license and asks for agreement.
    2. Locates the installed pymsio package via the active Python environment.
    3. Downloads the required DLLs from GitHub into pymsio/dlls/thermo_fisher/.

    Run this script AFTER installing delpi (uv pip install .) so that the DLLs
    are placed inside the correct installed pymsio package location.
#>

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$REPO_BASE = "https://github.com/thermofisherlsms/RawFileReader/raw/main"
$DLL_NAMES = @(
    "ThermoFisher.CommonCore.Data.dll",
    "ThermoFisher.CommonCore.RawFileReader.dll"
)
$LICENSE_URL = "$REPO_BASE/License.doc"

# ── License agreement ────────────────────────────────────────────────────────
Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Thermo RawFileReader License Agreement" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
Write-Host "This script will download Thermo Fisher RawFileReader DLLs."
Write-Host "These DLLs are Copyright (c) Thermo Fisher Scientific."
Write-Host ""
Write-Host "By proceeding, you agree to the Thermo RawFileReader license:"
Write-Host "  $LICENSE_URL" -ForegroundColor Yellow
Write-Host ""
Write-Host "Full license: https://github.com/thermofisherlsms/RawFileReader/blob/main/License.doc"
Write-Host ""

$response = Read-Host "Do you agree to the Thermo RawFileReader license? [y/N]"
if ($response -notin @("y", "Y", "yes", "Yes", "YES")) {
    Write-Host "License not accepted. Aborting." -ForegroundColor Red
    exit 1
}

# ── Locate installed pymsio package ─────────────────────────────────────────
Write-Host ""
Write-Host "[*] Locating installed pymsio package..." -ForegroundColor Green

$pymsioDir = python -c "import pymsio, os; print(os.path.dirname(pymsio.__file__))" 2>&1
if ($LASTEXITCODE -ne 0 -or -not $pymsioDir) {
    Write-Host "ERROR: pymsio is not installed. Run 'uv pip install .' first." -ForegroundColor Red
    exit 1
}
Write-Host "    Found pymsio at: $pymsioDir" -ForegroundColor Green

$DllDir = Join-Path (Join-Path $pymsioDir "dlls") "thermo_fisher"

# ── Download DLLs ────────────────────────────────────────────────────────────
Write-Host ""
Write-Host "[*] Downloading Thermo DLLs..." -ForegroundColor Green

if (-not (Test-Path $DllDir)) {
    New-Item -ItemType Directory -Path $DllDir -Force | Out-Null
}

foreach ($dll in $DLL_NAMES) {
    $url = "$REPO_BASE/Libs/Net471/$dll"
    $dest = Join-Path $DllDir $dll
    Write-Host "    Downloading $dll ..."
    Invoke-WebRequest -Uri $url -OutFile $dest -UseBasicParsing
    if (Test-Path $dest) {
        Write-Host "    OK: $dest" -ForegroundColor Green
    } else {
        Write-Host "    FAILED: $dest" -ForegroundColor Red
        exit 1
    }
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Thermo DLL installation complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
