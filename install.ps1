<#
.SYNOPSIS
    Installs pymsio and downloads Thermo RawFileReader DLLs (Windows).
.DESCRIPTION
    1. Displays the Thermo RawFileReader license and asks for agreement.
    2. Downloads the required DLLs from GitHub into pymsio/dlls/thermo_fisher/.
    3. Installs pymsio via pip.
#>
param(
    [switch]$SkipPipInstall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

$REPO_BASE = "https://github.com/thermofisherlsms/RawFileReader/raw/main"
$DLL_NAMES = @(
    "ThermoFisher.CommonCore.Data.dll",
    "ThermoFisher.CommonCore.RawFileReader.dll"
)
$LICENSE_URL = "$REPO_BASE/License.doc"

$ScriptDir = Split-Path -Parent $MyInvocation.MyCommand.Definition
$DllDir = Join-Path (Join-Path (Join-Path $ScriptDir "pymsio") "dlls") "thermo_fisher"

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

# ── Install pymsio ───────────────────────────────────────────────────────────
if (-not $SkipPipInstall) {
    Write-Host ""
    Write-Host "[*] Installing pymsio ..." -ForegroundColor Green
    Push-Location $ScriptDir
    pip install .
    Pop-Location
} else {
    Write-Host ""
    Write-Host "[*] Skipping pip install (use 'pip install .' manually)." -ForegroundColor Yellow
}

Write-Host ""
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host "  Pymsio installation complete!" -ForegroundColor Cyan
Write-Host "============================================================" -ForegroundColor Cyan
Write-Host ""
