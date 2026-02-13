#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Installs pymsio and downloads Thermo RawFileReader DLLs (Linux).
#
#   1. Displays the Thermo RawFileReader license and asks for agreement.
#   2. Downloads the required DLLs from GitHub into pymsio/dlls/thermo_fisher/.
#   3. Optionally installs Mono (required by pythonnet on Linux).
#   4. Installs pymsio via pip.
#
# Usage:
#   ./install.sh                  # full install
#   ./install.sh --skip-pip       # download DLLs only, skip pip install
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_BASE="https://github.com/thermofisherlsms/RawFileReader/raw/main"
DLL_NAMES=(
    "ThermoFisher.CommonCore.Data.dll"
    "ThermoFisher.CommonCore.RawFileReader.dll"
)
LICENSE_URL="$REPO_BASE/License.doc"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
DLL_DIR="$SCRIPT_DIR/pymsio/dlls/thermo_fisher"

SKIP_PIP=false
for arg in "$@"; do
    case "$arg" in
        --skip-pip) SKIP_PIP=true ;;
    esac
done

# ── License agreement ────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  Thermo RawFileReader License Agreement"
echo "============================================================"
echo ""
echo "This script will download Thermo Fisher RawFileReader DLLs."
echo "These DLLs are Copyright (c) Thermo Fisher Scientific."
echo ""
echo "By proceeding, you agree to the Thermo RawFileReader license:"
echo "  $LICENSE_URL"
echo ""
echo "Full license: https://github.com/thermofisherlsms/RawFileReader/blob/main/License.doc"
echo ""

read -rp "Do you agree to the Thermo RawFileReader license? [y/N] " response
case "$response" in
    [yY]|[yY][eE][sS]) ;;
    *)
        echo "License not accepted. Aborting."
        exit 1
        ;;
esac

# ── Download DLLs ────────────────────────────────────────────────────────────
echo ""
echo "[*] Downloading Thermo DLLs..."

mkdir -p "$DLL_DIR"

for dll in "${DLL_NAMES[@]}"; do
    url="$REPO_BASE/Libs/Net471/$dll"
    dest="$DLL_DIR/$dll"
    echo "    Downloading $dll ..."
    if command -v curl &>/dev/null; then
        curl -fsSL -o "$dest" "$url"
    elif command -v wget &>/dev/null; then
        wget -q -O "$dest" "$url"
    else
        echo "    ERROR: Neither curl nor wget found. Please install one." >&2
        exit 1
    fi

    if [ -f "$dest" ]; then
        echo "    OK: $dest"
    else
        echo "    FAILED: $dest"
        exit 1
    fi
done

# ── Mono check ───────────────────────────────────────────────────────────────
echo ""
if command -v mono &>/dev/null; then
    echo "[*] Mono is already installed: $(mono --version | head -1)"
else
    read -rp "[?] Mono is required on Linux. Install it now? [y/N] " mono_response
    case "$mono_response" in
        [yY]|[yY][eE][sS])
            echo "[*] Installing Mono..."
            bash "$SCRIPT_DIR/install_mono.sh"
            ;;
        *)
            echo "[!] Skipping Mono installation. pythonnet may not work without it."
            ;;
    esac
fi

# ── Install pymsio ───────────────────────────────────────────────────────────
if [ "$SKIP_PIP" = false ]; then
    echo ""
    echo "[*] Installing pymsio ..."
    cd "$SCRIPT_DIR"
    pip install .
else
    echo ""
    echo "[*] Skipping pip install (use 'pip install .' manually)."
fi

echo ""
echo "============================================================"
echo "  Installation complete!"
echo "============================================================"
echo ""
