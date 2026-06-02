#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Downloads Thermo RawFileReader DLLs into the installed pymsio package (Linux).
#
#   1. Displays the Thermo RawFileReader license and asks for agreement.
#   2. Locates the installed pymsio package via the active Python environment.
#   3. Downloads the required DLLs from GitHub into pymsio/dlls/thermo_fisher/.
#   4. Optionally installs Mono (required by pythonnet on Linux).
#
# Run this script AFTER installing delpi (uv pip install .) so that the DLLs
# are placed inside the correct installed pymsio package location.
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO_BASE="https://github.com/thermofisherlsms/RawFileReader/raw/main"
DLL_NAMES=(
    "ThermoFisher.CommonCore.Data.dll"
    "ThermoFisher.CommonCore.RawFileReader.dll"
)
LICENSE_URL="$REPO_BASE/License.doc"

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

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

# ── Locate installed pymsio package ─────────────────────────────────────────
echo ""
echo "[*] Locating installed pymsio package..."

PYMSIO_DIR=$(python -c "import pymsio, os; print(os.path.dirname(pymsio.__file__))" 2>&1) || true
if [ -z "$PYMSIO_DIR" ]; then
    echo "ERROR: pymsio is not installed. Run 'uv pip install .' first." >&2
    exit 1
fi
echo "    Found pymsio at: $PYMSIO_DIR"

DLL_DIR="$PYMSIO_DIR/dlls/thermo_fisher"

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

echo ""
echo "============================================================"
echo "  Thermo DLL installation complete!"
echo "============================================================"
echo ""
