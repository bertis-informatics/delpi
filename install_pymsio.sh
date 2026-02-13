#!/usr/bin/env bash
# ──────────────────────────────────────────────────────────────────────────────
# Clones and installs pymsio in one step (Linux).
#
#   1. Clones the pymsio repository from GitHub.
#   2. Runs the pymsio install script (license agreement, DLL download,
#      Mono check, pip install).
#   3. Returns to the original directory.
#
# Usage:
#   ./install_pymsio.sh                          # clone into ./pymsio
#   ./install_pymsio.sh --install-dir /tmp/pymsio  # clone into specified dir
# ──────────────────────────────────────────────────────────────────────────────
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ORIGINAL_DIR="$(pwd)"
INSTALL_DIR=""
PYMSIO_REPO="https://github.com/bertis-informatics/pymsio.git"

# ── Parse arguments ──────────────────────────────────────────────────────────
while [[ $# -gt 0 ]]; do
    case "$1" in
        --install-dir)
            INSTALL_DIR="$2"
            shift 2
            ;;
        *)
            echo "Unknown option: $1" >&2
            exit 1
            ;;
    esac
done

if [ -z "$INSTALL_DIR" ]; then
    INSTALL_DIR="$SCRIPT_DIR/pymsio"
fi

# ── Clone pymsio ─────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo "  pymsio Installer"
echo "============================================================"
echo ""

if [ -d "$INSTALL_DIR/.git" ]; then
    echo "[*] pymsio repository already exists at $INSTALL_DIR"
    echo "    Pulling latest changes..."
    cd "$INSTALL_DIR"
    git pull
    cd "$ORIGINAL_DIR"
else
    echo "[*] Cloning pymsio from $PYMSIO_REPO ..."
    git clone "$PYMSIO_REPO" "$INSTALL_DIR"
fi

# ── Run pymsio install script ────────────────────────────────────────────────
echo ""
echo "[*] Running pymsio install script ..."

INSTALL_SCRIPT="$INSTALL_DIR/install.sh"
if [ ! -f "$INSTALL_SCRIPT" ]; then
    echo "ERROR: install.sh not found at $INSTALL_SCRIPT" >&2
    exit 1
fi

cd "$INSTALL_DIR"
chmod +x install.sh
./install.sh

# ── Done ─────────────────────────────────────────────────────────────────────
cd "$ORIGINAL_DIR"
echo ""
echo "============================================================"
echo "  pymsio installation complete!"
echo "============================================================"
echo ""
