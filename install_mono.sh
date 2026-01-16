#!/usr/bin/env bash
set -euo pipefail

echo "[*] Checking existing Mono installation..."
if command -v mono >/dev/null 2>&1; then
  echo "[*] Mono already installed:"
  mono --version
  exit 0
fi

echo "[*] Installing Mono (.NET) for Ubuntu..."

# 1) Update package index and install required helpers
sudo apt update
sudo apt install -y dirmngr gnupg apt-transport-https ca-certificates software-properties-common

# 2) Add the official Mono GPG key
sudo apt-key adv --keyserver hkp://keyserver.ubuntu.com:80 \
  --recv-keys 3FA7E0328081BFF6A14DA29AA6A19B38D3D831EF

# 3) Add the Mono official repository (stable-focal works for Ubuntu 20.04/22.04)
echo "deb https://download.mono-project.com/repo/ubuntu stable-focal main" | \
  sudo tee /etc/apt/sources.list.d/mono-official-stable.list

# 4) Refresh package lists
sudo apt update

# 5) Install Mono runtime + libraries
sudo apt install -y mono-complete

# 6) Verify
echo
echo "[*] Mono version:"
mono --version || echo "Mono not found in PATH?!"

echo
echo "[*] Mono installation finished."