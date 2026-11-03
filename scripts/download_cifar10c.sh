#!/usr/bin/env bash
# Download, verify and extract CIFAR-10-C. Run from the repo root; arguments are passed on,
# for example: bash scripts/download_cifar10c.sh --data-dir /content/data --verify-only
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PYTHON:-python}" -m scripts.download_cifar10c "$@"
