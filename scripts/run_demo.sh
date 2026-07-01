#!/usr/bin/env bash
# One-command demo for judges — no API key required.
set -euo pipefail
cd "$(dirname "$0")/.."
python demo.py --mock --yes
