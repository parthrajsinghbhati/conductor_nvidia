#!/usr/bin/env bash
# Live NVIDIA API demo — requires NVIDIA_API_KEY in .env
set -euo pipefail
cd "$(dirname "$0")/.."

if [[ ! -f .env ]] || ! grep -qE '^NVIDIA_API_KEY=nvapi-' .env 2>/dev/null; then
  echo "❌ Set NVIDIA_API_KEY in .env first (get a free key at https://build.nvidia.com)"
  echo "   cp .env.example .env   # then edit .env"
  exit 1
fi

echo "Validating API key…"
python scripts/validate_key.py || exit 1

echo ""
echo "Starting live demo (quick mode: 3 questions, ~10–20 min)…"
python demo.py --real --quick --yes
