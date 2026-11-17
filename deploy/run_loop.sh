#!/usr/bin/env bash
# Cron entry: loads .env into the environment (never echoed) and runs the nightly loop.
set -euo pipefail
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
set -a
if [ -f .env ]; then
  # shellcheck disable=SC1091
  . ./.env
fi
set +a
exec .venv/bin/python -m ml.loop.run --grow "${LOOP_GROW:-200}"
