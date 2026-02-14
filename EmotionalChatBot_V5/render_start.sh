#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Bootstrap DB schema + default bot (idempotent-ish)
# Render may start the service before you attach a Postgres DB / env var is populated.
if [ -n "${DATABASE_URL:-}" ]; then
  python3 devtools/ensure_schema.py
else
  echo "[render_start] DATABASE_URL is not set; skipping schema bootstrap"
fi

# Start FastAPI (Render provides $PORT)
exec uvicorn web_app:app --host 0.0.0.0 --port "${PORT:-8000}"

