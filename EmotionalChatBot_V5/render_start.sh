#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

# Bootstrap DB schema + default bot (idempotent-ish)
python3 devtools/ensure_schema.py

# Start FastAPI (Render provides $PORT)
exec uvicorn web_app:app --host 0.0.0.0 --port "${PORT:-8000}"

