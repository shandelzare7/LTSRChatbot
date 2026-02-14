#!/usr/bin/env bash
set -euo pipefail

# One-shot setup for Aliyun Ubuntu 24.04 LTS
# - installs deps (nginx, postgres, python)
# - clones repo
# - creates venv + installs requirements
# - creates local postgres user/db + initializes schema
# - configures systemd service (uvicorn on 127.0.0.1:8000)
# - configures nginx reverse proxy on :80 (WebSocket + upload)
# - configures daily pg_dump backups
#
# This script does NOT require OPENAI_API_KEY to be set:
# the app will run with MockLLM until you set OPENAI_API_KEY in .env and restart.

REPO_URL="${REPO_URL:-https://github.com/shandelzare7/LTSRChatbot.git}"
APP_DIR="${APP_DIR:-/opt/ltsrchatbot}"
APP_SUBDIR="${APP_SUBDIR:-EmotionalChatBot_V5}"

DB_NAME="${DB_NAME:-ltsrchatbot_v5}"
DB_USER="${DB_USER:-chatbot_user}"
SECRETS_FILE="${SECRETS_FILE:-/root/ltsrchatbot_secrets.env}"

log() { echo "[aliyun-setup] $*"; }

export DEBIAN_FRONTEND=noninteractive

log "Apt install dependencies"
apt-get update -y
apt-get upgrade -y
apt-get install -y \
  git curl ufw nginx openssl ca-certificates \
  python3 python3-venv python3-pip build-essential libpq-dev \
  postgresql postgresql-contrib

systemctl enable --now postgresql

log "Firewall allow 22/80"
ufw allow 22 >/dev/null || true
ufw allow 80 >/dev/null || true
ufw --force enable >/dev/null || true

log "Clone/pull repo into ${APP_DIR}"
mkdir -p "${APP_DIR}"
cd "${APP_DIR}"
if [ ! -d ".git" ]; then
  git clone "${REPO_URL}" .
else
  git pull
fi

if [ ! -f "${APP_DIR}/${APP_SUBDIR}/requirements.txt" ]; then
  echo "ERROR: missing ${APP_DIR}/${APP_SUBDIR}/requirements.txt" >&2
  echo "Repo URL: ${REPO_URL}" >&2
  ls -la "${APP_DIR}" >&2
  exit 1
fi

log "Create venv + pip install"
if [ ! -d "${APP_DIR}/.venv" ]; then
  python3 -m venv "${APP_DIR}/.venv"
fi
source "${APP_DIR}/.venv/bin/activate"
pip install -U pip >/dev/null
pip install -r "${APP_DIR}/${APP_SUBDIR}/requirements.txt" >/dev/null

log "Ensure DB password exists (stored at ${SECRETS_FILE})"
DB_PASS=""
if [ -f "${SECRETS_FILE}" ]; then
  # shellcheck disable=SC1090
  source "${SECRETS_FILE}" || true
  DB_PASS="${DB_PASS:-}"
fi
if [ -z "${DB_PASS}" ]; then
  DB_PASS="$(openssl rand -base64 48 | tr -dc 'A-Za-z0-9' | head -c 24)"
  printf "DB_PASS=%s\n" "${DB_PASS}" > "${SECRETS_FILE}"
  chmod 600 "${SECRETS_FILE}"
fi

log "Create postgres role/user if missing"
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_roles WHERE rolname='${DB_USER}'" | grep -q 1; then
  sudo -u postgres psql -c "CREATE USER ${DB_USER} WITH PASSWORD '${DB_PASS}';"
fi

log "Create database if missing"
if ! sudo -u postgres psql -tAc "SELECT 1 FROM pg_database WHERE datname='${DB_NAME}'" | grep -q 1; then
  sudo -u postgres createdb -O "${DB_USER}" "${DB_NAME}"
fi
sudo -u postgres psql -d "${DB_NAME}" -c "GRANT ALL PRIVILEGES ON DATABASE ${DB_NAME} TO ${DB_USER};" >/dev/null

log "Write app .env (OPENAI_API_KEY empty => MockLLM)"
ENV_FILE="${APP_DIR}/${APP_SUBDIR}/.env"
printf "%s\n" \
  "OPENAI_API_KEY=" \
  "LANGCHAIN_TRACING_V2=0" \
  "DATABASE_URL=postgresql+asyncpg://${DB_USER}:${DB_PASS}@localhost:5432/${DB_NAME}" \
  > "${ENV_FILE}"
chmod 600 "${ENV_FILE}"

log "Init schema via seed_local_postgres.py"
cd "${APP_DIR}/${APP_SUBDIR}"
python devtools/seed_local_postgres.py >/dev/null

log "Write systemd service /etc/systemd/system/ltsrchatbot.service"
cat > /etc/systemd/system/ltsrchatbot.service <<EOF
[Unit]
Description=LTSRChatbot FastAPI
After=network.target postgresql.service

[Service]
Type=simple
WorkingDirectory=${APP_DIR}/${APP_SUBDIR}
EnvironmentFile=${APP_DIR}/${APP_SUBDIR}/.env
ExecStart=${APP_DIR}/.venv/bin/uvicorn web_app:app --host 127.0.0.1 --port 8000 --proxy-headers
Restart=always
RestartSec=3

[Install]
WantedBy=multi-user.target
EOF

systemctl daemon-reload
systemctl enable --now ltsrchatbot

log "Configure nginx reverse proxy :80 -> 127.0.0.1:8000"
cat > /etc/nginx/sites-available/ltsrchatbot <<'EOF'
server {
    listen 80;
    server_name _;

    client_max_body_size 50m;

    location / {
        proxy_pass http://127.0.0.1:8000;
        proxy_http_version 1.1;

        proxy_set_header Host $host;
        proxy_set_header X-Real-IP $remote_addr;
        proxy_set_header X-Forwarded-For $proxy_add_x_forwarded_for;
        proxy_set_header X-Forwarded-Proto $scheme;

        proxy_set_header Upgrade $http_upgrade;
        proxy_set_header Connection "upgrade";

        proxy_read_timeout 3600;
        proxy_send_timeout 3600;
    }
}
EOF

rm -f /etc/nginx/sites-enabled/default || true
ln -sf /etc/nginx/sites-available/ltsrchatbot /etc/nginx/sites-enabled/ltsrchatbot
nginx -t >/dev/null
systemctl enable --now nginx
systemctl restart nginx

log "Daily pg_dump backup at 03:00 (retain 14 days)"
mkdir -p /var/backups/ltsrchatbot
chmod 700 /var/backups/ltsrchatbot
printf "%s\n" "localhost:5432:${DB_NAME}:${DB_USER}:${DB_PASS}" > /root/.pgpass
chmod 600 /root/.pgpass

cat > /usr/local/bin/ltsr_pg_backup.sh <<'EOF'
#!/usr/bin/env bash
set -euo pipefail
DB_NAME="ltsrchatbot_v5"
DB_USER="chatbot_user"
OUT="/var/backups/ltsrchatbot/backup_$(date +%F).sql.gz"
pg_dump -h localhost -U "$DB_USER" "$DB_NAME" | gzip > "$OUT"
find /var/backups/ltsrchatbot -type f -name "backup_*.sql.gz" -mtime +14 -delete
EOF
chmod +x /usr/local/bin/ltsr_pg_backup.sh
( crontab -l 2>/dev/null | grep -v ltsr_pg_backup.sh || true; echo "0 3 * * * /usr/local/bin/ltsr_pg_backup.sh" ) | crontab -

log "Self-check"
echo "--- systemd ---"
systemctl status ltsrchatbot --no-pager | sed -n '1,18p' || true
echo "--- listen ---"
ss -ltnp | grep -E ':80|:8000' || true
echo "--- curl ---"
curl -sS -I http://127.0.0.1:8000/ | head -n 5 || true
curl -sS -I http://127.0.0.1/ | head -n 5 || true

log "OK. DB_PASS is stored at ${SECRETS_FILE}"

