#!/usr/bin/env python3
from __future__ import annotations

"""
Push a local API key (from EmotionalChatBot_V5/.env) to a remote server .env,
optionally set OpenAI-compatible base_url/model, restart systemd service,
and print only the key length (never the key).

Default target matches the current Aliyun setup in this repo:
- server: root@8.218.152.147
- ssh key: ~/.ssh/ltsr_deploy_key_nopass
- remote env: /opt/ltsrchatbot/EmotionalChatBot_V5/.env
- service: ltsrchatbot
"""

import argparse
import os
import re
import subprocess
import tempfile
from pathlib import Path


def _read_local_key(local_env_path: Path, var_name: str) -> str:
    text = local_env_path.read_text(encoding="utf-8", errors="ignore")
    m = re.search(r"^" + re.escape(var_name) + r"=(.*)$", text, re.MULTILINE)
    if not m:
        raise SystemExit(f"{var_name} not found in {local_env_path}")
    key = (m.group(1) or "").strip()
    if not key:
        raise SystemExit(f"{var_name} is empty in {local_env_path}")
    return key


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--server", default="root@8.218.152.147")
    parser.add_argument("--ssh-key", default=os.path.expanduser("~/.ssh/ltsr_deploy_key_nopass"))
    parser.add_argument("--remote-env", default="/opt/ltsrchatbot/EmotionalChatBot_V5/.env")
    parser.add_argument("--service", default="ltsrchatbot")
    parser.add_argument("--local-env", default=str(Path(__file__).resolve().parents[1] / ".env"))
    parser.add_argument(
        "--local-key-var",
        default="OPENAI_API_KEY",
        help="Which variable in local .env to read (e.g. QWEN_API_KEY).",
    )
    parser.add_argument(
        "--openai-base-url",
        default="",
        help="Optional OpenAI-compatible base_url to write into remote .env (OPENAI_BASE_URL).",
    )
    parser.add_argument(
        "--openai-model",
        default="",
        help="Optional model name to write into remote .env (OPENAI_MODEL).",
    )
    args = parser.parse_args()

    local_env = Path(args.local_env).resolve()
    key = _read_local_key(local_env, args.local_key_var)

    ssh_base = [
        "ssh",
        "-i",
        args.ssh_key,
        "-o",
        "BatchMode=yes",
        "-o",
        "StrictHostKeyChecking=no",
        args.server,
    ]
    scp_base = [
        "scp",
        "-i",
        args.ssh_key,
        "-o",
        "StrictHostKeyChecking=no",
    ]

    # Remote script uses env var KEY (do not print it), updates env file, restarts service.
    remote_script = f"""#!/usr/bin/env bash
set -euo pipefail
ENV_FILE="{args.remote_env}"
SERVICE="{args.service}"

if [ -z "$KEY" ]; then
  echo "empty key" >&2
  exit 1
fi

python3 - <<'PY'
import os
import re
from pathlib import Path

env_file = Path(os.environ["ENV_FILE"])
key = os.environ["KEY"]
base_url = os.environ.get("OPENAI_BASE_URL", "").strip()
model = os.environ.get("OPENAI_MODEL", "").strip()

lines = env_file.read_text(encoding="utf-8", errors="ignore").splitlines() if env_file.exists() else []
out = []
seen = False
for line in lines:
    if line.startswith("OPENAI_API_KEY="):
        out.append("OPENAI_API_KEY=" + key)
        seen = True
    elif base_url and line.startswith("OPENAI_BASE_URL="):
        out.append("OPENAI_BASE_URL=" + base_url)
    elif model and line.startswith("OPENAI_MODEL="):
        out.append("OPENAI_MODEL=" + model)
    else:
        out.append(line)
if not seen:
    out.insert(0, "OPENAI_API_KEY=" + key)
if base_url and not any(l.startswith("OPENAI_BASE_URL=") for l in out):
    out.insert(1 if out and out[0].startswith("OPENAI_API_KEY=") else 0, "OPENAI_BASE_URL=" + base_url)
if model and not any(l.startswith("OPENAI_MODEL=") for l in out):
    out.insert(1 if out and out[0].startswith("OPENAI_API_KEY=") else 0, "OPENAI_MODEL=" + model)
env_file.write_text("\\n".join(out) + "\\n", encoding="utf-8")
PY

systemctl restart "$SERVICE"
sleep 2

python3 - <<'PY'
import re
from pathlib import Path
p = Path("{args.remote_env}")
t = p.read_text(encoding="utf-8", errors="ignore")
m = re.search(r"^OPENAI_API_KEY=(.*)$", t, re.M)
val = (m.group(1).strip() if m else "")
print("OPENAI_API_KEY_len", len(val))
bm = re.search(r"^OPENAI_BASE_URL=(.*)$", t, re.M)
mm = re.search(r"^OPENAI_MODEL=(.*)$", t, re.M)
print("OPENAI_BASE_URL_set", bool((bm.group(1).strip() if bm else \"\")))
print("OPENAI_MODEL", (mm.group(1).strip() if mm else \"\"))
PY
"""

    with tempfile.TemporaryDirectory() as td:
        local_script = Path(td) / "set_openai_key.sh"
        local_script.write_text(remote_script, encoding="utf-8")
        subprocess.run(["chmod", "+x", str(local_script)], check=True)

        remote_path = "/root/set_openai_key.sh"
        subprocess.run(scp_base + [str(local_script), f"{args.server}:{remote_path}"], check=True)

    # backup remote env (best-effort)
    subprocess.run(
        ssh_base + [f"cp -a '{args.remote_env}' '{args.remote_env}.bak.$(date +%F_%H%M%S)' || true"],
        check=False,
    )

    # run remote script with key via stdin exactly once, then export KEY for the script
    exports = ""
    if args.openai_base_url:
        exports += f" export OPENAI_BASE_URL='{args.openai_base_url}';"
    if args.openai_model:
        exports += f" export OPENAI_MODEL='{args.openai_model}';"

    cmd = (
        f"set -euo pipefail; "
        f"read -r KEY; export KEY; "
        + exports
        + f" ENV_FILE='{args.remote_env}' bash /root/set_openai_key.sh"
    )
    subprocess.run(ssh_base + [cmd], input=(key + "\n").encode("utf-8"), check=True)

    # quick check
    subprocess.run(
        ssh_base + [f"systemctl is-active {args.service} && curl -sS http://127.0.0.1:8000/api/bots | head -c 200"],
        check=True,
    )

    print("OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

