"""
auto_configure_database_url.py

自动探测本地 PostgreSQL 连接串，并写入 EmotionalChatBot_V5/.env 的 DATABASE_URL。

原理：
- 使用 asyncpg 尝试连接常见的本地组合（host/port/user/db）
- 成功后写入 .env（不会回显密码）

用法：
  cd EmotionalChatBot_V5
  python3 devtools/auto_configure_database_url.py
"""

from __future__ import annotations

import asyncio
import getpass
import os
import re
from pathlib import Path
from typing import Iterable, Optional, Tuple

import asyncpg


def _redact(url: str) -> str:
    # redact password if present: postgresql://user:pass@host/db
    return re.sub(r"(postgresql\+asyncpg://[^:]+:)[^@]+@", r"\1<redacted>@", url)


def _candidate_urls() -> Iterable[str]:
    user = getpass.getuser()
    hosts = ["localhost", "127.0.0.1"]
    ports = [5432, 5433]
    dbs = ["ltsrchatbot", "ltsr_chatbot", "postgres", user]

    # no password candidates
    for h in hosts:
        for p in ports:
            for db in dbs:
                yield f"postgresql+asyncpg://{user}@{h}:{p}/{db}"

    # common postgres superuser name (may work in some setups)
    for h in hosts:
        for p in ports:
            for db in dbs:
                yield f"postgresql+asyncpg://postgres@{h}:{p}/{db}"


async def _can_connect(url: str) -> Tuple[bool, str]:
    try:
        # asyncpg accepts DSN without driver prefix; strip "+asyncpg"
        dsn = url.replace("postgresql+asyncpg://", "postgresql://", 1)
        conn = await asyncpg.connect(dsn=dsn, timeout=2.5)
        try:
            v = await conn.fetchval("select version()")
        finally:
            await conn.close()
        return True, str(v or "")
    except Exception as e:
        return False, str(e)


def _write_env(env_path: Path, database_url: str) -> None:
    env_path.parent.mkdir(parents=True, exist_ok=True)
    if env_path.exists():
        text = env_path.read_text(encoding="utf-8")
    else:
        text = ""

    lines = text.splitlines()
    out = []
    replaced = False
    for ln in lines:
        if ln.strip().startswith("DATABASE_URL="):
            out.append(f"DATABASE_URL={database_url}")
            replaced = True
        else:
            out.append(ln)
    if not replaced:
        if out and out[-1].strip() != "":
            out.append("")
        out.append(f"DATABASE_URL={database_url}")
    env_path.write_text("\n".join(out).rstrip() + "\n", encoding="utf-8")


async def main() -> int:
    env_path = Path(__file__).resolve().parents[1] / ".env"
    if os.getenv("DATABASE_URL"):
        print("DATABASE_URL 已在环境变量中设置：不自动覆盖。")
        print("当前（打码）:", _redact(os.getenv("DATABASE_URL") or ""))
        return 0

    print("开始探测本地 PostgreSQL ...")
    last_err: Optional[str] = None
    for url in _candidate_urls():
        ok, info = await _can_connect(url)
        if ok:
            print("✅ 连接成功，将写入 .env：", _redact(url))
            _write_env(env_path, url)
            print("服务器信息:", info.split("\n")[0])
            return 0
        last_err = info

    print("❌ 未能自动探测到可用的本地 PostgreSQL 连接串。")
    if last_err:
        print("最后一次错误(摘要):", last_err[:180])
    print("\n你可以手动在 .env 填写 DATABASE_URL，例如：")
    print("DATABASE_URL=postgresql+asyncpg://USER:PASSWORD@localhost:5432/DBNAME")
    return 2


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))

