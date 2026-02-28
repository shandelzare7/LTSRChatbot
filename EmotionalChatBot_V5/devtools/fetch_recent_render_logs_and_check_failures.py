"""
从 Render 数据库拉取最近 N 条 web_chat_log，保存到本地并检查「无有效候选」/ Generate 异常等。

用法:
  cd EmotionalChatBot_V5
  RENDER_DATABASE_URL=... python -m devtools.fetch_recent_render_logs_and_check_failures

  可选参数:
    --limit 30    拉取最近 30 条（默认 20）
    --date 2026-02-28  只拉取该日期的 log（按 updated_at）
"""
from __future__ import annotations

import asyncio
import os
import re
import sys
from datetime import datetime, timezone, timedelta
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from sqlalchemy import select
from sqlalchemy.ext.asyncio import AsyncSession

from app.core import WebChatLog, _create_async_engine_from_database_url


async def main() -> None:
    limit = 20
    date_filter = None
    argv = list(sys.argv[1:])
    while argv:
        if argv[0] == "--limit" and len(argv) >= 2:
            limit = int(argv[1])
            argv.pop(0)
            argv.pop(0)
        elif argv[0] == "--date" and len(argv) >= 2:
            date_filter = argv[1].strip()[:10]
            argv.pop(0)
            argv.pop(0)
        else:
            argv.pop(0)

    url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL 或 DATABASE_URL（可从 .env 读取）")
        sys.exit(1)

    print(f"使用: {'RENDER_DATABASE_URL' if os.getenv('RENDER_DATABASE_URL') else 'DATABASE_URL'}")
    print(f"拉取最近 {limit} 条 web_chat_log" + (f"，日期={date_filter}" if date_filter else ""))
    print()

    engine = _create_async_engine_from_database_url(url)
    from sqlalchemy.ext.asyncio import async_sessionmaker
    async_session_factory = async_sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)

    try:
        async with async_session_factory() as session:
            q = select(WebChatLog).order_by(WebChatLog.updated_at.desc())
            if date_filter:
                try:
                    d = datetime.strptime(date_filter, "%Y-%m-%d").date()
                    start_utc = datetime(d.year, d.month, d.day, 0, 0, 0, 0, tzinfo=timezone.utc)
                    end_utc = start_utc + timedelta(days=1)
                    q = q.where(WebChatLog.updated_at >= start_utc, WebChatLog.updated_at < end_utc)
                except ValueError:
                    pass
            q = q.limit(limit)
            result = await session.execute(q)
            rows = list(result.scalars().all())
    finally:
        await engine.dispose()

    if not rows:
        print("未找到任何 web_chat_log 记录")
        return

    out_base = PROJECT_ROOT / "devtools" / "downloaded_logs"
    out_dir = out_base / "render_recent"
    out_dir.mkdir(parents=True, exist_ok=True)

    # 用于汇总：哪些 log 里出现「无有效候选」或 Generate 异常
    no_candidates: list[tuple[str, str]] = []
    generate_errors: list[tuple[str, str]] = []
    judge_warnings: list[tuple[str, str]] = []

    for i, row in enumerate(rows):
        content = row.content or ""
        sid = (row.session_id or "unknown").replace("/", "_").replace("\\", "_")[:64]
        updated = getattr(row, "updated_at", None) or ""
        if hasattr(updated, "isoformat"):
            updated = updated.isoformat()
        label = f"user_{row.user_id}_{sid}"
        fname = f"{label}.log"
        path = out_dir / fname
        if path.exists():
            path = out_dir / f"{label}_{i}.log"
        path.write_text(content, encoding="utf-8")

        # 检查该 log 是否包含失败特征
        if "无有效候选" in content:
            no_candidates.append((str(updated), str(row.session_id)))
        if "[Generate]" in content and ("异常" in content or "Exception" in content or "Error" in content):
            generate_errors.append((str(updated), str(row.session_id)))
        if "[Judge]" in content and "无有效候选" in content:
            judge_warnings.append((str(updated), str(row.session_id)))

    print(f"已保存 {len(rows)} 条 log → {out_dir}\n")
    print("=" * 60)
    print("失败/异常汇总（在 log 内容中检索）")
    print("=" * 60)
    if no_candidates or judge_warnings:
        print("\n【无有效候选】出现次数:", len(no_candidates) or len(judge_warnings))
        for updated, sid in (no_candidates or judge_warnings)[:15]:
            print(f"  - {updated}  session_id={sid[:40]}...")
    else:
        print("\n【无有效候选】: 未在本次拉取的 log 中检出")
    if generate_errors:
        print("\n【Generate 异常/Error】出现次数:", len(generate_errors))
        for updated, sid in generate_errors[:10]:
            print(f"  - {updated}  session_id={sid[:40]}...")
    else:
        print("\n【Generate 异常】: 未在本次拉取的 log 中检出")
    print("\n说明: 无有效候选 = Generate 全部路由失败或返回空，Judge 收不到任何 text。")
    print("      常见原因: LLM API 超时/502/503、API Key 失效、或 agenerate 返回空 choices。")
    print("=" * 60)


if __name__ == "__main__":
    asyncio.run(main())
