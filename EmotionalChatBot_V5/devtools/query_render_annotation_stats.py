"""
查询 Render（或 DATABASE_URL）上标注员（annotator）相关数据统计。

表：annotation_results（提交的标注结果）、annotation_task_cache（已分配任务的标记员缓存）

用法:
  RENDER_DATABASE_URL=postgresql+asyncpg://... python -m devtools.query_render_annotation_stats
  或: DATABASE_URL=postgresql+asyncpg://... python -m devtools.query_render_annotation_stats
"""
from __future__ import annotations

import asyncio
import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from sqlalchemy import text
from app.core import _create_async_engine_from_database_url


async def main() -> None:
    url = os.getenv("RENDER_DATABASE_URL") or os.getenv("DATABASE_URL")
    if not url:
        print("ERROR: 请设置 RENDER_DATABASE_URL 或 DATABASE_URL")
        sys.exit(1)

    source = "RENDER_DATABASE_URL" if os.getenv("RENDER_DATABASE_URL") else "DATABASE_URL"
    print(f"数据来源: {source}")
    print()

    engine = _create_async_engine_from_database_url(url)

    async with engine.connect() as conn:
        # 1) 检查表是否存在
        check = await conn.execute(text("""
            SELECT EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'annotation_results'
            ) AS results_exists,
            EXISTS (
                SELECT FROM information_schema.tables
                WHERE table_schema = 'public' AND table_name = 'annotation_task_cache'
            ) AS cache_exists
        """))
        row = check.fetchone()
        results_exists, cache_exists = row[0], row[1]

        if not results_exists:
            print("⚠️ 表 annotation_results 不存在（可能尚未有任何人提交过标注，或未执行过标注相关接口）。")
            await engine.dispose()
            return
        if not cache_exists:
            print("⚠️ 表 annotation_task_cache 不存在。")
            await engine.dispose()
            return

        # 2) 总体统计
        total_results = await conn.execute(text("SELECT COUNT(*) FROM annotation_results"))
        total_rows = total_results.scalar()
        distinct_annotators = await conn.execute(
            text("SELECT COUNT(DISTINCT annotator_id) FROM annotation_results")
        )
        n_annotators = distinct_annotators.scalar()

        cache_count = await conn.execute(text("SELECT COUNT(*) FROM annotation_task_cache"))
        n_cached = cache_count.scalar()

        print("========== 标记员 / 标注数据总览 ==========")
        print(f"  annotation_results 总条数:     {total_rows}")
        print(f"  有提交记录的标记员数 (去重):   {n_annotators}")
        print(f"  annotation_task_cache 条数:   {n_cached} （已分配过任务的标记员数）")
        print()

        if total_rows == 0:
            print("当前无任何标注提交记录，属空库状态。")
            await engine.dispose()
            return

        # 3) 按 task_type 统计
        by_type = await conn.execute(text("""
            SELECT task_type, COUNT(*) AS cnt
            FROM annotation_results
            GROUP BY task_type
            ORDER BY cnt DESC
        """))
        print("---------- 按题型 (task_type) 统计 ----------")
        for r in by_type:
            print(f"  {r[0] or '(空)'}: {r[1]}")
        print()

        # 4) 每位标记员提交数
        per_annotator = await conn.execute(text("""
            SELECT annotator_id, COUNT(*) AS cnt
            FROM annotation_results
            GROUP BY annotator_id
            ORDER BY cnt DESC
        """))
        rows = per_annotator.fetchall()
        print("---------- 每位标记员提交数 (按提交量降序) ----------")
        for r in rows:
            print(f"  {r[0]!r}: {r[1]} 条")
        print()

        # 5) 简单判断是否“正常”
        print("========== 简要结论 ==========")
        if n_annotators == 0:
            print("  未发现任何有提交记录的标记员，请确认是否已有标注员在线上提交。")
        elif total_rows < 10:
            print("  总提交量较少（<10 条），可能处于刚起步或测试阶段。")
        else:
            avg = total_rows / n_annotators
            print(f"  平均每位标记员约 {avg:.1f} 条提交；共 {n_annotators} 位有提交记录。")
        if n_cached > 0 and n_cached != n_annotators:
            print(f"  已分配任务的标记员数 ({n_cached}) 与有提交记录的人数 ({n_annotators}) 不一致，属正常（有人领了任务未交或只交部分）。")

    await engine.dispose()
    print("\n✅ 查询完成。")


if __name__ == "__main__":
    asyncio.run(main())
