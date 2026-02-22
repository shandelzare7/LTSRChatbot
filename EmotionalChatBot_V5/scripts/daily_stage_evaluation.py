#!/usr/bin/env python3
"""
定时SPT阶段评估脚本

每天早上5点自动执行，评估所有最近N天内有对话的会话的SPT阶段，
更新数据库中的current_stage字段，并记录评估日志。

用法:
    python scripts/daily_stage_evaluation.py
    或通过cron: 0 5 * * * python /path/to/scripts/daily_stage_evaluation.py

环境变量:
    DATABASE_URL: 数据库连接字符串（必需）
    DAYS_THRESHOLD: 查询最近N天的对话（默认7天）
    LOG_LEVEL: 日志级别（默认INFO）
"""

from __future__ import annotations

import argparse
import asyncio
import logging
import os
import sys
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

# 添加项目根目录到路径
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

try:
    from utils.env_loader import load_project_env
    load_project_env(project_root)
except Exception:
    pass

from sqlalchemy import select, distinct, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.core.database import DBManager, Message, User, _create_async_engine_from_database_url
from app.nodes.stage_manager import KnappStageManager


# 配置日志
def setup_logging(log_level: str = "INFO", log_to_file: bool = True) -> logging.Logger:
    """设置日志记录"""
    logger = logging.getLogger("daily_stage_evaluation")
    logger.setLevel(getattr(logging, log_level.upper(), logging.INFO))
    
    # 清除现有handlers
    logger.handlers.clear()
    
    # 控制台输出（stdout，Render会捕获）
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(logging.INFO)
    console_formatter = logging.Formatter(
        "%(asctime)s [%(levelname)s] %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S"
    )
    console_handler.setFormatter(console_formatter)
    logger.addHandler(console_handler)
    
    # 文件输出（如果可写）
    if log_to_file:
        log_dir = project_root / "logs"
        log_dir.mkdir(exist_ok=True)
        log_file = log_dir / f"stage_evaluation_{datetime.now(timezone.utc).strftime('%Y-%m-%d')}.log"
        try:
            file_handler = logging.FileHandler(log_file, encoding="utf-8")
            file_handler.setLevel(logging.DEBUG)
            file_formatter = logging.Formatter(
                "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                datefmt="%Y-%m-%d %H:%M:%S"
            )
            file_handler.setFormatter(file_formatter)
            logger.addHandler(file_handler)
        except Exception as e:
            logger.warning(f"无法创建日志文件 {log_file}: {e}")
    
    return logger


async def get_active_users(db: DBManager, days: int = 7) -> List[tuple[str, str]]:
    """
    查询最近N天内有对话的用户
    
    Returns:
        List of (user_external_id, bot_id) tuples
    """
    cutoff_date = datetime.now(timezone.utc) - timedelta(days=days)
    
    async with db.Session() as session:
        # 查询最近有消息的用户，并获取对应的bot_id和external_id
        # Message.user_id -> User.id, User.bot_id, User.external_id
        query = (
            select(distinct(User.external_id), User.bot_id)
            .join(Message, Message.user_id == User.id)
            .where(Message.created_at >= cutoff_date)
        )
        result = await session.execute(query)
        rows = result.all()
        return [(str(external_id), str(bot_id)) for external_id, bot_id in rows]


async def load_user_state_for_evaluation(
    db: DBManager, user_id: str, bot_id: str
) -> Optional[Dict[str, Any]]:
    """
    为评估加载用户状态
    
    Returns:
        包含评估所需字段的state字典，如果加载失败返回None
    """
    try:
        # 使用DBManager的load_state方法加载完整状态
        state_data = await db.load_state(user_id, bot_id)
        
        # 确保包含评估所需的所有字段
        current_stage = state_data.get("current_stage") or "initiating"
        relationship_state = state_data.get("relationship_state") or {}
        relationship_assets = state_data.get("relationship_assets") or {}
        spt_info = state_data.get("spt_info") or {}
        chat_buffer = state_data.get("chat_buffer") or []
        user_basic_info = state_data.get("user_basic_info") or {}
        user_inferred_profile = state_data.get("user_inferred_profile") or {}
        
        # 构建评估用的state字典
        eval_state = {
            "current_stage": current_stage,
            "relationship_state": relationship_state,
            "relationship_assets": relationship_assets,
            "spt_info": spt_info,
            "chat_buffer": chat_buffer,
            "user_basic_info": user_basic_info,
            "user_inferred_profile": user_inferred_profile,
            "relationship_deltas": {},  # 定时评估时没有deltas
            "relationship_deltas_applied": {},
            "latest_relationship_analysis": {"detected_signals": []},
        }
        
        return eval_state
        
    except Exception as e:
        return None


async def update_user_stage(
    db: DBManager, user_id: str, bot_id: str, new_stage: str, logger: logging.Logger
) -> bool:
    """
    更新用户阶段
    
    Returns:
        True if updated successfully, False otherwise
    """
    try:
        async with db.Session() as session:
            async with session.begin():
                user = await db._get_or_create_user(session, bot_id, user_id)
                user.current_stage = new_stage
                await session.commit()
                return True
    except Exception as e:
        logger.error(f"更新用户阶段失败 user_id={user_id}, bot_id={bot_id}, new_stage={new_stage}: {e}")
        return False


async def evaluate_user_stage(
    db: DBManager,
    stage_manager: KnappStageManager,
    user_id: str,
    bot_id: str,
    logger: logging.Logger,
) -> Dict[str, Any]:
    """
    评估单个用户的阶段
    
    Returns:
        包含评估结果的字典
    """
    result = {
        "user_id": user_id,
        "bot_id": bot_id,
        "success": False,
        "old_stage": None,
        "new_stage": None,
        "transition_type": None,
        "reason": None,
        "error": None,
    }
    
    try:
        # 加载状态
        state = await load_user_state_for_evaluation(db, user_id, bot_id)
        if state is None:
            result["error"] = "无法加载用户状态"
            return result
        
        current_stage = state.get("current_stage") or "initiating"
        result["old_stage"] = current_stage
        
        # 执行评估
        eval_result = stage_manager.evaluate_transition(current_stage, state)
        
        new_stage = eval_result.get("new_stage", current_stage)
        transition_type = eval_result.get("transition_type", "STAY")
        reason = eval_result.get("reason", "")
        
        result["new_stage"] = new_stage
        result["transition_type"] = transition_type
        result["reason"] = reason
        result["success"] = True
        
        # 如果阶段发生变化，更新数据库
        if new_stage != current_stage:
            updated = await update_user_stage(db, user_id, bot_id, new_stage, logger)
            if not updated:
                result["error"] = "数据库更新失败"
                result["success"] = False
        
        return result
        
    except Exception as e:
        logger.error(f"评估用户阶段失败 user_id={user_id}, bot_id={bot_id}: {e}", exc_info=True)
        result["error"] = str(e)
        return result


async def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="定时SPT阶段评估脚本")
    parser.add_argument(
        "--days",
        type=int,
        default=int(os.getenv("DAYS_THRESHOLD", "7")),
        help="查询最近N天的对话（默认7天）",
    )
    parser.add_argument(
        "--log-level",
        type=str,
        default=os.getenv("LOG_LEVEL", "INFO"),
        help="日志级别（默认INFO）",
    )
    parser.add_argument(
        "--no-file-log",
        action="store_true",
        help="不写入日志文件（仅输出到stdout）",
    )
    args = parser.parse_args()
    
    # 设置日志
    logger = setup_logging(args.log_level, log_to_file=not args.no_file_log)
    
    logger.info("=" * 80)
    logger.info("开始定时SPT阶段评估")
    logger.info(f"查询最近 {args.days} 天内有对话的用户")
    logger.info("=" * 80)
    
    # 检查数据库连接
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        logger.error("DATABASE_URL 环境变量未设置")
        sys.exit(1)
    
    # 初始化数据库管理器
    try:
        engine = _create_async_engine_from_database_url(database_url)
        db = DBManager(engine)
        await db.ensure_memory_schema()
    except Exception as e:
        logger.error(f"数据库连接失败: {e}", exc_info=True)
        sys.exit(1)
    
    # 初始化阶段管理器
    try:
        stage_manager = KnappStageManager()
    except Exception as e:
        logger.error(f"初始化阶段管理器失败: {e}", exc_info=True)
        sys.exit(1)
    
    # 查询活跃用户
    try:
        active_users = await get_active_users(db, days=args.days)
        logger.info(f"找到 {len(active_users)} 个活跃用户")
    except Exception as e:
        logger.error(f"查询活跃用户失败: {e}", exc_info=True)
        sys.exit(1)
    
    if not active_users:
        logger.info("没有活跃用户需要评估")
        await engine.dispose()
        return
    
    # 评估每个用户
    results: List[Dict[str, Any]] = []
    transitions_count = 0
    errors_count = 0
    
    for user_id, bot_id in active_users:
        result = await evaluate_user_stage(db, stage_manager, user_id, bot_id, logger)
        results.append(result)
        
        if result["success"]:
            if result["old_stage"] != result["new_stage"]:
                transitions_count += 1
                logger.info(
                    f"阶段变更: user_id={user_id}, bot_id={bot_id}, "
                    f"{result['old_stage']} -> {result['new_stage']} "
                    f"({result['transition_type']}): {result['reason']}"
                )
            else:
                logger.debug(
                    f"阶段不变: user_id={user_id}, bot_id={bot_id}, "
                    f"stage={result['old_stage']}"
                )
        else:
            errors_count += 1
            logger.warning(
                f"评估失败: user_id={user_id}, bot_id={bot_id}, "
                f"error={result.get('error', 'Unknown error')}"
            )
    
    # 输出摘要
    logger.info("=" * 80)
    logger.info("评估完成")
    logger.info(f"总用户数: {len(active_users)}")
    logger.info(f"阶段变更数: {transitions_count}")
    logger.info(f"错误数: {errors_count}")
    logger.info("=" * 80)
    
    # 清理资源
    await engine.dispose()
    
    # 如果有错误，返回非零退出码
    if errors_count > 0:
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())
