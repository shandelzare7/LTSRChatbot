#!/bin/bash
# Cron安装脚本：
#   每天早上5点执行阶段评估（daily_stage_evaluation.py）
#   每天早上7点更新每日聊天素材（update_daily_context.py）
#
# 用法: bash scripts/setup_cron.sh [--bot-id <bot_id>]
# 参数:
#   --bot-id  Bot ID，传给 update_daily_context.py（默认 default_bot）

set -euo pipefail

# 解析参数
BOT_ID="default_bot"
for arg in "$@"; do
    case $arg in
        --bot-id=*) BOT_ID="${arg#*=}" ;;
        --bot-id)   shift; BOT_ID="$1" ;;
    esac
done

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/.." && pwd)"
PYTHON_PATH="$(which python3)"
LOG_DIR="$PROJECT_ROOT/logs"

# 确保日志目录存在
mkdir -p "$LOG_DIR"

# 检查Python是否存在
if [ -z "$PYTHON_PATH" ]; then
    echo "错误: 未找到 python3"
    exit 1
fi

# ── 任务1：阶段评估（每天 05:00）──────────────────────────────────────────
STAGE_SCRIPT="$SCRIPT_DIR/daily_stage_evaluation.py"
if [ -f "$STAGE_SCRIPT" ]; then
    CRON_STAGE="0 5 * * * cd $PROJECT_ROOT && $PYTHON_PATH $STAGE_SCRIPT >> $LOG_DIR/cron_stage.log 2>&1"
    if crontab -l 2>/dev/null | grep -q "daily_stage_evaluation.py"; then
        echo "警告: stage 评估 Cron 已存在，将更新"
        (crontab -l 2>/dev/null | grep -v "daily_stage_evaluation.py" || true) | crontab -
    fi
    (crontab -l 2>/dev/null || true; echo "$CRON_STAGE") | crontab -
    echo "✓ 阶段评估 Cron 已安装 (每天 05:00)"
else
    echo "跳过阶段评估（脚本不存在: $STAGE_SCRIPT）"
fi

# ── 任务2：每日素材更新（每天 07:00）─────────────────────────────────────
CONTEXT_SCRIPT="$SCRIPT_DIR/update_daily_context.py"
if [ ! -f "$CONTEXT_SCRIPT" ]; then
    echo "错误: 脚本不存在: $CONTEXT_SCRIPT"
    exit 1
fi
CRON_CONTEXT="0 7 * * * cd $PROJECT_ROOT && $PYTHON_PATH $CONTEXT_SCRIPT --bot-id $BOT_ID >> $LOG_DIR/cron_context.log 2>&1"
if crontab -l 2>/dev/null | grep -q "update_daily_context.py"; then
    echo "警告: 素材更新 Cron 已存在，将更新"
    (crontab -l 2>/dev/null | grep -v "update_daily_context.py" || true) | crontab -
fi
(crontab -l 2>/dev/null || true; echo "$CRON_CONTEXT") | crontab -
echo "✓ 每日素材更新 Cron 已安装 (每天 07:00, bot-id=$BOT_ID)"

echo ""
echo "当前 cron 任务："
crontab -l 2>/dev/null | grep -E "daily_stage|update_daily_context" || echo "（无）"
echo ""
echo "查看日志："
echo "  tail -f $LOG_DIR/cron_stage.log"
echo "  tail -f $LOG_DIR/cron_context.log"
