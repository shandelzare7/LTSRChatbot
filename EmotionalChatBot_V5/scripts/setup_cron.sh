#!/bin/bash
# Cron安装脚本：每天早上5点执行阶段评估
# 用法: bash scripts/setup_cron.sh

set -euo pipefail

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

# 检查脚本是否存在
SCRIPT_PATH="$SCRIPT_DIR/daily_stage_evaluation.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "错误: 脚本不存在: $SCRIPT_PATH"
    exit 1
fi

# 添加cron任务（如果不存在）
CRON_CMD="0 5 * * * cd $PROJECT_ROOT && $PYTHON_PATH $SCRIPT_PATH >> $LOG_DIR/cron.log 2>&1"

# 检查是否已存在
if crontab -l 2>/dev/null | grep -q "daily_stage_evaluation.py"; then
    echo "警告: Cron任务已存在，将更新"
    # 移除旧任务
    (crontab -l 2>/dev/null | grep -v "daily_stage_evaluation.py" || true) | crontab -
fi

# 添加新任务
(crontab -l 2>/dev/null || true; echo "$CRON_CMD") | crontab -

echo "✓ Cron任务已安装"
echo "  时间: 每天早上5:00"
echo "  命令: $CRON_CMD"
echo ""
echo "查看cron任务: crontab -l"
echo "删除cron任务: crontab -e"
echo "查看日志: tail -f $LOG_DIR/cron.log"
