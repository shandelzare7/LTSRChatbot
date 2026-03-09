#!/bin/bash
# Phase 4 完成后自动启动 Phase 5
cd "$(dirname "$0")/.."

echo "=========================================="
echo "等待 Phase 4 完成..."
echo "=========================================="

# 等待 Phase 4 的进程结束（通过检查是否有 run_b2b_batch 在运行 phase4 config）
while pgrep -f "b2b_phase4_configs" > /dev/null 2>&1; do
    sleep 30
done

echo "=========================================="
echo "Phase 4 已完成，开始 Phase 5"
echo "$(date)"
echo "=========================================="

python devtools/run_b2b_batch.py --config devtools/b2b_phase5_configs.json 2>&1 | tee devtools/phase5_output.log

echo "=========================================="
echo "Phase 5 完成"
echo "$(date)"
echo "=========================================="
