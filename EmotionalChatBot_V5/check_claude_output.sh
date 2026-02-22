#!/bin/bash
FILE="thesis_chapter_technical.md"
MAX_WAIT=600  # 最多等待10分钟
ELAPSED=0
INTERVAL=10

echo "监控 Claude Code 输出文件: $FILE"
echo "每 ${INTERVAL} 秒检查一次，最多等待 ${MAX_WAIT} 秒"
echo "---"

while [ $ELAPSED -lt $MAX_WAIT ]; do
    if [ -f "$FILE" ]; then
        SIZE=$(wc -c < "$FILE" 2>/dev/null || echo 0)
        if [ "$SIZE" -gt 100 ]; then
            echo "✓ 文件已生成！"
            echo "大小: $SIZE 字节"
            echo "行数: $(wc -l < "$FILE")"
            echo ""
            echo "---前50行预览---"
            head -50 "$FILE"
            exit 0
        fi
    fi
    echo "[$(date +%H:%M:%S)] 等待中... (已等待 ${ELAPSED}秒)"
    sleep $INTERVAL
    ELAPSED=$((ELAPSED + INTERVAL))
done

echo "超时：文件仍未生成或为空"
ps aux | grep "claude -p" | grep -v grep | head -1
exit 1
