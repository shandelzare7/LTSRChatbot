#!/bin/bash
# 启动 Cloudflare Tunnel 并显示链接（保持运行）

echo "🌐 启动 Cloudflare Tunnel..."
echo "   确保 FastAPI 已在 http://127.0.0.1:8000 运行"
echo ""

# 检查 FastAPI 是否运行
if ! lsof -ti:8000 > /dev/null 2>&1; then
    echo "❌ FastAPI 未运行，请先启动 FastAPI:"
    echo "   cd $(pwd) && python web_app.py"
    exit 1
fi

echo "✅ FastAPI 正在运行"
echo ""

# 启动 cloudflared 并捕获链接
cloudflared tunnel --url http://127.0.0.1:8000 2>&1 | while IFS= read -r line; do
    echo "$line"
    # 检测到链接后，提取并高亮显示
    if echo "$line" | grep -q "trycloudflare.com"; then
        LINK=$(echo "$line" | grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" | head -1)
        if [ ! -z "$LINK" ]; then
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "✅ Cloudflare 临时链接已生成:"
            echo ""
            echo "   $LINK"
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "💡 提示:"
            echo "   - 此链接在 cloudflared 运行期间有效"
            echo "   - 按 Ctrl+C 停止 tunnel（链接会失效）"
            echo "   - 生成分享链接: WEB_DOMAIN='$LINK' python generate_share_links.py"
            echo ""
        fi
    fi
done
