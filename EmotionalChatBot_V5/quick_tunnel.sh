#!/bin/bash
# 快速获取 Cloudflare Tunnel 临时链接

echo "🌐 启动 Cloudflare Tunnel..."
echo "   确保 FastAPI 已在 http://127.0.0.1:8000 运行"
echo ""

# 启动 cloudflared 并捕获前 20 行输出
cloudflared tunnel --url http://127.0.0.1:8000 2>&1 | while IFS= read -r line; do
    echo "$line"
    # 检测到链接后，提取并显示
    if echo "$line" | grep -q "trycloudflare.com"; then
        LINK=$(echo "$line" | grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" | head -1)
        if [ ! -z "$LINK" ]; then
            echo ""
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo "✅ Cloudflare 临时链接:"
            echo "   $LINK"
            echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
            echo ""
            echo "💡 提示:"
            echo "   - 此链接在 cloudflared 运行期间有效"
            echo "   - 按 Ctrl+C 停止 tunnel"
            echo "   - 生成分享链接: WEB_DOMAIN='$LINK' python generate_share_links.py"
            echo ""
        fi
    fi
done
