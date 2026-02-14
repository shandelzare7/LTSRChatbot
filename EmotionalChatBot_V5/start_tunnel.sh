#!/bin/bash
# 启动 Cloudflare Tunnel（需要 FastAPI 先运行）

echo "🌐 启动 Cloudflare Tunnel..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""
echo "⚠️  重要提示："
echo "   1. 请确保 FastAPI 应用已在运行（python web_app.py）"
echo "   2. 保持此终端窗口打开，关闭后链接会失效"
echo "   3. 复制下面显示的链接即可分享"
echo ""
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

cloudflared tunnel --url http://127.0.0.1:8000
