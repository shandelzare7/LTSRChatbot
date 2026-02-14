#!/bin/bash
echo "📊 Cloudflare Tunnel 状态检查"
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"

# 检查 FastAPI
if lsof -ti:8000 > /dev/null 2>&1; then
    echo "✅ FastAPI: 正在运行 (端口 8000)"
else
    echo "❌ FastAPI: 未运行"
fi

# 检查 cloudflared
TUNNEL_PID=$(ps aux | grep "cloudflared tunnel" | grep -v grep | awk '{print $2}' | head -1)
if [ ! -z "$TUNNEL_PID" ]; then
    echo "✅ Cloudflare Tunnel: 正在运行 (PID: $TUNNEL_PID)"
    LINK=$(grep -oE "https://[a-z0-9-]+\.trycloudflare\.com" /tmp/cloudflare_tunnel.log 2>/dev/null | tail -1)
    if [ ! -z "$LINK" ]; then
        echo "🔗 当前链接: $LINK"
    else
        echo "⚠️  无法从日志中提取链接"
    fi
else
    echo "❌ Cloudflare Tunnel: 未运行"
fi

echo ""
