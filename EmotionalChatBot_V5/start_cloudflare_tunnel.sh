#!/bin/bash
# Cloudflare Tunnel 一键启动脚本

echo "🚀 启动 FastAPI + Cloudflare Tunnel..."

# 检查 cloudflared 是否安装
if ! command -v cloudflared &> /dev/null; then
    echo "❌ cloudflared 未安装"
    echo "请先安装: brew install cloudflared (macOS) 或访问 https://github.com/cloudflare/cloudflared/releases"
    exit 1
fi

# 检查 Python 环境
if ! command -v python &> /dev/null; then
    echo "❌ Python 未找到"
    exit 1
fi

# 启动 FastAPI（后台）
echo "📦 启动 FastAPI 应用..."
python web_app.py > /dev/null 2>&1 &
FASTAPI_PID=$!

# 等待 FastAPI 启动
echo "⏳ 等待 FastAPI 启动..."
sleep 5

# 检查 FastAPI 是否运行
if ! ps -p $FASTAPI_PID > /dev/null; then
    echo "❌ FastAPI 启动失败"
    exit 1
fi

echo "✅ FastAPI 已启动 (PID: $FASTAPI_PID)"
echo ""
echo "🌐 启动 Cloudflare Tunnel..."
echo "━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━"
echo ""

# 启动 Cloudflare Tunnel（前台，显示链接）
cloudflared tunnel --url http://127.0.0.1:8000

# 清理：当 cloudflared 退出时，也停止 FastAPI
echo ""
echo "🛑 正在关闭..."
kill $FASTAPI_PID 2>/dev/null
echo "✅ 已关闭"
