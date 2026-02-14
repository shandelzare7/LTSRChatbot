# 🔄 重启 Web 应用以启用日志功能

## ✅ 日志功能已添加

我已经为 Web 应用添加了完整的日志记录功能。现在需要重启 FastAPI 服务才能生效。

## 📋 重启步骤

### 1. 停止当前运行的 FastAPI

```bash
# 查找 FastAPI 进程
ps aux | grep "python web_app.py" | grep -v grep

# 停止进程（替换 PID 为实际进程号）
kill <PID>

# 或者停止所有相关进程
pkill -f "python web_app.py"
```

### 2. 停止 Cloudflare Tunnel（如果正在运行）

```bash
pkill cloudflared
```

### 3. 重新启动服务

**方式 1：使用一键脚本（推荐）**
```bash
./start_cloudflare_tunnel.sh
```

**方式 2：手动启动**
```bash
# 终端1：启动 FastAPI
python web_app.py

# 终端2：启动 Cloudflare Tunnel
cloudflared tunnel --url http://127.0.0.1:8000
```

## 📝 日志文件位置

重启后，所有 Web 对话日志将保存在：
```
logs/web_chat_{时间戳}_{session_id}.log
```

例如：
- `logs/web_chat_2026-02-11_20-15-30_a1b2c3d4.log`

## ✅ 验证日志功能

重启后，通过分享链接发送一条消息，然后检查：

```bash
# 查看最新的 Web 日志文件
ls -lt logs/web_chat_*.log | head -1

# 查看日志内容
tail -50 logs/web_chat_*.log
```

日志文件应该包含：
- 会话信息（用户ID、Bot ID等）
- 用户消息
- Graph 运行过程的详细日志
- Bot 回复

## 🎯 现在可以：

1. ✅ 重启 FastAPI 服务
2. ✅ 通过分享链接进行对话
3. ✅ 在 `logs/` 目录查看完整的对话日志
