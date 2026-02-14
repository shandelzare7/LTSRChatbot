# Cloudflare Tunnel 临时链接部署指南

## 🚀 快速开始（最简单方式）

### 方法 1：使用 cloudflared（推荐）

#### 1. 安装 cloudflared

**macOS:**
```bash
brew install cloudflared
```

**Linux:**
```bash
# 下载最新版本
wget https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-linux-amd64
chmod +x cloudflared-linux-amd64
sudo mv cloudflared-linux-amd64 /usr/local/bin/cloudflared
```

**Windows:**
下载：https://github.com/cloudflare/cloudflared/releases/latest/download/cloudflared-windows-amd64.exe

#### 2. 启动 FastAPI 应用

```bash
# 在项目目录下
python web_app.py
# 或者
uvicorn web_app:app --host 127.0.0.1 --port 8000
```

#### 3. 创建临时隧道

在另一个终端运行：

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

你会看到类似输出：
```
+--------------------------------------------------------------------------------------------+
|  Your quick Tunnel has been created! Visit it at (it may take some time to be reachable): |
|  https://xxxx-xxxx-xxxx.trycloudflare.com                                                 |
+--------------------------------------------------------------------------------------------+
```

#### 4. 访问链接

复制显示的链接（如 `https://xxxx-xxxx-xxxx.trycloudflare.com`），这就是你的临时分享链接！

### 方法 2：使用一键脚本

创建一个启动脚本：

```bash
#!/bin/bash
# start_tunnel.sh

# 启动 FastAPI（后台）
python web_app.py &
FASTAPI_PID=$!

# 等待 FastAPI 启动
sleep 3

# 启动 Cloudflare Tunnel
echo "启动 Cloudflare Tunnel..."
cloudflared tunnel --url http://127.0.0.1:8000

# 清理
trap "kill $FASTAPI_PID" EXIT
```

## 📝 详细步骤

### 步骤 1：确保 FastAPI 运行在本地

```bash
cd /Users/huangshenze/Downloads/LTSRChatbot/EmotionalChatBot_V5
python web_app.py
```

应用会在 `http://127.0.0.1:8000` 启动

### 步骤 2：安装 cloudflared

检查是否已安装：
```bash
cloudflared --version
```

如果未安装，按上面的方法安装。

### 步骤 3：创建隧道

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

### 步骤 4：获取分享链接

复制终端中显示的链接，格式类似：
```
https://random-string-here.trycloudflare.com
```

### 步骤 5：更新分享链接脚本

运行脚本生成正确的分享链接：

```bash
# 设置临时域名（从 cloudflared 输出中获取）
export WEB_DOMAIN="xxxx-xxxx-xxxx.trycloudflare.com"
python generate_share_links.py
```

## 🔧 高级配置

### 使用命名隧道（持久化）

如果你想创建一个持久的隧道（即使重启也保持相同域名）：

```bash
# 1. 登录 Cloudflare（需要账户）
cloudflared tunnel login

# 2. 创建命名隧道
cloudflared tunnel create chatbot-tunnel

# 3. 创建配置文件
mkdir -p ~/.cloudflared
cat > ~/.cloudflared/config.yml << EOF
tunnel: chatbot-tunnel
credentials-file: ~/.cloudflared/chatbot-tunnel.json

ingress:
  - hostname: chatbot.yourdomain.com
    service: http://127.0.0.1:8000
  - service: http_status:404
EOF

# 4. 运行隧道
cloudflared tunnel run chatbot-tunnel
```

### 后台运行

使用 `nohup` 或 `screen`/`tmux`：

```bash
# 使用 nohup
nohup cloudflared tunnel --url http://127.0.0.1:8000 > tunnel.log 2>&1 &

# 或使用 screen
screen -S tunnel
cloudflared tunnel --url http://127.0.0.1:8000
# 按 Ctrl+A 然后 D 分离会话
```

## ⚠️ 注意事项

1. **临时链接限制**
   - 免费临时链接会在一定时间后过期（通常几小时）
   - 每次重启 cloudflared 会生成新链接
   - 适合测试和临时分享

2. **安全性**
   - 临时链接是公开的，任何人都可以访问
   - 不要在生产环境使用
   - 适合测试和演示

3. **性能**
   - 通过 Cloudflare 的全球网络
   - 自动 HTTPS
   - 免费版本有速率限制

## 🎯 快速命令参考

```bash
# 启动应用
python web_app.py

# 创建临时隧道（新终端）
cloudflared tunnel --url http://127.0.0.1:8000

# 生成分享链接
WEB_DOMAIN="your-tunnel-url.trycloudflare.com" python generate_share_links.py
```

## 📱 分享链接示例

临时链接格式：
```
https://xxxx-xxxx-xxxx.trycloudflare.com/chat/{bot_id}
```

例如：
- `https://xxxx-xxxx-xxxx.trycloudflare.com/chat/4d803b5a-cb30-4d14-89eb-88d259564610`

## 🔄 更新分享链接

每次重启 cloudflared 后，运行：

```bash
# 1. 获取新的隧道URL（从 cloudflared 输出）
# 2. 设置环境变量
export WEB_DOMAIN="新的隧道URL"
# 3. 生成链接
python generate_share_links.py
```
