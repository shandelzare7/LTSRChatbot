# 🚀 快速部署到 Cloudflare 临时链接

## 最简单方式（一键启动）

### 1. 安装 cloudflared

**macOS:**
```bash
brew install cloudflared
```

**其他系统:** 访问 https://github.com/cloudflare/cloudflared/releases

### 2. 运行一键脚本

```bash
./start_cloudflare_tunnel.sh
```

脚本会自动：
1. 启动 FastAPI 应用
2. 创建 Cloudflare Tunnel
3. 显示临时链接

### 3. 复制分享链接

脚本运行后会显示类似：
```
https://xxxx-xxxx-xxxx.trycloudflare.com
```

这就是你的临时分享链接！

## 手动方式

### 步骤 1：启动 FastAPI

```bash
python web_app.py
```

### 步骤 2：创建隧道（新终端）

```bash
cloudflared tunnel --url http://127.0.0.1:8000
```

### 步骤 3：获取链接

复制终端显示的链接，例如：
```
https://random-string.trycloudflare.com
```

### 步骤 4：生成分享链接

```bash
WEB_DOMAIN="你的隧道链接" python generate_share_links.py
```

## 📱 分享链接格式

```
https://你的隧道链接.trycloudflare.com/chat/{bot_id}
```

例如：
- `https://xxxx-xxxx-xxxx.trycloudflare.com/chat/4d803b5a-cb30-4d14-89eb-88d259564610`

## ⚠️ 注意事项

1. **临时链接**：每次重启会生成新链接
2. **免费使用**：适合测试和临时分享
3. **公开访问**：任何人都可以通过链接访问
4. **保持运行**：关闭终端后链接会失效

## 🔄 更新分享链接

每次重启后，运行：
```bash
WEB_DOMAIN="新的隧道链接" python generate_share_links.py
```
