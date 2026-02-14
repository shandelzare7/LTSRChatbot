# Web 应用快速启动指南

## 🚀 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

确保 `.env` 文件包含必要的配置：

```bash
DATABASE_URL=postgresql+asyncpg://user:password@localhost/dbname
OPENAI_API_KEY=your_openai_api_key
```

### 3. 启动 Web 应用

```bash
python web_app.py
```

应用将在 `http://localhost:8000` 启动

### 4. 访问应用

打开浏览器访问：`http://localhost:8000`

## 📋 功能说明

### 主要功能

1. **Bot 选择页面**
   - 首次访问时显示所有可用的 Chatbot
   - 点击 Bot 卡片开始对话

2. **聊天界面**
   - 实时对话
   - 消息历史记录
   - 重置会话功能

3. **会话管理**
   - 自动通过 Cookie 管理会话
   - 会话有效期：7天
   - 支持多用户同时使用

## 🔗 分享链接

### 方式 1：通用链接（推荐）

```
https://yourdomain.com/
```

用户访问后会：
1. 如果没有会话 → 显示 Bot 选择页面
2. 如果有会话 → 直接进入聊天界面

### 方式 2：直接链接到特定 Bot

可以修改代码支持 URL 参数：

```python
# 在 web_app.py 中添加
@app.get("/chat/{bot_id}", response_class=HTMLResponse)
async def chat_with_bot(bot_id: str, request: Request):
    # 自动选择该 bot 并初始化会话
    ...
```

## 🌐 Cloudflare 部署

### 步骤 1：服务器部署

在服务器上运行：

```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000 --workers 4
```

### 步骤 2：Cloudflare 配置

1. **DNS 设置**
   - 添加 A 记录指向服务器 IP
   - 启用代理（橙色云朵）

2. **SSL/TLS**
   - 加密模式：完全（严格）
   - 自动 HTTPS 重定向：开启

3. **缓存规则**
   - `/static/*` → 缓存
   - `/api/*` → 不缓存

详细部署说明请参考 `WEB_DEPLOYMENT.md`

## 🐛 故障排查

### Cookie 未设置

- 检查浏览器是否允许 Cookie
- 确保使用 HTTPS（生产环境）
- 检查 `ENVIRONMENT` 环境变量

### 数据库连接失败

- 检查 `DATABASE_URL` 是否正确
- 确保数据库服务正在运行
- 检查网络连接

### Bot 列表为空

- 确保数据库中有 Bot 数据
- 运行 seed 脚本创建示例 Bot
- 检查数据库连接

## 📝 API 端点

- `GET /` - 主入口页面
- `GET /api/bots` - 获取 Bot 列表
- `POST /api/session/init` - 初始化会话
- `POST /api/chat` - 发送消息
- `POST /api/session/reset` - 重置会话
- `GET /api/session/status` - 获取会话状态

## 🔒 安全建议

1. **生产环境**
   - 设置 `ENVIRONMENT=production`
   - 限制 CORS 允许的域名
   - 启用 HTTPS
   - 设置速率限制

2. **会话存储**
   - 当前使用内存存储（适合单机）
   - 多服务器部署建议使用 Redis

3. **错误处理**
   - 不要暴露敏感错误信息
   - 记录详细的错误日志

## 📚 更多信息

- 详细部署指南：`WEB_DEPLOYMENT.md`
- API 文档：访问 `http://localhost:8000/docs`（FastAPI 自动生成）
