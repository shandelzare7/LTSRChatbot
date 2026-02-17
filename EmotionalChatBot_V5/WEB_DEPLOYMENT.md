# FastAPI + Cloudflare Web 部署指南

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置环境变量

确保 `.env` 文件中包含：

```bash
DATABASE_URL=postgresql+asyncpg://user:password@localhost/dbname
OPENAI_API_KEY=your_openai_api_key
ENVIRONMENT=development  # 或 production
PORT=8000  # 可选，默认8000
# Web Push（可选：关掉网页也能推送）
# 生成 key：python devtools/generate_vapid_keys.py
VAPID_PUBLIC_KEY=...
VAPID_PRIVATE_KEY=...
VAPID_SUBJECT=mailto:you@example.com
```

### 3. 启动 Web 应用

```bash
python web_app.py
```

或者使用 uvicorn：

```bash
uvicorn web_app:app --host 0.0.0.0 --port 8000
```

### 4. 访问应用

打开浏览器访问：`http://localhost:8000`

## Cloudflare 部署配置

### 方案 A：FastAPI 服务器 + Cloudflare CDN（推荐）

#### 1. 服务器部署

在服务器上运行 FastAPI 应用：

```bash
# 使用 systemd 或 supervisor 管理进程
uvicorn web_app:app --host 0.0.0.0 --port 8000 --workers 4
```

#### 2. Cloudflare DNS 配置

1. 登录 Cloudflare 控制台
2. 添加你的域名
3. 添加 A 记录：
   - 名称：`@` 或 `www`
   - IPv4 地址：你的服务器 IP
   - 代理状态：已代理（橙色云朵）

#### 3. Cloudflare SSL/TLS 设置

1. 进入 SSL/TLS 设置
2. 加密模式选择：**完全（严格）**
3. 自动 HTTPS 重定向：开启

#### 4. Cloudflare 缓存规则

在 **规则 > 页面规则** 中设置：

- **静态资源缓存**：
  - URL：`yourdomain.com/static/*`
  - 设置：缓存级别 = 标准，边缘缓存 TTL = 1 个月

- **API 不缓存**：
  - URL：`yourdomain.com/api/*`
  - 设置：缓存级别 = 绕过

#### 5. 安全设置

在 **安全 > WAF** 中：
- 启用基本防火墙规则
- 设置速率限制（防止滥用）

### 方案 B：Cloudflare Workers（高级）

如果需要使用 Cloudflare Workers，需要将 FastAPI 逻辑转换为 Workers 格式。这需要：

1. 使用 `@cloudflare/workers` 或 `wrangler`
2. 将数据库连接改为通过外部 API 或使用 Cloudflare D1
3. 注意 Workers 的 CPU 时间限制（10-50ms）

## 生产环境优化

### 1. 会话存储

当前使用内存存储会话，生产环境建议使用 Redis：

```python
# 修改 app/web/session.py
import redis
redis_client = redis.Redis(host='localhost', port=6379, db=0)

def create_session(user_id: str, bot_id: str) -> str:
    session_id = generate_session_id()
    session_data = {
        "user_id": user_id,
        "bot_id": bot_id,
        "created_at": datetime.now().isoformat(),
        "last_active": datetime.now().isoformat(),
    }
    redis_client.setex(
        f"session:{session_id}",
        86400 * 7,  # 7天过期
        json.dumps(session_data)
    )
    return session_id
```

### 2. 环境变量

生产环境设置：

```bash
ENVIRONMENT=production
DATABASE_URL=postgresql+asyncpg://...
OPENAI_API_KEY=...
PORT=8000
```

### 3. 日志和监控

- 使用 `logging` 模块记录请求日志
- 集成 Sentry 或类似服务进行错误追踪
- 监控 API 响应时间和错误率

### 4. 性能优化

- 使用连接池管理数据库连接
- 启用 gzip 压缩
- 使用 CDN 加速静态资源
- 考虑使用 WebSocket 实现实时通信

## API 端点说明

### GET `/`
主入口，根据 Cookie 返回 bot 选择页面或聊天界面

### GET `/api/bots`
获取所有可用的 bot 列表

### POST `/api/session/init`
初始化会话，选择 bot
- Body: `{"bot_id": "bot_uuid"}`
- 返回：设置 Cookie `session_id`

### POST `/api/chat`
发送消息并获取回复
- Body: `{"message": "用户消息"}`
- 需要 Cookie: `session_id`

### POST `/api/session/reset`
重置会话，清空对话历史
- 需要 Cookie: `session_id`

### GET `/api/session/status`
获取当前会话状态
- 需要 Cookie: `session_id`

## 故障排查

### 1. Cookie 未设置

- 检查 `ENVIRONMENT` 环境变量
- 确保使用 HTTPS（生产环境）
- 检查浏览器 Cookie 设置

### 2. 数据库连接失败

- 检查 `DATABASE_URL` 是否正确
- 确保数据库可从服务器访问
- 检查防火墙设置

### 3. LLM 调用失败

- 检查 `OPENAI_API_KEY` 是否有效
- 检查 API 配额和限制
- 查看服务器日志

## 安全建议

1. **HTTPS 强制**：生产环境必须使用 HTTPS
2. **Cookie 安全**：启用 `Secure` 和 `HttpOnly`
3. **CORS 限制**：生产环境限制允许的域名
4. **速率限制**：防止 API 滥用
5. **输入验证**：验证所有用户输入
6. **错误处理**：不要暴露敏感错误信息

## 测试

### 本地测试

```bash
# 启动应用
python web_app.py

# 在浏览器访问
http://localhost:8000
```

### API 测试

```bash
# 获取 bot 列表
curl http://localhost:8000/api/bots

# 初始化会话
curl -X POST http://localhost:8000/api/session/init \
  -H "Content-Type: application/json" \
  -d '{"bot_id": "your_bot_id"}' \
  -c cookies.txt

# 发送消息
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "你好"}' \
  -b cookies.txt
```

## 常见问题

**Q: 如何支持多个域名？**
A: 在 CORS 配置中添加多个域名，或使用 Cloudflare 的域名转发功能。

**Q: 会话过期时间可以调整吗？**
A: 可以，修改 `app/web/session.py` 中的 `timedelta(days=7)`。

**Q: 如何添加用户认证？**
A: 可以在 `init_session` 中添加用户登录逻辑，或集成 OAuth。

**Q: 支持 WebSocket 实时通信吗？**
A: 当前使用轮询，可以升级为 WebSocket 实现实时通信。
