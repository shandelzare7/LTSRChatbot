# Deploy to Render

This repo includes a Render Blueprint at `render.yaml`.

## 1) Push to GitHub

Render deploys from a Git repository. Make sure you **do not commit** `.env` (already ignored by `.gitignore`).

## 2) Create via Blueprint

In Render dashboard:

- New → **Blueprint**
- Select your GitHub repo
- Apply

This provisions:
- `ltsrchatbot-web` (Python web service)
- `ltsrchatbot-db` (PostgreSQL)

## 3) Set required env vars (Web service)

In `ltsrchatbot-web` → Environment:

- `OPENAI_API_KEY`
- `OPENAI_BASE_URL` (optional, for OpenAI-compatible providers)
- `OPENAI_MODEL`

`DATABASE_URL` is injected automatically from the Render Postgres service.

## Provider examples

### Qwen (DashScope OpenAI-compatible)

- `OPENAI_BASE_URL=https://dashscope.aliyuncs.com/compatible-mode/v1`
- `OPENAI_MODEL=qwen-plus`
- `OPENAI_API_KEY=<your_dashscope_key>`

### DeepSeek (OpenAI-compatible)

- `OPENAI_BASE_URL=https://api.deepseek.com/v1`
- `OPENAI_MODEL=deepseek-chat`
- `OPENAI_API_KEY=<your_deepseek_key>`

## Notes

- Startup runs `EmotionalChatBot_V5/devtools/ensure_schema.py` to create tables and ensure a default bot exists.
- Session storage is in-memory; keep the web service to **1 instance** unless you add shared session storage (e.g. Redis).

