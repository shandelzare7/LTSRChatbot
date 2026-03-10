"""
FastAPI Web Application for EmotionalChatBot V5.0
支持通过 Web 界面与 Chatbot 对话
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timezone, timedelta
import contextvars
import logging
import time
import asyncio
import io
import json
import html
import zipfile
from typing import Any, Optional
import uuid

# 加载 .env（若存在）
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
try:
    from utils.env_loader import load_project_env
    load_project_env(root)
except Exception:
    pass

from fastapi import FastAPI, Request, Response, Cookie, HTTPException
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse, FileResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.graph import build_graph
from app.core import (
    DBManager,
    Bot,
    User,
    Message,
    WebChatLog,
    generate_bot_profile,
    get_random_relationship_template,
)
from app.services.llm import LLMAPIError
from app.web.session import (
    create_session,
    get_session,
    delete_session,
    set_session_visit_source,
    generate_user_id_from_request,
)
from main import _make_initial_state
from sqlalchemy import select, case
from sqlalchemy import text
from utils.yaml_loader import get_project_root
import sys

# 初始化 FastAPI 应用
app = FastAPI(title="EmotionalChatBot Web", version="5.0")

# 日志文件管理
_log_files: dict[str, tuple] = {}  # {session_id: (file_handle, path)}

# Web 并发输入控制：同一 session 允许“中断上一轮结算并合并消息”
# key=session_id -> state
_inflight_chat: dict[str, dict] = {}

# Persistent cookies (so history survives Render restarts / IP changes)
_COOKIE_WEB_USER_ID = "web_user_id"
_COOKIE_BOT_ID = "bot_id"
_COOKIE_VISIT_SOURCE = "visit_source"


def _get_user_bot_from_session_or_cookies(
    *,
    session_id: Optional[str],
    web_user_id: Optional[str],
    bot_id_cookie: Optional[str],
) -> Optional[tuple[str, str]]:
    """
    Resolve (user_external_id, bot_id) from in-memory session if present,
    otherwise fall back to persistent cookies.
    """
    if session_id:
        sess = get_session(session_id)
        if isinstance(sess, dict) and sess.get("user_id") and sess.get("bot_id"):
            return str(sess["user_id"]), str(sess["bot_id"])
    if web_user_id and bot_id_cookie:
        return str(web_user_id), str(bot_id_cookie)
    return None


def _set_visit_source_cookie_if_present(response: Response, request: Request) -> None:
    """若 URL 带 ?source=xxx，写入 cookie 供后续创建 User 时写入 DB 追踪。path=/ 确保所有请求都会带上。"""
    source = (request.query_params.get("source") or "").strip()
    if not source:
        return
    secure = os.getenv("ENVIRONMENT") == "production"
    response.set_cookie(
        key=_COOKIE_VISIT_SOURCE,
        value=source,
        path="/",
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=86400 * 7,
    )


def _set_persistent_session_cookies(response: Response, *, user_id: str, bot_id: str, session_id: str) -> None:
    secure = os.getenv("ENVIRONMENT") == "production"
    # Primary session pointer (in-memory map)
    response.set_cookie(
        key="session_id",
        value=session_id,
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=86400 * 7,
    )
    # Persistent identity + bot selection (survive server restarts/IP changes)
    response.set_cookie(
        key=_COOKIE_WEB_USER_ID,
        value=str(user_id),
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=86400 * 30,
    )
    response.set_cookie(
        key=_COOKIE_BOT_ID,
        value=str(bot_id),
        httponly=True,
        secure=secure,
        samesite="lax",
        max_age=86400 * 30,
    )


def _get_inflight_state(session_id: str) -> dict:
    st = _inflight_chat.get(session_id)
    if isinstance(st, dict):
        return st
    st = {
        "lock": asyncio.Lock(),
        "pending_user_msgs": [],  # List[{"content": str, "ts": str}]
        "latest_req_id": None,
        "waiter": None,  # asyncio.Future
        "fast_task": None,  # asyncio.Task
        "tail_task": None,  # asyncio.Task
    }
    _inflight_chat[session_id] = st
    return st


def get_or_create_log_file(session_id: str, user_id: str, bot_id: str):
    """获取或创建会话日志文件"""
    if session_id in _log_files:
        return _log_files[session_id]
    
    proot = get_project_root()
    log_dir = proot / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    # 使用 session_id 前8位作为文件名的一部分
    session_short = session_id[:8] if len(session_id) >= 8 else session_id
    path = log_dir / f"web_chat_{ts}_{session_short}.log"
    f = open(path, "w", encoding="utf-8")
    
    # 写入会话信息
    f.write("=" * 80 + "\n")
    f.write("EmotionalChatBot V5.0 Web 对话日志\n")
    f.write("=" * 80 + "\n")
    f.write(f"会话ID: {session_id}\n")
    f.write(f"用户ID: {user_id}\n")
    f.write(f"Bot ID: {bot_id}\n")
    f.write(f"开始时间: {datetime.now().isoformat()}\n")
    f.write("=" * 80 + "\n\n")
    f.flush()
    
    _log_files[session_id] = (f, path)
    return f, path


class FileOnlyWriter:
    """仅写入日志文件，不输出到控制台。用于 graph 内部节点 log"""
    def __init__(self, file_handle):
        self._file = file_handle

    def write(self, s: str):
        if self._file:
            try:
                self._file.write(s)
                self._file.flush()
            except OSError:
                pass

    def flush(self):
        if self._file:
            try:
                self._file.flush()
            except OSError:
                pass


# 当前请求对应的会话 log 文件（仅 web 轮次内有效），用于把 logging 输出也写入同一文件并落入 DB
_current_web_log_file: contextvars.ContextVar[Optional[Any]] = contextvars.ContextVar("web_log_file", default=None)


class WebChatLogHandler(logging.Handler):
    """将 logger 输出写入当前请求的 web_chat log 文件，便于与 print 一起落入 Render 的 web_chat_logs 表。"""

    def emit(self, record: logging.LogRecord) -> None:
        f = _current_web_log_file.get()
        if f is None:
            return
        try:
            msg = self.format(record)
            f.write(f"[LOG] {msg}\n")
            f.flush()
        except OSError:
            pass


# 模块加载时挂到 root logger，emit 内通过 contextvar 取文件，无文件时跳过
_web_chat_log_handler = WebChatLogHandler()
_web_chat_log_handler.setLevel(logging.DEBUG)
_web_chat_log_handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
logging.getLogger().addHandler(_web_chat_log_handler)


def log_web_chat(session_id: str, user_id: str, bot_id: str, user_message: str, bot_reply: str):
    """记录 Web 对话到日志文件"""
    try:
        log_file, log_path = get_or_create_log_file(session_id, user_id, bot_id)
        now_iso = datetime.now(timezone.utc).isoformat(timespec="seconds")
        
        log_file.write(f"\n[{now_iso}] === 用户: {user_message}\n")
        log_file.write("-" * 80 + "\n")
        log_file.write(f"[{now_iso}] === Bot: {bot_reply}\n")
        log_file.write("=" * 80 + "\n\n")
        log_file.flush()
    except Exception as e:
        print(f"日志记录失败: {e}", file=sys.stderr)


def _get_log_dir() -> Path:
    proot = get_project_root()
    log_dir = proot / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir


def _require_admin(request: Request) -> None:
    """
    Minimal admin auth for log download endpoint.
    Set env var `ADMIN_TOKEN` and pass request header `X-Admin-Token`.
    """
    token = (os.getenv("ADMIN_TOKEN") or "").strip()
    if not token:
        # Hide the endpoint when not configured
        raise HTTPException(status_code=404, detail="Not found")
    got = (request.headers.get("x-admin-token") or "").strip()
    if got != token:
        raise HTTPException(status_code=403, detail="Forbidden")


@app.get("/api/admin/web_chat_logs_latest.zip")
async def admin_download_latest_web_chat_logs_zip(request: Request, n: int = 2):
    """
    Download latest N web chat logs as a zip file.
    Prefer DB snapshots (persistent). Fall back to local filesystem.
    """
    _require_admin(request)
    n = max(1, min(int(n or 2), 10))

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
        # 1) DB snapshots
        try:
            db = get_db_manager()
            async with db.Session() as s:
                result = await s.execute(select(WebChatLog).order_by(WebChatLog.updated_at.desc()).limit(n))
                rows = list(result.scalars().all())
            if rows:
                for r in rows:
                    name = r.filename or f"web_chat_{(r.updated_at or r.created_at).isoformat()}.log"
                    z.writestr(str(name), str(r.content or ""))
            else:
                raise RuntimeError("no_db_rows")
        except Exception:
            # 2) filesystem fallback (ephemeral)
            log_dir = _get_log_dir()
            files = sorted(log_dir.glob("web_chat_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)[:n]
            if not files:
                raise HTTPException(status_code=404, detail="No logs found")
            for p in files:
                z.writestr(p.name, p.read_text(encoding="utf-8", errors="replace"))
    mem.seek(0)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_name = f"web_chat_latest_{ts}.zip"
    return StreamingResponse(
        mem,
        media_type="application/zip",
        headers={"Content-Disposition": f'attachment; filename="{out_name}"'},
    )

# CORS 配置（支持 Cloudflare 域名）
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # 生产环境应限制为具体域名
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# 静态文件服务
static_dir = root / "static"
static_dir.mkdir(exist_ok=True)
app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")


@app.get("/sw.js")
async def service_worker():
    """
    Service Worker must be served from the app root to control '/' scope.
    Used for browser notifications (best-effort; requires HTTPS/localhost).
    """
    p = static_dir / "sw.js"
    if not p.exists():
        raise HTTPException(status_code=404, detail="sw.js not found")
    return FileResponse(
        path=str(p),
        media_type="application/javascript",
        headers={"Cache-Control": "no-cache, no-store, must-revalidate"},
    )


# Pydantic 模型
class ChatRequest(BaseModel):
    message: str


class SessionInitRequest(BaseModel):
    bot_id: str
    source: Optional[str] = None  # 可选，与 ?source= 一致，便于前端从当前 URL 传入


# Push subscription payload
class SessionResumeRequest(BaseModel):
    user_db_id: str


class PushSubscribeRequest(BaseModel):
    subscription: dict


# 全局变量
_graph = None
_graph_fast = None
_graph_tail = None
_db_manager = None


def get_graph():
    """懒加载 graph"""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def _truthy(v: Optional[str]) -> bool:
    return str(v or "").strip().lower() in ("1", "true", "yes", "y", "on")


def get_graph_fast():
    """Web fast-return graph: end at processor (no tail writes)."""
    global _graph_fast
    if _graph_fast is None:
        _graph_fast = build_graph(entry_point="loader", end_at="processor")
    return _graph_fast


def get_graph_tail():
    """Web tail graph: evolver -> ... -> memory_writer."""
    global _graph_tail
    if _graph_tail is None:
        _graph_tail = build_graph(entry_point="evolver", end_at="memory_writer")
    return _graph_tail


def get_db_manager():
    """懒加载 DBManager"""
    global _db_manager
    if _db_manager is None:
        if not os.getenv("DATABASE_URL"):
            raise RuntimeError("DATABASE_URL 未设置")
        _db_manager = DBManager.from_env()
    return _db_manager


def _get_vapid_keys() -> tuple[str, str, str]:
    """
    Web Push VAPID keys.
    - VAPID_PUBLIC_KEY: base64url string (no padding)
    - VAPID_PRIVATE_KEY: base64url string
    - VAPID_SUBJECT: e.g. 'mailto:you@example.com' or 'https://your-domain'
    """
    pub = (os.getenv("VAPID_PUBLIC_KEY") or "").strip()
    priv = (os.getenv("VAPID_PRIVATE_KEY") or "").strip()
    subject = (os.getenv("VAPID_SUBJECT") or "").strip() or "mailto:admin@example.com"
    if not pub or not priv:
        raise RuntimeError("VAPID keys not configured (VAPID_PUBLIC_KEY/VAPID_PRIVATE_KEY)")
    return pub, priv, subject


async def _ensure_push_schema(db: DBManager) -> None:
    """
    Ensure push_subscriptions table exists (idempotent).
    We key by session_id (cookie) because sessions are per-browser.
    """
    create_table = """
    CREATE TABLE IF NOT EXISTS push_subscriptions (
      session_id TEXT PRIMARY KEY,
      user_external_id TEXT,
      bot_id UUID,
      subscription JSONB NOT NULL,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW()
    )"""
    create_index = """
    CREATE INDEX IF NOT EXISTS idx_push_subscriptions_user_bot ON push_subscriptions(user_external_id, bot_id)"""
    async with db.engine.connect() as conn:
        await conn.execute(text(create_table))
        await conn.execute(text(create_index))
        await conn.commit()


async def _upsert_push_subscription(
    db: DBManager, *, session_id: str, user_external_id: str, bot_id: str, subscription: dict
) -> None:
    import json

    await _ensure_push_schema(db)
    sql = """
    INSERT INTO push_subscriptions(session_id, user_external_id, bot_id, subscription, updated_at)
    VALUES (:session_id, :user_external_id, CAST(:bot_id AS uuid), CAST(:sub AS jsonb), NOW())
    ON CONFLICT (session_id) DO UPDATE
      SET user_external_id=EXCLUDED.user_external_id,
          bot_id=EXCLUDED.bot_id,
          subscription=EXCLUDED.subscription,
          updated_at=NOW();
    """
    async with db.engine.connect() as conn:
        await conn.execute(
            text(sql),
            {
                "session_id": session_id,
                "user_external_id": user_external_id,
                "bot_id": bot_id,
                "sub": json.dumps(subscription, ensure_ascii=False),
            },
        )
        await conn.commit()


async def _delete_push_subscription(
    db: DBManager,
    *,
    session_id: str,
    user_external_id: Optional[str] = None,
    bot_id: Optional[str] = None,
) -> None:
    """按 session_id 删除；若同时提供 user_external_id 与 bot_id，则删除该 user+bot 下所有行（含其他 session），避免重启后残留。"""
    await _ensure_push_schema(db)
    async with db.engine.connect() as conn:
        await conn.execute(text("DELETE FROM push_subscriptions WHERE session_id=:sid"), {"sid": session_id})
        if user_external_id and bot_id:
            await conn.execute(
                text("DELETE FROM push_subscriptions WHERE user_external_id=:uid AND bot_id=CAST(:bid AS uuid)"),
                {"uid": user_external_id, "bid": str(bot_id)},
            )
        await conn.commit()


def _parse_subscription_row(sub: Any) -> Optional[dict]:
    """JSONB 可能被驱动返回为 dict 或 str，统一为 dict。"""
    if isinstance(sub, dict):
        return sub
    if isinstance(sub, str):
        try:
            return json.loads(sub)
        except Exception:
            return None
    return None


async def _get_push_subscription(
    db: DBManager,
    *,
    session_id: Optional[str] = None,
    user_external_id: Optional[str] = None,
    bot_id: Optional[str] = None,
) -> Optional[dict]:
    """
    查推送订阅：优先用 session_id；若未提供或未命中则用 (user_external_id, bot_id) 取最新一条。
    这样 Render 重启后 session 变了仍能发推送。
    """
    await _ensure_push_schema(db)
    async with db.engine.connect() as conn:
        if session_id:
            row = (await conn.execute(text("SELECT subscription FROM push_subscriptions WHERE session_id=:sid"), {"sid": session_id})).first()
            if row:
                parsed = _parse_subscription_row(row[0])
                if parsed:
                    return parsed
        if user_external_id and bot_id:
            row = (
                await conn.execute(
                    text("SELECT subscription FROM push_subscriptions WHERE user_external_id=:uid AND bot_id=CAST(:bid AS uuid) ORDER BY updated_at DESC LIMIT 1"),
                    {"uid": user_external_id, "bid": str(bot_id)},
                )
            ).first()
            if row:
                parsed = _parse_subscription_row(row[0])
                if parsed:
                    return parsed
    return None


async def _send_web_push(*, subscription: dict, title: str, body: str, url: str = "/", tag: str = "ltsr-push") -> None:
    """
    Best-effort send Web Push.
    Uses pywebpush (sync) in a thread executor.
    """
    import json
    from pywebpush import webpush  # type: ignore

    pub, priv, subject = _get_vapid_keys()
    payload = json.dumps({"title": title, "body": body, "url": url, "tag": tag}, ensure_ascii=False)

    def _do():
        # webpush() will raise on network/provider errors; we treat as best-effort.
        webpush(
            subscription_info=subscription,
            data=payload,
            vapid_private_key=priv,
            vapid_claims={"sub": subject},
        )

    loop = asyncio.get_running_loop()
    await loop.run_in_executor(None, _do)


@app.get("/api/push/public-key")
async def get_push_public_key():
    pub, _, _ = _get_vapid_keys()
    return {"public_key": pub}


@app.post("/api/push/subscribe")
async def push_subscribe(
    req: PushSubscribeRequest,
    session_id: Optional[str] = Cookie(None),
    web_user_id: Optional[str] = Cookie(None, alias=_COOKIE_WEB_USER_ID),
    bot_id_cookie: Optional[str] = Cookie(None, alias=_COOKIE_BOT_ID),
):
    resolved = _get_user_bot_from_session_or_cookies(
        session_id=session_id, web_user_id=web_user_id, bot_id_cookie=bot_id_cookie
    )
    if not resolved or not session_id:
        raise HTTPException(status_code=401, detail="未找到会话")
    user_external_id, bot_id = resolved
    # Validate VAPID config early
    _get_vapid_keys()
    db = get_db_manager()
    await _upsert_push_subscription(
        db,
        session_id=session_id,
        user_external_id=str(user_external_id or ""),
        bot_id=str(bot_id or ""),
        subscription=req.subscription or {},
    )
    return {"status": "success"}


@app.post("/api/push/unsubscribe")
async def push_unsubscribe(
    request: Request,
    session_id: Optional[str] = Cookie(None),
    web_user_id: Optional[str] = Cookie(None, alias=_COOKIE_WEB_USER_ID),
    bot_id_cookie: Optional[str] = Cookie(None, alias=_COOKIE_BOT_ID),
):
    if not session_id:
        raise HTTPException(status_code=401, detail="未找到会话")
    db = get_db_manager()
    # 按 session 删除；若有 cookie 的 user_id/bot_id 则同时删该 user+bot 下所有订阅，避免重启后残留
    uid = (web_user_id or "").strip() or None
    bid = (bot_id_cookie or "").strip() or None
    await _delete_push_subscription(db, session_id=session_id, user_external_id=uid, bot_id=bid)
    return {"status": "success"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request):
    """直接进入 b 版 bot 选择页（暂不使用首页）。?source=xxx 会写入 cookie 供后续写入 User 表追踪。"""
    resp = HTMLResponse(
        content=get_bot_selection_html_b(),
        headers={"Cache-Control": "no-store, no-cache, must-revalidate"},
    )
    _set_visit_source_cookie_if_present(resp, request)
    return resp


@app.get("/bots", response_class=HTMLResponse)
async def bot_selection_page(request: Request):
    """始终返回 b 版 bot 选择页；?source=xxx 会写入 cookie 供后续写入 User 表追踪。"""
    resp = HTMLResponse(content=get_bot_selection_html_b())
    _set_visit_source_cookie_if_present(resp, request)
    return resp


@app.get("/chat/{bot_id}", response_class=HTMLResponse)
async def chat_with_bot(
    bot_id: str, request: Request, response: Response, 
    session_id: Optional[str] = Cookie(None)
):
    """直接链接到特定bot：自动初始化会话"""
    try:
        # 验证bot是否存在
        db = get_db_manager()
        async with db.Session() as session:
            bot_uuid = None
            try:
                import uuid as uuid_lib
                bot_uuid = uuid_lib.UUID(bot_id)
            except ValueError:
                pass
            
            if bot_uuid:
                result = await session.execute(
                    select(Bot).where(Bot.id == bot_uuid)
                )
            else:
                result = await session.execute(
                    select(Bot).where(Bot.name == bot_id)
                )
            bot = result.scalar_one_or_none()
            
            if not bot:
                # Bot不存在，返回选择页面
                return get_bot_selection_html_b()
            
            bot_id_str = str(bot.id)
        
        # 检查是否有有效会话且bot匹配
        if session_id:
            existing_session = get_session(session_id)
            if existing_session and existing_session["bot_id"] == bot_id_str:
                # 已有匹配的会话：若有 ?source= 或 cookie 的 visit_source，立即写 cookie、session，并更新 DB 中该 User 的 visit_source（即使用户已存在、本轮未发消息）
                _set_visit_source_cookie_if_present(response, request)
                src = (request.query_params.get("source") or request.cookies.get(_COOKIE_VISIT_SOURCE) or "").strip()
                if src:
                    set_session_visit_source(session_id, src)
                    try:
                        db = get_db_manager()
                        ok = await db.set_visit_source(
                            existing_session["user_id"],
                            bot_id_str,
                            src,
                        )
                        if not ok:
                            logging.getLogger(__name__).warning(
                                "set_visit_source: user not found external_id=%r bot_id=%r",
                                existing_session.get("user_id"), bot_id_str,
                            )
                    except Exception as e:
                        logging.getLogger(__name__).warning(
                            "set_visit_source failed: %s", e, exc_info=True,
                        )
                return get_chat_html_b(bot_id_str)
        
        # 创建新会话（优先复用持久 cookie 的 web_user_id，避免 IP 变化导致“历史丢失”）
        _set_visit_source_cookie_if_present(response, request)
        visit_src = (request.query_params.get("source") or request.cookies.get(_COOKIE_VISIT_SOURCE) or "").strip() or None
        user_id = (request.cookies.get(_COOKIE_WEB_USER_ID) or "").strip() or generate_user_id_from_request(request)
        new_session_id = create_session(user_id, bot_id_str, visit_source=visit_src)
        _set_persistent_session_cookies(response, user_id=user_id, bot_id=bot_id_str, session_id=new_session_id)
        
        # 返回聊天界面（b 版）
        return get_chat_html_b(bot_id_str)
    except Exception as e:
        # 出错时返回选择页面
        return get_bot_selection_html_b()


@app.get("/api/bots")
async def list_bots():
    """获取所有可用bot列表"""
    try:
        db = get_db_manager()
        async with db.Session() as session:
            result = await session.execute(select(Bot).order_by(Bot.name))
            bots = result.scalars().all()
            return {
                "bots": [
                    {
                        "id": str(bot.id),
                        "name": bot.name or "Unnamed Bot",
                        "basic_info": bot.basic_info or {},
                        "persona": bot.persona if isinstance(getattr(bot, "persona", None), dict) else {},
                        "big_five": bot.big_five or {},
                    }
                    for bot in bots
                ]
            }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取bot列表失败: {str(e)}")


@app.post("/api/session/init")
async def init_session(
    request: Request, response: Response, data: SessionInitRequest
):
    """初始化会话：选择bot"""
    try:
        # 生成或获取 user_id（优先复用持久 cookie 的 web_user_id，避免 IP 变化导致“历史丢失”）
        user_id = (request.cookies.get(_COOKIE_WEB_USER_ID) or "").strip() or generate_user_id_from_request(request)
        
        # 验证bot_id是否存在
        db = get_db_manager()
        async with db.Session() as session:
            bot_uuid = None
            try:
                import uuid as uuid_lib
                bot_uuid = uuid_lib.UUID(data.bot_id)
            except ValueError:
                pass
            
            if bot_uuid:
                result = await session.execute(
                    select(Bot).where(Bot.id == bot_uuid)
                )
            else:
                result = await session.execute(
                    select(Bot).where(Bot.name == data.bot_id)
                )
            bot = result.scalar_one_or_none()
            
            if not bot:
                raise HTTPException(status_code=404, detail="Bot不存在")
            
            bot_id = str(bot.id)
        
        # 创建会话 + 设置 Cookie；visit_source 优先从 body/query 取（前端可从当前 URL 传入），否则从 cookie
        visit_src = (
            (getattr(data, "source", None) or "").strip()
            or (request.query_params.get("source") or "").strip()
            or (request.cookies.get(_COOKIE_VISIT_SOURCE) or "").strip()
        ) or None
        session_id = create_session(user_id, bot_id, visit_source=visit_src)
        _set_persistent_session_cookies(response, user_id=user_id, bot_id=bot_id, session_id=session_id)
        
        return {
            "session_id": session_id,
            "bot_id": bot_id,
            "bot_name": bot.name,
            "status": "ready",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"初始化会话失败: {str(e)}")


@app.post("/api/session/resume")
async def resume_session(
    request: Request, response: Response, data: SessionResumeRequest
):
    """通过 User 数据库 ID (UUID) 恢复之前的会话"""
    try:
        import uuid as uuid_lib
        try:
            user_uuid = uuid_lib.UUID(data.user_db_id)
        except ValueError:
            raise HTTPException(status_code=400, detail="无效的会话ID格式，需要 UUID")

        db = get_db_manager()
        async with db.Session() as session:
            result = await session.execute(
                select(User).where(User.id == user_uuid)
            )
            user = result.scalar_one_or_none()
            if not user:
                raise HTTPException(status_code=404, detail="未找到该会话ID对应的用户")

            bot_id_str = str(user.bot_id)
            external_id = user.external_id

            # 获取 bot 名称
            result = await session.execute(
                select(Bot).where(Bot.id == user.bot_id)
            )
            bot = result.scalar_one_or_none()
            bot_name = bot.name if bot else None

        # 创建会话并设置持久 Cookie
        new_session_id = create_session(external_id, bot_id_str)
        _set_persistent_session_cookies(response, user_id=external_id, bot_id=bot_id_str, session_id=new_session_id)

        return {
            "session_id": new_session_id,
            "bot_id": bot_id_str,
            "bot_name": bot_name,
            "user_db_id": str(user_uuid),
            "status": "ready",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"恢复会话失败: {str(e)}")


@app.post("/api/chat")
async def chat(
    request: Request,
    response: Response,
    chat_data: ChatRequest,
    session_id: Optional[str] = Cookie(None),
    web_user_id: Optional[str] = Cookie(None, alias=_COOKIE_WEB_USER_ID),
    bot_id_cookie: Optional[str] = Cookie(None, alias=_COOKIE_BOT_ID),
):
    """处理聊天消息"""
    # Resolve user/bot; if in-memory session lost (Render restart), fall back to cookies and re-create session_id.
    resolved = _get_user_bot_from_session_or_cookies(
        session_id=session_id, web_user_id=web_user_id, bot_id_cookie=bot_id_cookie
    )
    if not resolved:
        raise HTTPException(status_code=401, detail="未找到会话，请先选择bot")
    user_id, bot_id = resolved
    if not session_id or not get_session(session_id):
        # recreate an in-memory session for concurrency control（从 cookie 带回 visit_source，避免 lab 等来源丢失）
        visit_src = (request.cookies.get(_COOKIE_VISIT_SOURCE) or "").strip() or None
        session_id = create_session(user_id, bot_id, visit_source=visit_src)
        _set_persistent_session_cookies(response, user_id=user_id, bot_id=bot_id, session_id=session_id)
    
    if not chat_data.message or not chat_data.message.strip():
        raise HTTPException(status_code=400, detail="消息不能为空")
    
    try:
        # -----
        # Interruptible in-flight chat:
        # - Always persist the user message immediately (so history keeps 2 separate user rows if user sends twice fast).
        # - If another message arrives while previous generation/settlement is running, cancel it and restart with merged user_input.
        # - Still pass the two user messages separately into chat_buffer (two consecutive user tags).
        # -----
        db = get_db_manager()
        inflight = _get_inflight_state(session_id)

        # Use milliseconds to keep ordering stable for bursty inputs
        received_iso = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
        msg_text = chat_data.message.strip()

        # 1) Persist this user message ASAP (history correctness)
        try:
            await db.append_message(
                user_id,
                bot_id,
                role="user",
                content=msg_text,
                created_at=received_iso,
                meta={"source": "web", "session_id": session_id},
            )
        except Exception:
            # best-effort: don't block generation
            pass

        # 2) Register as pending and cancel any in-flight tasks
        loop = asyncio.get_running_loop()
        waiter = loop.create_future()
        req_id = str(uuid.uuid4())

        async with inflight["lock"]:
            inflight["pending_user_msgs"].append({"content": msg_text, "ts": received_iso})

            # Supersede previous waiter (if any)
            old_waiter = inflight.get("waiter")
            if old_waiter is not None and hasattr(old_waiter, "done") and (not old_waiter.done()):
                try:
                    old_waiter.set_result({"status": "superseded"})
                except Exception:
                    pass

            inflight["waiter"] = waiter
            inflight["latest_req_id"] = req_id

            # Cancel in-flight generation task only.
            # tail_task writes the bot message to DB — do NOT cancel it or the message is lost.
            for k in ("fast_task",):
                t = inflight.get(k)
                if t is not None and hasattr(t, "done") and (not t.done()):
                    try:
                        t.cancel()
                    except Exception:
                        pass

            async def _run_one(req_id_local: str, skip_bot_write: bool = False):
                """skip_bot_write=True 时跳过写入 bot 消息（重试时避免重复落库）"""
                t_total = time.perf_counter()
                # Snapshot pending messages for this run
                async with inflight["lock"]:
                    pending = list(inflight.get("pending_user_msgs") or [])

                if not pending:
                    return

                merged_text = "\n".join([str(x.get("content") or "").strip() for x in pending if str(x.get("content") or "").strip()]).strip()
                if not merged_text:
                    return

                # Load fresh state (includes any just-appended user messages)；若有 session/cookie 的 visit_source 则写入 User 表追踪
                visit_src = None
                if session_id:
                    sess = get_session(session_id)
                    if isinstance(sess, dict):
                        visit_src = (sess.get("visit_source") or "").strip() or None
                if not visit_src:
                    visit_src = (request.cookies.get(_COOKIE_VISIT_SOURCE) or "").strip() or None
                # REAL_MODE: 新消息到来时，取消该用户/bot 下所有挂起的延迟任务。
                # 用户主动发消息意味着对话已恢复，之前排队的延迟回复不再需要。
                if _truthy(os.getenv("REAL_MODE_ENABLED", "false")):
                    try:
                        cancelled = await db.cancel_pending_responses(
                            user_external_id=user_id, bot_id=bot_id
                        )
                        if cancelled:
                            logging.getLogger(__name__).info(
                                "[WEB] cancelled %d pending response(s) for user=%s bot=%s",
                                cancelled, user_id, bot_id,
                            )
                    except Exception:
                        pass

                t0 = time.perf_counter()
                db_state = await db.load_state(user_id, bot_id, visit_source=visit_src)
                t_load_ms = (time.perf_counter() - t0) * 1000.0

                state = _make_initial_state(user_id, bot_id)
                state.update(db_state)

                # Provide bursty user inputs as 2 separate chat_buffer messages,
                # but use merged_text as "current user input" for detection/reasoning.
                state["messages"] = []  # let loader prefer state.user_input
                state["chat_buffer"] = [
                    HumanMessage(content=str(x.get("content") or ""), additional_kwargs={"timestamp": str(x.get("ts") or "")})
                    for x in pending
                ]
                state["user_input"] = merged_text
                state["external_user_text"] = merged_text
                state["current_time"] = str(pending[-1].get("ts") or received_iso)
                state["user_received_at"] = str(pending[0].get("ts") or received_iso)

                # Web：仅在显式设置环境变量时覆盖 LATS 配置（默认走系统原始策略/预算与评审）
                if os.getenv("WEB_LATS_ROLLOUTS") is not None:
                    try:
                        state["lats_rollouts"] = int(os.getenv("WEB_LATS_ROLLOUTS") or 0)
                    except Exception:
                        pass
                if os.getenv("WEB_LATS_EXPAND_K") is not None:
                    try:
                        state["lats_expand_k"] = int(os.getenv("WEB_LATS_EXPAND_K") or 0)
                    except Exception:
                        pass
                if os.getenv("WEB_ENABLE_LLM_SOFT_SCORER") is not None:
                    state["lats_enable_llm_soft_scorer"] = (
                        str(os.getenv("WEB_ENABLE_LLM_SOFT_SCORER", "0")).lower() in ("1", "true", "yes", "on")
                    )

                # Run fast graph (stdout -> log)
                log_file, log_path = get_or_create_log_file(session_id, user_id, bot_id)
                try:
                    log_file.write(f"[WEB_PERF] db.load_state_ms={t_load_ms:.1f} pending_msgs={len(pending)}\n")
                    log_file.flush()
                except Exception:
                    pass

                original_stdout = sys.stdout
                sys.stdout = FileOnlyWriter(log_file)
                _log_file_token = _current_web_log_file.set(log_file)  # 让 logger 也写入同一文件，最终落入 DB
                try:
                    fast_return = _truthy(os.getenv("WEB_FAST_RETURN", "1"))
                    graph = get_graph_fast() if fast_return else get_graph()
                    t0 = time.perf_counter()
                    result = await graph.ainvoke(state, config={"recursion_limit": 50})
                    t_graph_ms = (time.perf_counter() - t0) * 1000.0
                finally:
                    _current_web_log_file.reset(_log_file_token)
                    sys.stdout = original_stdout

                # ── 缺席门控：bot 决定延迟回复，本轮不生成内容 ──────────────────
                if result.get("absence_triggered"):
                    _ab_reason = result.get("absence_reason", "")
                    _ab_secs   = result.get("absence_delay_seconds", 0)
                    logging.getLogger(__name__).info(
                        "[WEB] absence_triggered: user=%s bot=%s reason=%s delay=%.0fs (%.1fh)",
                        user_id, bot_id, _ab_reason, _ab_secs, _ab_secs / 3600.0,
                    )
                    async with inflight["lock"]:
                        if inflight.get("latest_req_id") != req_id_local:
                            return
                        inflight["pending_user_msgs"] = []
                        _w = inflight.get("waiter")
                        if _w is not None and not _w.done():
                            _w.set_result({
                                "status": "absence",
                                "segments": [],
                                "reply": "",
                                "absence_reason": _ab_reason,
                                "absence_delay_seconds": _ab_secs,
                            })
                    return

                reply = result.get("final_response") or ""
                if not reply and result.get("final_segments"):
                    reply = " ".join(result["final_segments"])
                if not reply:
                    bot_name = (state.get("bot_basic_info") or {}).get("name") or "Ta"
                    reply = result.get("draft_response") or f"（{bot_name}决定不回你了）"

                # 本轮回复耗时（秒），追加到 reply_duration_seconds_list 供下一轮 strategy_resolver 判定是否走 fast
                new_reply_duration_list = (result.get("reply_duration_seconds_list") or []) + [t_graph_ms / 1000.0]

                # 流水线必有 processor，直接使用 humanized_output.segments（含 delay/action）。
                def _normalize_segment(s):
                    if isinstance(s, str):
                        return {"content": s, "delay": 0.0, "action": "typing"}
                    if isinstance(s, dict) and "content" in s:
                        return {**s, "content": str(s["content"]) if not isinstance(s.get("content"), str) else s["content"]}
                    return {"content": str(s), "delay": 0.0, "action": "typing"}

                humanized = result.get("humanized_output") or {}
                segments_with_delay = humanized.get("segments") or []
                segments = [_normalize_segment(s) for s in segments_with_delay]
                segments = [s for s in segments if (s.get("content") or "").strip()]
                if not segments and reply:
                    segments = [{"content": str(reply).strip(), "delay": 0.0, "action": "typing"}]

                ai_sent_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                if isinstance(result, dict):
                    result["ai_sent_at"] = ai_sent_at

                # 同步写入 bot 回复（切分后、发送给前端之前）；重试时 skip_bot_write=True 不再写，避免重复落库
                if db and not skip_bot_write:
                    try:
                        _stage = (result.get("current_stage") if isinstance(result, dict) else None) or (state.get("current_stage") if isinstance(state, dict) else None)
                        
                        # ai_sent_at is ISO string, parse it to calculate segment times
                        try:
                            # Note: fromisoformat in older python might be strict about Z vs +00:00, but here we generated it
                            if ai_sent_at.endswith('Z'):
                                base_time = datetime.fromisoformat(ai_sent_at[:-1] + '+00:00')
                            else:
                                base_time = datetime.fromisoformat(ai_sent_at)
                        except Exception:
                            base_time = datetime.now(timezone.utc)
                            
                        cumulative_delay = 0.0
                        for _idx, _seg in enumerate(segments):
                            _text = str(_seg.get("content", "") or "").strip()
                            # Accumulate delay: frontend shows "typing" for this duration BEFORE showing the message
                            _delay = float(_seg.get("delay") or 0.0)
                            cumulative_delay += _delay
                            
                            if _text:
                                # Calculate timestamp for this segment
                                seg_ts = base_time + timedelta(seconds=cumulative_delay)
                                seg_ts_iso = seg_ts.isoformat(timespec="milliseconds")
                                
                                await db.append_message(
                                    user_id, bot_id,
                                    role="ai",
                                    content=_text,
                                    created_at=seg_ts_iso,
                                    meta={"source": "web", "session_id": session_id, "segment_index": _idx, "current_stage": _stage},
                                )
                    except Exception:
                        pass

                # Tail settlement: skip writing user message (already persisted individually above)
                if _truthy(os.getenv("WEB_FAST_RETURN", "1")):
                    tail_state = dict(state)
                    if isinstance(result, dict):
                        tail_state.update(result)
                    tail_state["ai_sent_at"] = ai_sent_at
                    tail_state["skip_user_message_write"] = True
                    tail_state["skip_bot_message_write"] = True  # already written above
                    tail_state["reply_duration_seconds_list"] = new_reply_duration_list
                    tail_state["relationship_state"] = {
                        **(tail_state.get("relationship_state") or {}),
                        "reply_duration_seconds_list": new_reply_duration_list,
                    }

                    async def _run_tail_local():
                        log_file2, _ = get_or_create_log_file(session_id, user_id, bot_id)
                        orig = sys.stdout
                        sys.stdout = FileOnlyWriter(log_file2)
                        _tail_token = _current_web_log_file.set(log_file2)
                        try:
                            tail_graph = get_graph_tail()
                            await tail_graph.ainvoke(tail_state, config={"recursion_limit": 50})
                        except asyncio.CancelledError:
                            raise
                        except Exception as e:
                            try:
                                print(f"[WEB_BG] tail graph failed: {e}")
                            except Exception:
                                pass
                        finally:
                            _current_web_log_file.reset(_tail_token)
                            sys.stdout = orig

                    try:
                        tail_task = asyncio.create_task(_run_tail_local())
                        async with inflight["lock"]:
                            inflight["tail_task"] = tail_task
                    except Exception:
                        pass
                else:
                    # 非 fast return：主图已跑完 memory_writer，但当时 state 无本轮耗时，此处补写
                    try:
                        if db:
                            await db.update_reply_duration_list(user_id, bot_id, new_reply_duration_list)
                        else:
                            from app.core import LocalStoreManager
                            store = LocalStoreManager()
                            store.update_reply_duration_list(user_id, bot_id, new_reply_duration_list)
                    except Exception:
                        pass

                # Write perf + persist web_chat log snapshot
                try:
                    t_total_ms = (time.perf_counter() - t_total) * 1000.0
                    log_file3, log_path = get_or_create_log_file(session_id, user_id, bot_id)
                    log_file3.write(f"[WEB_PERF] graph_ms={t_graph_ms:.1f} total_ms={t_total_ms:.1f} log={log_path}\n")
                    log_file3.flush()
                    log_web_chat(session_id, user_id, bot_id, merged_text, reply)

                    # 每轮对话后把当前 log 快照写入 DB（Render 无持久盘，依赖 web_chat_logs 表）
                    try:
                        content = log_path.read_text(encoding="utf-8", errors="replace")
                        await db.upsert_web_chat_log(
                            user_external_id=str(user_id),
                            bot_id=str(bot_id),
                            session_id=str(session_id),
                            filename=log_path.name,
                            content=content,
                        )
                    except Exception:
                        pass

                    # 额外：把 log 备份按类归档写入 users.assets（不涉及记忆/结算，仅做运维备份）
                    try:
                        await db.append_user_log_backup(
                            user_external_id=str(user_id),
                            bot_id=str(bot_id),
                            session_id=str(session_id),
                            kind="web_chat_turn",
                            payload={
                                "session_id": str(session_id),
                                "user_input": str(merged_text or ""),
                                "bot_reply": str(reply or ""),
                                "ai_sent_at": str(ai_sent_at or ""),
                                "user_received_at": str(received_iso or ""),
                            },
                        )
                    except Exception:
                        pass
                except Exception:
                    pass

                # Web Push (best-effort): send once per bot turn, regardless of waiter state.
                try:
                    sub = await _get_push_subscription(
                        db,
                        session_id=session_id,
                        user_external_id=user_id,
                        bot_id=bot_id,
                    )
                    if isinstance(sub, dict) and sub:
                        bot_name = ""
                        try:
                            bot_name = str((state.get("bot_basic_info") or {}).get("name") or "")  # type: ignore[union-attr]
                        except Exception:
                            bot_name = ""
                        title = bot_name or "Chatbot"
                        body = (str(segments[0].get("content", "")) if segments else str(reply or ""))[:200]
                        await _send_web_push(subscription=sub, title=title, body=body, url="/", tag="ltsr-bot-message")
                    else:
                        logging.getLogger(__name__).info(
                            "push: no subscription session_id=%r user_id=%r bot_id=%r",
                            session_id, user_id, bot_id,
                        )
                except Exception as e:
                    logging.getLogger(__name__).warning("push send failed: %s", e, exc_info=True)

                # Complete waiter only if still latest
                async with inflight["lock"]:
                    if inflight.get("latest_req_id") != req_id_local:
                        return
                    inflight["pending_user_msgs"] = []
                    w = inflight.get("waiter")
                    if w is not None and (not w.done()):
                        w.set_result(
                            {
                                "reply": reply,
                                "segments": segments,
                                "status": "success",
                                "user_created_at": str(pending[0].get("ts") or received_iso),
                                "ai_created_at": ai_sent_at,
                                "merged_user_messages": len(pending),
                            }
                        )

            async def _run_one_with_retry(req_id_local: str):
                _MAX_RETRIES = 2
                _last_error = None
                for _attempt in range(1, _MAX_RETRIES + 1):
                    async with inflight["lock"]:
                        if inflight.get("latest_req_id") != req_id_local:
                            return
                    try:
                        await _run_one(req_id_local, skip_bot_write=(_attempt > 1))
                        return
                    except asyncio.CancelledError:
                        raise
                    except Exception as _exc:
                        _last_error = _exc
                        print(f"[WEB_RETRY] attempt {_attempt}/{_MAX_RETRIES} failed: {type(_exc).__name__}: {_exc}")
                        if _attempt < _MAX_RETRIES:
                            await asyncio.sleep(1.0)
                            continue

                async with inflight["lock"]:
                    if inflight.get("latest_req_id") != req_id_local:
                        return
                    inflight["pending_user_msgs"] = []
                    w = inflight.get("waiter")
                    if w is not None and (not w.done()):
                        error_msg = f"{type(_last_error).__name__}: {_last_error}" if _last_error else "未知错误"
                        w.set_result({"status": "error", "detail": error_msg})

            inflight["fast_task"] = asyncio.create_task(_run_one_with_retry(req_id))

        # 3) Wait for reply or superseded
        out = await waiter
        if isinstance(out, dict) and out.get("status") == "superseded":
            # Not an error: this request was intentionally superseded by a newer user message.
            # Frontend should ignore this response and wait for the newer request's result.
            return JSONResponse(status_code=200, content={"status": "superseded"})
        return out
    except LLMAPIError as e:
        print(f"[WEB_ERROR] LLMAPIError: {e}")
        raise HTTPException(status_code=503, detail="服务暂时不可用，请稍后重试。")
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        error_type = type(e).__name__
        error_msg = str(e)
        print(f"[WEB_ERROR] Chat error ({error_type}): {error_msg}")
        print(f"[WEB_ERROR] Traceback:\n{error_detail}")
        # 返回更详细的错误信息给前端
        detail_msg = f"{error_type}: {error_msg}"
        if os.getenv("ENVIRONMENT") != "production":
            # 开发环境返回完整堆栈信息
            detail_msg += f"\n\n堆栈信息:\n{error_detail}"
        raise HTTPException(status_code=500, detail=detail_msg)


@app.post("/api/session/reset")
async def reset_session(
    request: Request,
    response: Response,
    session_id: Optional[str] = Cookie(None),
    web_user_id: Optional[str] = Cookie(None, alias=_COOKIE_WEB_USER_ID),
    bot_id_cookie: Optional[str] = Cookie(None, alias=_COOKIE_BOT_ID),
):
    """清空该用户在当前 bot 下的所有对话与记忆（数据库/本地存储）。"""
    resolved = _get_user_bot_from_session_or_cookies(
        session_id=session_id, web_user_id=web_user_id, bot_id_cookie=bot_id_cookie
    )
    if not resolved:
        raise HTTPException(status_code=401, detail="未找到会话")
    user_id, bot_id = resolved
    # Ensure an in-memory session exists (so subsequent calls don't error)
    if not session_id or not get_session(session_id):
        session_id = create_session(user_id, bot_id)
        _set_persistent_session_cookies(response, user_id=user_id, bot_id=bot_id, session_id=session_id)

    try:
        # 1) Prefer DB: delete messages + memories/transcripts/notes + reset summary/stage
        try:
            db = get_db_manager()
            counts = await db.clear_all_memory_for(user_id, bot_id, reset_profile=True)
            return {"status": "success", "message": "对话历史已清空", "deleted": counts}
        except Exception:
            pass

        # 2) Fallback: local store (no DATABASE_URL)
        try:
            from app.core import LocalStoreManager

            store = LocalStoreManager()
            ok = store.clear_relationship(user_id, bot_id)
            return {
                "status": "success",
                "message": "对话历史已清空",
                "deleted": {"local_store": 1 if ok else 0},
            }
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"重置会话失败: {str(e)}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"重置会话失败: {str(e)}")


@app.get("/api/chat/history")
async def get_chat_history(
    session_id: Optional[str] = Cookie(None),
    web_user_id: Optional[str] = Cookie(None, alias=_COOKIE_WEB_USER_ID),
    bot_id_cookie: Optional[str] = Cookie(None, alias=_COOKIE_BOT_ID),
    limit: int = 2000,
):
    """获取当前用户在该 bot 下的全部对话历史（按时间升序）。"""
    resolved = _get_user_bot_from_session_or_cookies(
        session_id=session_id, web_user_id=web_user_id, bot_id_cookie=bot_id_cookie
    )
    if not resolved:
        raise HTTPException(status_code=401, detail="未找到会话，请先选择bot")
    user_external_id, bot_id_str = resolved

    # guardrails
    try:
        limit = int(limit or 0)
    except Exception:
        limit = 2000
    limit = max(1, min(limit, 5000))

    try:
        db = get_db_manager()
    except Exception:
        # no db configured -> no history
        return {"status": "success", "messages": []}

    async with db.Session() as db_session:
        # fetch bot
        bot_uuid = None
        try:
            import uuid as uuid_lib

            bot_uuid = uuid_lib.UUID(bot_id_str)
        except Exception:
            bot_uuid = None

        if bot_uuid:
            result = await db_session.execute(select(Bot).where(Bot.id == bot_uuid))
        else:
            result = await db_session.execute(select(Bot).where(Bot.name == bot_id_str))
        bot = result.scalar_one_or_none()
        if not bot:
            return {"status": "success", "messages": []}

        # fetch user row without creating new ones
        result = await db_session.execute(
            select(User).where(User.bot_id == bot.id, User.external_id == user_external_id)
        )
        user = result.scalar_one_or_none()
        if not user:
            return {"status": "success", "messages": []}

        role_order = case(
            (Message.role == "user", 0),
            (Message.role == "ai", 1),
            else_=2,
        )
        result = await db_session.execute(
            select(Message)
            .where(Message.user_id == user.id)
            .order_by(Message.created_at.asc(), role_order.asc(), Message.id.asc())
            .limit(limit)
        )
        msgs = list(result.scalars().all())

    out = []
    for m in msgs:
        role = str(getattr(m, "role", "") or "")
        if role not in ("user", "ai", "system"):
            continue
        out.append(
            {
                "role": role,
                "content": str(getattr(m, "content", "") or ""),
                "created_at": (
                    m.created_at.isoformat() if getattr(m, "created_at", None) is not None else None
                ),
            }
        )
    return {"status": "success", "messages": out}


@app.get("/api/session/status")
async def get_session_status(
    session_id: Optional[str] = Cookie(None),
    web_user_id: Optional[str] = Cookie(None, alias=_COOKIE_WEB_USER_ID),
    bot_id_cookie: Optional[str] = Cookie(None, alias=_COOKIE_BOT_ID),
):
    """获取会话状态"""
    resolved = _get_user_bot_from_session_or_cookies(
        session_id=session_id, web_user_id=web_user_id, bot_id_cookie=bot_id_cookie
    )
    if not resolved:
        return {"has_session": False}
    user_external_id, bot_id_str = resolved

    bot_name = None
    bot_basic_info = {}
    has_history = False
    user_db_id = None
    bot_big_five = {}
    bot_mood_state = {}
    user_dimensions = {}
    user_current_stage = None

    try:
        db = get_db_manager()
        async with db.Session() as db_session:
            # 获取 bot 信息
            bot_uuid = None
            try:
                import uuid as uuid_lib
                bot_uuid = uuid_lib.UUID(bot_id_str)
            except ValueError:
                pass

            if bot_uuid:
                result = await db_session.execute(select(Bot).where(Bot.id == bot_uuid))
            else:
                result = await db_session.execute(select(Bot).where(Bot.name == bot_id_str))
            bot = result.scalar_one_or_none()

            if bot:
                bot_name = bot.name
                bot_basic_info = bot.basic_info or {}
                bot_big_five = dict(bot.big_five or {})
                bot_mood_state = dict(bot.mood_state or {})
                # 老 Bot 可能没有 big_five/mood_state：用默认生成/占位，仅用于状态面板展示
                _has_bf = any(
                    bot_big_five.get(k) is not None
                    for k in ("openness", "O", "conscientiousness", "C", "extraversion", "E")
                )
                if not _has_bf:
                    try:
                        _, bf, _ = generate_bot_profile(str(bot.id))
                        bot_big_five = dict(bf)
                    except Exception:
                        pass
                _has_pad = any(
                    bot_mood_state.get(k) is not None
                    for k in ("pleasure", "arousal", "dominance", "busyness")
                )
                if not _has_pad:
                    bot_mood_state = {"pleasure": 0, "arousal": 0, "dominance": 0, "busyness": 0, "pad_scale": "m1_1"}

                # 确保 User 存在，以便页头能显示 user_db_id（否则要等首次发消息才创建）
                user = await db._get_or_create_user(db_session, str(bot.id), user_external_id)
                user_db_id = str(user.id)
                user_dimensions = dict(user.dimensions or {})
                if not user_dimensions:
                    user_dimensions = get_random_relationship_template()
                user_current_stage = user.current_stage or None
                result = await db_session.execute(
                    select(Message.id).where(Message.user_id == user.id).limit(1)
                )
                has_history = result.scalar_one_or_none() is not None
                # 无历史时写入开场白到聊天历史，便于前端拉取并展示
                if not has_history:
                    first_msg = "你好，我是" + (bot_name or "Chatbot")
                    db_session.add(
                        Message(user_id=user.id, role="ai", content=first_msg)
                    )
                    await db_session.commit()
                    has_history = True
    except Exception:
        # 状态接口尽量不因数据库异常影响页面；前端会降级显示通用开场白
        logger.exception("[session/status] 获取会话状态异常，状态面板将降级显示")

    return {
        "has_session": True,
        "bot_id": bot_id_str,
        "user_id": user_external_id,
        "user_db_id": user_db_id,
        "bot_name": bot_name,
        "bot_basic_info": bot_basic_info,
        "has_history": has_history,
        "bot_big_five": bot_big_five,
        "bot_mood_state": bot_mood_state,
        "user_dimensions": user_dimensions,
        "user_current_stage": user_current_stage,
    }


@app.get("/api/share-link/{bot_id}")
async def get_share_link(bot_id: str, request: Request):
    """生成分享链接"""
    try:
        # 验证bot是否存在
        db = get_db_manager()
        async with db.Session() as session:
            bot_uuid = None
            try:
                import uuid as uuid_lib
                bot_uuid = uuid_lib.UUID(bot_id)
            except ValueError:
                pass
            
            if bot_uuid:
                result = await session.execute(
                    select(Bot).where(Bot.id == bot_uuid)
                )
            else:
                result = await session.execute(
                    select(Bot).where(Bot.name == bot_id)
                )
            bot = result.scalar_one_or_none()
            
            if not bot:
                raise HTTPException(status_code=404, detail="Bot不存在")
            
            bot_id_str = str(bot.id)
        
        # 生成分享链接
        # 获取基础URL
        base_url = str(request.base_url).rstrip('/')
        # 如果配置了自定义域名，使用配置的域名
        custom_domain = os.getenv("WEB_DOMAIN")
        if custom_domain:
            base_url = f"https://{custom_domain}"
        
        share_link = f"{base_url}/chat/{bot_id_str}"
        
        return {
            "bot_id": bot_id_str,
            "bot_name": bot.name,
            "share_link": share_link,
            "qr_code_url": f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={share_link}",
        }
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"生成分享链接失败: {str(e)}")


@app.get("/api/share-links")
async def get_all_share_links(request: Request):
    """获取所有bot的分享链接"""
    try:
        db = get_db_manager()
        async with db.Session() as session:
            result = await session.execute(select(Bot).order_by(Bot.name))
            bots = result.scalars().all()
        
        # 获取基础URL
        base_url = str(request.base_url).rstrip('/')
        custom_domain = os.getenv("WEB_DOMAIN")
        if custom_domain:
            base_url = f"https://{custom_domain}"
        
        share_links = []
        for bot in bots:
            bot_id_str = str(bot.id)
            share_link = f"{base_url}/chat/{bot_id_str}"
            share_links.append({
                "bot_id": bot_id_str,
                "bot_name": bot.name or "Unnamed Bot",
                "share_link": share_link,
                "qr_code_url": f"https://api.qrserver.com/v1/create-qr-code/?size=200x200&data={share_link}",
            })
        
        return {
            "base_url": base_url,
            "share_links": share_links,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"获取分享链接失败: {str(e)}")


def _get_render_base_url(request: Request) -> str:
    """Render 基础 URL：优先 RENDER_EXTERNAL_URL，其次 WEB_DOMAIN，最后 request.base_url"""
    base = os.getenv("RENDER_EXTERNAL_URL", "").strip().rstrip("/")
    if base:
        return base
    if os.getenv("WEB_DOMAIN"):
        return f"https://{os.getenv('WEB_DOMAIN').strip()}"
    return str(request.base_url).rstrip("/")


@app.get("/api/render-links")
async def get_render_links(request: Request, source: Optional[str] = None):
    """
    生成带不同 source 参数的 Render 链接，用于分流/统计。
    - 无 query：返回所有预设 source 的链接（lab, demo 等）
    - ?source=lab：只返回该 source 的链接
    预设 source 列表可通过环境变量 RENDER_LINK_SOURCES 配置，逗号分隔，如：lab,demo,invite,share
    """
    base_url = _get_render_base_url(request)
    sources_str = (os.getenv("RENDER_LINK_SOURCES") or "lab,demo,invite,share").strip()
    sources = [s.strip() for s in sources_str.split(",") if s.strip()]
    if not sources:
        sources = ["lab", "demo", "invite", "share"]

    if source is not None:
        # 只返回指定 source 的一条链接
        one = source.strip()
        if not one:
            raise HTTPException(status_code=400, detail="source 不能为空")
        url = f"{base_url}/?source={one}"
        return {"base_url": base_url, "source": one, "url": url}

    links = {s: f"{base_url}/?source={s}" for s in sources}
    return {"base_url": base_url, "links": links}


# HTML 模板函数

def get_home_html() -> str:
    """V2 极简科研风首页：左文右图 + 大按钮 + 按钮下方内容占位区"""
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>面向多角色与持续会话的对话系统界面原型</title>
    <link rel="stylesheet" href="/static/b/styles.css">
    <link rel="stylesheet" href="/static/b/home.css?v=3">
</head>
<body>
    <main class="app-shell home-min">
        <div class="home-hero">
            <div class="home-copy">
                <div class="home-brand">
                    EmotionalChatBot <span class="home-muted">v5.0</span>
                </div>

                <h1 class="home-title">面向多角色与持续会话的对话系统界面原型</h1>
                <div class="home-subtitle">
                    A Multi‑Persona Chat Interface Prototype for Longitudinal Dialogue
                </div>

                <p class="home-intro">
                    本页面为研究原型入口，用于展示多角色选择与持续对话的交互流程。
                    演示页面提供角色列表与会话入口。
                </p>

                <ul class="home-bullets">
                    <li><b>多角色（Persona）</b>：统一的角色呈现与选择入口。</li>
                    <li><b>持续会话</b>：会话状态持久化与恢复机制（演示页实现）。</li>
                    <li><b>研究用途</b>：用于交互流程验证与多角色对比实验准备。</li>
                </ul>

                <a class="btn-primary home-cta home-cta-lg" href="/bots">
                    进入演示 / Choose a Chatbot
                </a>

                <div class="home-meta">
                    Research Prototype · Last updated: 2026‑02 · License: MIT
                </div>
            </div>

            <figure class="home-figure bot-card">
                <div class="home-figure-placeholder">Figure 0 · UI Preview</div>
                <figcaption class="home-figcap">
                    图0：多角色选择与对话界面预览（示例）。
                </figcaption>
            </figure>
        </div>

        <section class="home-content-below" aria-label="页面内容">
            <div class="home-placeholder-block">
                <h2 class="home-placeholder-title">内容占位一</h2>
                <p class="home-placeholder-text">此处可放置说明、数据概览或功能入口。保留区域便于后续扩展。</p>
            </div>
            <div class="home-placeholder-block">
                <h2 class="home-placeholder-title">内容占位二</h2>
                <p class="home-placeholder-text">第二块占位区域，可用于展示实验设置、参与方式或相关链接。</p>
            </div>
            <div class="home-placeholder-block">
                <h2 class="home-placeholder-title">内容占位三</h2>
                <p class="home-placeholder-text">第三块占位区域，可放置更新日志、引用或联系方式等。</p>
            </div>
        </section>
    </main>
</body>
</html>"""


def get_bot_selection_html() -> str:
    """返回bot选择页面HTML"""
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>选择 Chatbot - EmotionalChatBot</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="container">
        <div class="bot-selection">
            <h1>🤖 选择一个 Chatbot 开始对话</h1>
            <div class="resume-session">
                <h3>通过会话ID 恢复之前的会话</h3>
                <div class="resume-session-row">
                    <input type="text" id="resume-user-id" class="message-input" placeholder="输入会话ID (UUID)..." autocomplete="off" />
                    <button class="btn-primary" onclick="resumeByUserId()">恢复会话</button>
                </div>
            </div>
            <div id="bot-list" class="bot-list">
                <div class="loading">加载中...</div>
            </div>
        </div>
    </div>
    <script src="/static/chat.js"></script>
    <script>
        // 初始化bot列表
        loadBots();
    </script>
</body>
</html>"""


def get_chat_html(bot_id: str) -> str:
    """返回聊天界面HTML"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Chat - EmotionalChatBot</title>
    <link rel="stylesheet" href="/static/styles.css">
</head>
<body>
    <div class="container">
        <div class="chat-container">
            <div class="chat-header">
                <div style="display:flex; align-items:center; gap:12px; flex:1; min-width:0;">
                    <h2 style="margin:0;">💬 对话中</h2>
                    <div id="user-db-id" class="user-id-bar"></div>
                </div>
                <div style="display:flex; gap:8px; align-items:center;">
                    <a href="/bots" class="btn-secondary" style="text-decoration:none;">返回首页</a>
                    <button id="notify-btn" class="btn-secondary" title="开启浏览器通知（需要授权）" style="display:none;">开启通知</button>
                </div>
            </div>
            <div id="chat-messages" class="chat-messages"></div>
            <div class="chat-input-container">
                <input 
                    type="text" 
                    id="message-input" 
                    class="message-input" 
                    placeholder="输入消息..."
                    autocomplete="off"
                />
                <button type="button" id="send-btn" class="btn-primary">发送</button>
            </div>
        </div>
    </div>
    <script src="/static/chat.js"></script>
    <script>
        // 初始化聊天界面
        initChat();
    </script>
</body>
</html>"""


# ---------- B 版界面（紫色玻璃风，带头像与 chip） ----------

# 全局顶栏公告配置（id / message / cta / href / variant: glass | soft）
# 置空则顶栏不展示任何公告；下次需要时按下面格式填回即可。
_ANNOUNCE_CONFIG: dict = {}


def _announcement_bar_root_html() -> str:
    """渲染顶栏挂载点（data-config 供 announcement-bar.js 读取）"""
    cfg = json.dumps(_ANNOUNCE_CONFIG, ensure_ascii=False)
    return f'<div id="announcement-bar-root" data-config="{html.escape(cfg, quote=True)}"></div>'


def get_bot_selection_html_b() -> str:
    """B 版：选 Bot 页 HTML"""
    return """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>选择 Chatbot</title>
    <link rel="stylesheet" href="/static/b/styles.css">
</head>
<body>
    """ + _announcement_bar_root_html() + """
    <div class="app-shell">
        <h1 class="h1" style="text-align:center; margin-bottom:24px;">选择一个 Chatbot 开始对话</h1>
        <div class="b-resume">
            <h3>通过会话ID 恢复之前的会话</h3>
            <div class="b-resume-row">
                <input type="text" id="resume-user-id" placeholder="输入会话ID (UUID)..." autocomplete="off" />
                <button class="btn-primary" onclick="resumeByUserId()">恢复会话</button>
            </div>
        </div>
        <div id="bot-list" class="bot-list">
            <div class="loading">加载中...</div>
        </div>
    </div>
    <script src="/static/b/announcement-bar.js"></script>
    <script src="/static/b/chat.js"></script>
    <script>loadBots();</script>
</body>
</html>"""


def get_chat_html_b(bot_id: str) -> str:
    """B 版：聊天页 HTML（含头部 bot 头像、气泡、输入条）"""
    return f"""<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>对话</title>
    <link rel="stylesheet" href="/static/b/styles.css">
</head>
<body>
    """ + _announcement_bar_root_html() + """
    <div class="app-shell">
        <div class="b-chat-header">
            <div class="b-chat-header-left">
                <div id="chat-header-avatar" class="avatar">?</div>
                <h1 id="chat-header-title" class="h1">对话中</h1>
                <div id="user-db-id" class="b-user-id"></div>
            </div>
            <div class="b-chat-header-right">
                <button type="button" id="state-panel-btn" class="icon-btn" title="角色状态">☰</button>
                <a href="/" class="btn-secondary">返回选 Bot</a>
            </div>
        </div>
        <div id="chat-messages" class="b-chat-messages"></div>
        <div class="chat-input-bar">
            <div class="chat-input">
                <input type="text" id="message-input" placeholder="输入消息..." autocomplete="off" />
                <button type="button" id="send-btn" class="btn-primary btn-send">发送</button>
            </div>
        </div>
    </div>
    <div id="state-panel-overlay" class="state-panel-overlay"></div>
    <div id="state-panel" class="state-panel" aria-label="角色状态面板">
        <button class="sp-close-btn" id="state-panel-close" type="button">✕</button>
        <div class="sp-section-title">大五人格</div>
        <div id="sp-big-five" class="big-five-bars">—</div>
        <div class="sp-section-title">情绪 PADB</div>
        <div id="sp-padb" class="big-five-bars">—</div>
        <div class="sp-section-title">关系维度</div>
        <div id="sp-relationship" class="big-five-bars">—</div>
        <div class="sp-section-title">Knapp 阶段</div>
        <div id="sp-stage">—</div>
    </div>
    <script src="/static/b/announcement-bar.js"></script>
    <script src="/static/b/chat.js"></script>
    <script>initChat();</script>
</body>
</html>"""


@app.get("/b", response_class=HTMLResponse)
async def bot_selection_page_b():
    """B 版：始终返回选 Bot 页"""
    return get_bot_selection_html_b()


@app.get("/b/chat/{bot_id}", response_class=HTMLResponse)
async def chat_with_bot_b(
    bot_id: str, request: Request, response: Response,
    session_id: Optional[str] = Cookie(None),
):
    """B 版：直接进入指定 bot 聊天，自动初始化会话"""
    try:
        db = get_db_manager()
        async with db.Session() as session:
            bot_uuid = None
            try:
                import uuid as uuid_lib
                bot_uuid = uuid_lib.UUID(bot_id)
            except ValueError:
                pass
            if bot_uuid:
                result = await session.execute(select(Bot).where(Bot.id == bot_uuid))
            else:
                result = await session.execute(select(Bot).where(Bot.name == bot_id))
            bot = result.scalar_one_or_none()
            if not bot:
                return get_bot_selection_html_b()
            bot_id_str = str(bot.id)

        if session_id:
            existing_session = get_session(session_id)
            if existing_session and existing_session["bot_id"] == bot_id_str:
                return get_chat_html_b(bot_id_str)

        user_id = (request.cookies.get(_COOKIE_WEB_USER_ID) or "").strip() or generate_user_id_from_request(request)
        new_session_id = create_session(user_id, bot_id_str)
        _set_persistent_session_cookies(response, user_id=user_id, bot_id=bot_id_str, session_id=new_session_id)
        return get_chat_html_b(bot_id_str)
    except Exception:
        return get_bot_selection_html_b()


# ══════════════════════════════════════════════════════════════════════════════
# Annotation Tool Routes  /annotation/
# ══════════════════════════════════════════════════════════════════════════════

_ANNOTATION_OUTPUT = Path(__file__).parent / "scripts" / "output"
_ANNOTATION_STATIC = Path(__file__).parent / "static" / "annotation"
_POOL_PATH = _ANNOTATION_OUTPUT / "deduped_pool.json"

# ── Annotation pool cache ──
_annotation_pool: list[dict] | None = None
_annotation_pool_indices: dict | None = None


def _pool_version() -> str:
    """Return a version hash based on pool file mtime+size, for cache invalidation."""
    if not _POOL_PATH.exists():
        return "missing"
    st = _POOL_PATH.stat()
    return f"{int(st.st_mtime)}_{st.st_size}"


def _invalidate_old_task_caches():
    """Delete all task cache files generated from an older pool version."""
    ver_file = _ANNOTATION_OUTPUT / ".pool_version"
    current_ver = _pool_version()
    old_ver = ver_file.read_text().strip() if ver_file.exists() else ""
    if old_ver == current_ver:
        return
    # Pool changed — wipe all cached task files
    import glob as _glob
    removed = 0
    for f in _glob.glob(str(_ANNOTATION_OUTPUT / "tasks_*.json")):
        Path(f).unlink(missing_ok=True)
        removed += 1
    if removed:
        logger.info("[Annotation] Pool updated, removed %d stale task caches", removed)
    ver_file.write_text(current_ver)


def _load_pool():
    """Load deduped pool into memory (cached)."""
    global _annotation_pool, _annotation_pool_indices
    if _annotation_pool is not None:
        return
    _invalidate_old_task_caches()
    if not _POOL_PATH.exists():
        raise FileNotFoundError("deduped_pool.json 不存在，请先运行 devtools/dedup_candidates.py")
    _annotation_pool = json.loads(_POOL_PATH.read_text("utf-8"))
    # Build indices
    from collections import defaultdict
    _STYLE_DIMS = ["FORMALITY", "POLITENESS", "FRIENDLINESS", "CERTAINTY", "EMOTIONAL_TONE"]
    by_route: dict[str, list[int]] = defaultdict(list)
    by_dim_tier: dict[str, dict[str, list[int]]] = {d: defaultdict(list) for d in _STYLE_DIMS}
    by_em: dict[int, list[int]] = defaultdict(list)
    for i, r in enumerate(_annotation_pool):
        by_route[r["route"]].append(i)
        tiers = r.get("tiers") or {}
        for d in _STYLE_DIMS:
            t = tiers.get(d, "M")
            by_dim_tier[d][t].append(i)
        by_em[int(r.get("em", 0))].append(i)
    _annotation_pool_indices = {
        "by_route": dict(by_route),
        "by_dim_tier": {d: dict(v) for d, v in by_dim_tier.items()},
        "by_em": dict(by_em),
    }


def _allocate_tasks(annotator_id: str) -> list[dict]:
    """Generate 1000 stratified tasks for one annotator, deterministic by ID."""
    import random as _random
    from collections import defaultdict

    _load_pool()
    pool = _annotation_pool
    idx = _annotation_pool_indices
    by_route = idx["by_route"]
    by_dim_tier = idx["by_dim_tier"]
    by_em = idx["by_em"]

    STYLE_DIMS = ["FORMALITY", "POLITENESS", "FRIENDLINESS", "CERTAINTY", "EMOTIONAL_TONE"]
    TIERS = ["EL", "L", "M", "H", "EH"]
    ROUTES = ["move_1", "move_2", "move_3", "move_4", "move_5",
              "move_6", "move_7", "move_8", "move_9", "move_10",
              "move_11", "move_12", "move_13", "free"]
    # B1: 10 种组合 × 12 对/格 × 5 维度 = 600 总（每维度 120 对）
    B1_GAP_SPEC = [
        (1, [("EL", "L"), ("L", "M"), ("M", "H"), ("H", "EH")], 12),  # 4×12=48
        (2, [("EL", "M"), ("L", "H"), ("M", "EH")], 12),              # 3×12=36
        (3, [("EL", "H"), ("L", "EH")], 12),                           # 2×12=24
        (4, [("EL", "EH")], 12),                                       # 1×12=12
    ]

    # Deterministic seed from annotator_id
    h = 0
    for c in annotator_id:
        h = ((h * 31) + ord(c)) & 0xFFFFFFFF
    rng = _random.Random(h)
    used: set[int] = set()
    tasks: list[dict] = []

    def _sample(indices, n):
        avail = [i for i in indices if i not in used]
        rng.shuffle(avail)
        return avail[:n]

    # ── 分配顺序：B1(style_compare) 优先，保证极端 tier 多样性 ──

    # ── Task B1: Style compare (400 = 5 dims × 80) ── 【最先分配】
    for dim in STYLE_DIMS:
        for gap, combos, per_combo in B1_GAP_SPEC:
            for tier_lo, tier_hi in combos:
                avail_lo = [i for i in by_dim_tier.get(dim, {}).get(tier_lo, []) if i not in used]
                avail_hi = [i for i in by_dim_tier.get(dim, {}).get(tier_hi, []) if i not in used]
                rng.shuffle(avail_lo)
                rng.shuffle(avail_hi)
                n = min(per_combo, len(avail_lo), len(avail_hi))
                for p in range(n):
                    idx_lo, idx_hi = avail_lo[p], avail_hi[p]
                    used.add(idx_lo)
                    used.add(idx_hi)
                    r_lo, r_hi = pool[idx_lo], pool[idx_hi]
                    swap = rng.random() < 0.5
                    if swap:
                        da, db, higher = r_hi, r_lo, "a"
                    else:
                        da, db, higher = r_lo, r_hi, "b"
                    tasks.append({
                        "task_type": "style_compare",
                        "task_id": f"B1_{len(tasks)}",
                        "dimension": dim,
                        "text_a_context": da["context"],
                        "text_a_bot": da["text"],
                        "text_b_context": db["context"],
                        "text_b_bot": db["text"],
                        "ground_truth": {
                            "tier_lo": tier_lo, "tier_hi": tier_hi,
                            "gap": gap, "higher": higher,
                        },
                        "is_qc_anchor": gap >= 3,
                        "is_repeat": False,
                    })

    # ── Task B2: Style label (100 = 5 dims × 20 = 5 × 5 tiers × 4) ──
    for dim in STYLE_DIMS:
        for tier in TIERS:
            tier_indices = by_dim_tier.get(dim, {}).get(tier, [])
            for i in _sample(tier_indices, 4):
                used.add(i)
                r = pool[i]
                tasks.append({
                    "task_type": "style_label",
                    "task_id": f"B2_{len(tasks)}",
                    "dimension": dim,
                    "context_user_text": r["context"],
                    "bot_text": r["text"],
                    "ground_truth": {"tier": tier, "value": round(r["style"].get(dim, 0), 4)},
                    "is_repeat": False,
                })

    # ── Task C: ExprMode (100, 纯随机) ──
    all_em_indices = []
    for em_list in by_em.values():
        all_em_indices.extend(em_list)
    for i in _sample(all_em_indices, 100):
        used.add(i)
        r = pool[i]
        tasks.append({
            "task_type": "expr_mode",
            "task_id": f"C_{len(tasks)}",
            "context_user_text": r["context"],
            "bot_text": r["text"],
            "ground_truth": {"em": r["em"]},
            "is_repeat": False,
        })

    # ── Task A: Move (150) ──
    per_route = 150 // len(ROUTES)
    remainder = 150 - per_route * len(ROUTES)
    for ri, route in enumerate(ROUTES):
        n = per_route + (1 if ri < remainder else 0)
        for i in _sample(by_route.get(route, []), n):
            used.add(i)
            r = pool[i]
            _route = r["route"]
            _mid = int(_route.split("_")[1]) if _route.startswith("move_") else 0
            tasks.append({
                "task_type": "move",
                "task_id": f"A_{len(tasks)}",
                "context_user_text": r["context"],
                "bot_text": r["text"],
                "ground_truth": {"route": _route},
                "must_include_move": _mid,
                "is_repeat": False,
            })

    # ── 5% hidden repeats ──
    by_type: dict[str, list[dict]] = defaultdict(list)
    for t in tasks:
        by_type[t["task_type"]].append(t)
    for ttype, ttasks in by_type.items():
        n_rep = max(1, int(len(ttasks) * 0.05))
        for src in rng.sample(ttasks, min(n_rep, len(ttasks))):
            rep = dict(src)
            rep["is_repeat"] = True
            rep["task_id"] = f"REP_{len(tasks)}"
            tasks.append(rep)

    # ── Sort by task type (grouped, not shuffled) ──
    type_order = {"move": 0, "style_compare": 1, "style_label": 2, "expr_mode": 3}
    tasks.sort(key=lambda t: type_order.get(t["task_type"], 9))
    # Shuffle within each group
    from itertools import groupby
    sorted_tasks = []
    for _key, group in groupby(tasks, key=lambda t: t["task_type"]):
        g = list(group)
        rng.shuffle(g)
        sorted_tasks.extend(g)
    tasks = sorted_tasks
    for i, t in enumerate(tasks):
        t["task_id"] = f"{annotator_id}_{i + 1:04d}"

    return tasks


def _get_or_create_tasks(annotator_id: str) -> list[dict]:
    """Get cached tasks file or generate new one."""
    _ANNOTATION_OUTPUT.mkdir(parents=True, exist_ok=True)
    # Use hash of annotator_id for filename
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in annotator_id)
    task_file = _ANNOTATION_OUTPUT / f"tasks_{safe_name}.json"

    if task_file.exists():
        return json.loads(task_file.read_text("utf-8"))

    tasks = _allocate_tasks(annotator_id)
    task_file.write_text(json.dumps(tasks, ensure_ascii=False))

    # Log annotator registration
    log_file = _ANNOTATION_OUTPUT / "annotator_log.jsonl"
    with open(log_file, "a", encoding="utf-8") as f:
        entry = {
            "annotator_id": annotator_id,
            "registered_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "total_tasks": len(tasks),
        }
        f.write(json.dumps(entry, ensure_ascii=False) + "\n")

    return tasks


def _get_submitted_task_ids(annotator_id: str) -> set[str]:
    """Collect task_ids already submitted by this annotator."""
    import csv as _csv
    submitted: set[str] = set()
    results_path = _ANNOTATION_OUTPUT / "annotation_results.csv"
    if not results_path.exists():
        return submitted
    try:
        with open(results_path, "r", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                if row.get("annotator_id", "").strip() == annotator_id:
                    tid = row.get("task_id", "").strip()
                    if tid:
                        submitted.add(tid)
    except Exception:
        pass
    return submitted


@app.get("/annotation/", response_class=HTMLResponse)
async def annotation_page():
    """标注工具主页面。"""
    html_path = _ANNOTATION_STATIC / "index.html"
    if not html_path.exists():
        return HTMLResponse("<h1>标注页面尚未部署</h1><p>请联系管理员。</p>", status_code=503)
    return HTMLResponse(html_path.read_text("utf-8"))


@app.get("/annotation/tasks")
async def get_annotation_tasks(annotator_id: str):
    """
    生成或加载该标注员的任务集，过滤已提交的，返回剩余任务。
    首次调用时动态生成任务并缓存到文件。
    """
    annotator_id = annotator_id.strip()
    if not annotator_id:
        raise HTTPException(status_code=400, detail="annotator_id 不能为空")

    try:
        all_tasks = _get_or_create_tasks(annotator_id)
    except FileNotFoundError as e:
        return JSONResponse({
            "tasks": [], "total": 0, "remaining": 0,
            "error": str(e),
        })

    submitted = _get_submitted_task_ids(annotator_id)
    remaining = [t for t in all_tasks if t["task_id"] not in submitted]

    # Load move definitions for frontend
    moves_data = []
    try:
        from utils.yaml_loader import load_pure_content_transformations
        raw = load_pure_content_transformations()
        if isinstance(raw, dict):
            raw = raw.get("pure_content_transformations") or []
        moves_data = [
            {"id": int(m.get("id")), "name": str(m.get("name") or ""),
             "desc": str(m.get("content_operation") or "")[:100]}
            for m in (raw if isinstance(raw, list) else [])
            if m.get("id") is not None
        ]
    except Exception:
        pass

    return JSONResponse({
        "tasks": remaining,
        "total": len(all_tasks),
        "remaining": len(remaining),
        "done": len(submitted),
        "moves": moves_data,
    })


@app.post("/annotation/submit")
async def submit_annotation(request: Request):
    """提交一条标注结果，统一追加到 annotation_results.csv。"""
    import csv as _csv

    data = await request.json()
    task_type    = (data.get("task_type") or "").strip().lower()
    annotator_id = (data.get("annotator_id") or "").strip()
    task_id      = (data.get("task_id") or "").strip()

    if not task_type or not annotator_id or not task_id:
        raise HTTPException(status_code=400, detail="task_type / annotator_id / task_id 不能为空")
    if task_type not in ("move", "style_compare", "style_label", "expr_mode"):
        raise HTTPException(status_code=400,
                            detail="task_type 必须是 move / style_compare / style_label / expr_mode")

    ts = datetime.now(timezone.utc).isoformat(timespec="seconds")
    fieldnames = ["annotator_id", "task_id", "task_type", "answer", "timestamp"]
    row = {
        "annotator_id": annotator_id,
        "task_id":       task_id,
        "task_type":     task_type,
        "answer":        json.dumps(data.get("answer", ""), ensure_ascii=False),
        "timestamp":     ts,
    }

    _ANNOTATION_OUTPUT.mkdir(parents=True, exist_ok=True)
    results_path = _ANNOTATION_OUTPUT / "annotation_results.csv"
    write_header = not results_path.exists()
    with open(results_path, "a", newline="", encoding="utf-8") as f:
        writer = _csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow(row)

    return JSONResponse({"ok": True})


@app.delete("/annotation/reset_tasks")
async def reset_annotator_tasks(annotator_id: str):
    """删除某个标注员的缓存任务文件，下次访问时重新生成。"""
    annotator_id = annotator_id.strip()
    if not annotator_id:
        raise HTTPException(status_code=400, detail="annotator_id 不能为空")
    safe_name = "".join(c if c.isalnum() or c in "-_" else "_" for c in annotator_id)
    task_file = _ANNOTATION_OUTPUT / f"tasks_{safe_name}.json"
    if task_file.exists():
        task_file.unlink()
        return JSONResponse({"ok": True, "message": f"已删除 {task_file.name}，下次访问将重新生成任务"})
    return JSONResponse({"ok": True, "message": "缓存文件不存在，无需删除"})


if __name__ == "__main__":
    import uvicorn

    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
