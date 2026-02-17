"""
FastAPI Web Application for EmotionalChatBot V5.0
支持通过 Web 界面与 Chatbot 对话
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import time
import asyncio
import io
import zipfile
from typing import Optional
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
from app.core.database import DBManager, Bot, User, Message, WebChatLog
from app.web.session import (
    create_session,
    get_session,
    delete_session,
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


# Push subscription payload
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
    """Web fast-return graph: end at final_validator (no tail writes)."""
    global _graph_fast
    if _graph_fast is None:
        _graph_fast = build_graph(entry_point="loader", end_at="final_validator")
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
    sql = """
    CREATE TABLE IF NOT EXISTS push_subscriptions (
      session_id TEXT PRIMARY KEY,
      user_external_id TEXT,
      bot_id UUID,
      subscription JSONB NOT NULL,
      created_at TIMESTAMPTZ DEFAULT NOW(),
      updated_at TIMESTAMPTZ DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_push_subscriptions_user_bot ON push_subscriptions(user_external_id, bot_id);
    """
    async with db.engine.connect() as conn:
        await conn.execute(text(sql))
        await conn.commit()


async def _upsert_push_subscription(
    db: DBManager, *, session_id: str, user_external_id: str, bot_id: str, subscription: dict
) -> None:
    import json

    await _ensure_push_schema(db)
    sql = """
    INSERT INTO push_subscriptions(session_id, user_external_id, bot_id, subscription, updated_at)
    VALUES (:session_id, :user_external_id, :bot_id::uuid, :sub::jsonb, NOW())
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


async def _delete_push_subscription(db: DBManager, *, session_id: str) -> None:
    await _ensure_push_schema(db)
    async with db.engine.connect() as conn:
        await conn.execute(text("DELETE FROM push_subscriptions WHERE session_id=:sid"), {"sid": session_id})
        await conn.commit()


async def _get_push_subscription(db: DBManager, *, session_id: str) -> Optional[dict]:
    await _ensure_push_schema(db)
    async with db.engine.connect() as conn:
        row = (await conn.execute(text("SELECT subscription FROM push_subscriptions WHERE session_id=:sid"), {"sid": session_id})).first()
    if not row:
        return None
    sub = row[0]
    return sub if isinstance(sub, dict) else None


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
async def push_subscribe(req: PushSubscribeRequest, session_id: Optional[str] = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="未找到会话")
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="会话无效或已过期")
    # Validate VAPID config early
    _get_vapid_keys()
    db = get_db_manager()
    await _upsert_push_subscription(
        db,
        session_id=session_id,
        user_external_id=str(session.get("user_id") or ""),
        bot_id=str(session.get("bot_id") or ""),
        subscription=req.subscription or {},
    )
    return {"status": "success"}


@app.post("/api/push/unsubscribe")
async def push_unsubscribe(session_id: Optional[str] = Cookie(None)):
    if not session_id:
        raise HTTPException(status_code=401, detail="未找到会话")
    db = get_db_manager()
    await _delete_push_subscription(db, session_id=session_id)
    return {"status": "success"}


@app.get("/", response_class=HTMLResponse)
async def index(request: Request, session_id: Optional[str] = Cookie(None)):
    """主入口：检查会话，返回相应页面"""
    # 检查是否有有效会话
    if session_id:
        session = get_session(session_id)
        if session:
            # 有有效会话，返回聊天界面
            return get_chat_html(session["bot_id"])
    
    # 无会话或过期，返回bot选择页面
    return get_bot_selection_html()


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
                return get_bot_selection_html()
            
            bot_id_str = str(bot.id)
        
        # 检查是否有有效会话且bot匹配
        if session_id:
            existing_session = get_session(session_id)
            if existing_session and existing_session["bot_id"] == bot_id_str:
                # 已有匹配的会话，直接返回聊天界面
                return get_chat_html(bot_id_str)
        
        # 创建新会话
        user_id = generate_user_id_from_request(request)
        new_session_id = create_session(user_id, bot_id_str)
        
        # 设置Cookie
        response.set_cookie(
            key="session_id",
            value=new_session_id,
            httponly=True,
            secure=os.getenv("ENVIRONMENT") == "production",
            samesite="lax",
            max_age=86400 * 7,
        )
        
        # 返回聊天界面
        return get_chat_html(bot_id_str)
    except Exception as e:
        # 出错时返回选择页面
        return get_bot_selection_html()


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
        # 生成或获取user_id
        user_id = generate_user_id_from_request(request)
        
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
        
        # 创建会话
        session_id = create_session(user_id, bot_id)
        
        # 设置Cookie
        response.set_cookie(
            key="session_id",
            value=session_id,
            httponly=True,
            secure=os.getenv("ENVIRONMENT") == "production",  # 生产环境启用
            samesite="lax",
            max_age=86400 * 7,  # 7天
        )
        
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


@app.post("/api/chat")
async def chat(
    request: Request,
    chat_data: ChatRequest,
    session_id: Optional[str] = Cookie(None),
):
    """处理聊天消息"""
    # 验证会话
    if not session_id:
        raise HTTPException(status_code=401, detail="未找到会话，请先选择bot")
    
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="会话无效或已过期，请重新选择bot")
    
    user_id = session["user_id"]
    bot_id = session["bot_id"]
    
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

            # Cancel in-flight tasks (generation and tail settlement)
            for k in ("fast_task", "tail_task"):
                t = inflight.get(k)
                if t is not None and hasattr(t, "done") and (not t.done()):
                    try:
                        t.cancel()
                    except Exception:
                        pass

            async def _run_one(req_id_local: str):
                t_total = time.perf_counter()
                # Snapshot pending messages for this run
                async with inflight["lock"]:
                    pending = list(inflight.get("pending_user_msgs") or [])

                if not pending:
                    return

                merged_text = "\n".join([str(x.get("content") or "").strip() for x in pending if str(x.get("content") or "").strip()]).strip()
                if not merged_text:
                    return

                # Load fresh state (includes any just-appended user messages)
                t0 = time.perf_counter()
                db_state = await db.load_state(user_id, bot_id)
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
                try:
                    fast_return = _truthy(os.getenv("WEB_FAST_RETURN", "1"))
                    graph = get_graph_fast() if fast_return else get_graph()
                    t0 = time.perf_counter()
                    result = await graph.ainvoke(state, config={"recursion_limit": 50})
                    t_graph_ms = (time.perf_counter() - t0) * 1000.0
                finally:
                    sys.stdout = original_stdout

                reply = result.get("final_response") or ""
                if not reply and result.get("final_segments"):
                    reply = " ".join(result["final_segments"])
                if not reply:
                    reply = result.get("draft_response") or "（无回复）"
                
                # 优先使用 humanized_output.segments（包含 delay 信息），否则回退到 final_segments（字符串数组）
                humanized = result.get("humanized_output") or {}
                segments_with_delay = humanized.get("segments") or []
                if segments_with_delay and isinstance(segments_with_delay, list) and len(segments_with_delay) > 0:
                    # 使用带 delay 的 segments（每个元素是 {"content": "...", "delay": 1.2, "action": "typing"}）
                    segments = segments_with_delay
                else:
                    # 回退：使用 final_segments（字符串数组），转换为带默认 delay 的对象数组
                    segments_raw = result.get("final_segments") or []
                    if not segments_raw and reply:
                        segments_raw = [reply]
                    segments = []
                    for i, seg in enumerate(segments_raw):
                        if isinstance(seg, str):
                            # 字符串：转换为对象，第一条 delay=0（前端会立即显示），后续使用默认 delay
                            segments.append({
                                "content": seg,
                                "delay": 0.0 if i == 0 else 0.8,  # 第一条 delay=0（前端不应用），后续默认 0.8 秒
                                "action": "typing"
                            })
                        elif isinstance(seg, dict) and "content" in seg:
                            # 已经是对象格式，直接使用（前端会正确处理：第一条立即显示，后续只对 typing 应用 delay）
                            segments.append(seg)
                        else:
                            # 其他格式，转换为字符串
                            segments.append({
                                "content": str(seg),
                                "delay": 0.0 if i == 0 else 0.8,
                                "action": "typing"
                            })

                ai_sent_at = datetime.now(timezone.utc).isoformat(timespec="milliseconds")
                if isinstance(result, dict):
                    result["ai_sent_at"] = ai_sent_at

                # Tail settlement: skip writing user message (already persisted individually above)
                if _truthy(os.getenv("WEB_FAST_RETURN", "1")):
                    tail_state = dict(state)
                    if isinstance(result, dict):
                        tail_state.update(result)
                    tail_state["ai_sent_at"] = ai_sent_at
                    tail_state["skip_user_message_write"] = True

                    async def _run_tail_local():
                        log_file2, _ = get_or_create_log_file(session_id, user_id, bot_id)
                        orig = sys.stdout
                        sys.stdout = FileOnlyWriter(log_file2)
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
                            sys.stdout = orig

                    try:
                        tail_task = asyncio.create_task(_run_tail_local())
                        async with inflight["lock"]:
                            inflight["tail_task"] = tail_task
                    except Exception:
                        pass

                # Write perf + persist web_chat log snapshot
                try:
                    t_total_ms = (time.perf_counter() - t_total) * 1000.0
                    log_file3, _ = get_or_create_log_file(session_id, user_id, bot_id)
                    log_file3.write(f"[WEB_PERF] graph_ms={t_graph_ms:.1f} total_ms={t_total_ms:.1f} log={log_path}\n")
                    log_file3.flush()
                    log_web_chat(session_id, user_id, bot_id, merged_text, reply)

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

                # Complete waiter only if still latest
                async with inflight["lock"]:
                    if inflight.get("latest_req_id") != req_id_local:
                        return
                    # consume pending messages
                    inflight["pending_user_msgs"] = []
                    w = inflight.get("waiter")
                    if w is not None and (not w.done()):
                        # Web Push (best-effort): send once per bot turn.
                        try:
                            sub = await _get_push_subscription(db, session_id=session_id)
                            if isinstance(sub, dict) and sub:
                                bot_name = ""
                                try:
                                    bot_name = str((state.get("bot_basic_info") or {}).get("name") or "")  # type: ignore[union-attr]
                                except Exception:
                                    bot_name = ""
                                title = bot_name or "Chatbot"
                                body = str(segments[0] if segments else reply)[:200]
                                await _send_web_push(subscription=sub, title=title, body=body, url="/", tag="ltsr-bot-message")
                        except Exception:
                            # never block chat on push failures
                            pass
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

            inflight["fast_task"] = asyncio.create_task(_run_one(req_id))

        # 3) Wait for reply or superseded
        out = await waiter
        if isinstance(out, dict) and out.get("status") == "superseded":
            # Not an error: this request was intentionally superseded by a newer user message.
            # Frontend should ignore this response and wait for the newer request's result.
            return JSONResponse(status_code=200, content={"status": "superseded"})
        return out
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
    request: Request, response: Response, session_id: Optional[str] = Cookie(None)
):
    """清空该用户在当前 bot 下的所有对话与记忆（数据库/本地存储）。"""
    if not session_id:
        raise HTTPException(status_code=401, detail="未找到会话")
    
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="会话无效")
    
    try:
        user_id = session["user_id"]
        bot_id = session["bot_id"]

        # 1) Prefer DB: delete messages + memories/transcripts/notes + reset summary/stage
        try:
            db = get_db_manager()
            counts = await db.clear_all_memory_for(user_id, bot_id, reset_profile=True)
            return {"status": "success", "message": "对话历史已清空", "deleted": counts}
        except Exception:
            pass

        # 2) Fallback: local store (no DATABASE_URL)
        try:
            from app.core.local_store import LocalStoreManager

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
    limit: int = 2000,
):
    """获取当前用户在该 bot 下的全部对话历史（按时间升序）。"""
    if not session_id:
        raise HTTPException(status_code=401, detail="未找到会话，请先选择bot")

    sess = get_session(session_id)
    if not sess:
        raise HTTPException(status_code=401, detail="会话无效或已过期，请重新选择bot")

    bot_id_str = sess["bot_id"]
    user_external_id = sess["user_id"]

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
async def get_session_status(session_id: Optional[str] = Cookie(None)):
    """获取会话状态"""
    if not session_id:
        return {"has_session": False}
    
    session = get_session(session_id)
    if not session:
        return {"has_session": False}

    bot_id_str = session["bot_id"]
    user_external_id = session["user_id"]

    bot_name = None
    bot_basic_info = {}
    has_history = False

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

                # 是否有历史消息（不使用 _get_or_create_user，避免状态查询意外写入）
                result = await db_session.execute(
                    select(User).where(
                        User.bot_id == bot.id,
                        User.external_id == user_external_id,
                    )
                )
                user = result.scalar_one_or_none()
                if user:
                    result = await db_session.execute(
                        select(Message.id).where(Message.user_id == user.id).limit(1)
                    )
                    has_history = result.scalar_one_or_none() is not None
    except Exception:
        # 状态接口尽量不因数据库异常影响页面；前端会降级显示通用开场白
        pass

    return {
        "has_session": True,
        "bot_id": bot_id_str,
        "user_id": user_external_id,
        "bot_name": bot_name,
        "bot_basic_info": bot_basic_info,
        "has_history": has_history,
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


# HTML 模板函数
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
                <h2>💬 对话中</h2>
                <div style="display:flex; gap:8px; align-items:center;">
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
                <button id="send-btn" class="btn-primary">发送</button>
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


if __name__ == "__main__":
    import uvicorn
    
    port = int(os.getenv("PORT", 8000))
    uvicorn.run(app, host="0.0.0.0", port=port)
