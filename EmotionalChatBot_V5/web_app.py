"""
FastAPI Web Application for EmotionalChatBot V5.0
支持通过 Web 界面与 Chatbot 对话
"""
import os
import sys
from pathlib import Path
from datetime import datetime, timezone
import time
import io
import zipfile
from typing import Optional

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
from fastapi.responses import HTMLResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage

from app.graph import build_graph
from app.core.database import DBManager, Bot, User, Message
from app.web.session import (
    create_session,
    get_session,
    delete_session,
    generate_user_id_from_request,
)
from main import _make_initial_state
from sqlalchemy import select, case
from utils.yaml_loader import get_project_root
import sys

# 初始化 FastAPI 应用
app = FastAPI(title="EmotionalChatBot Web", version="5.0")

# 日志文件管理
_log_files: dict[str, tuple] = {}  # {session_id: (file_handle, path)}


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
        now_iso = datetime.now().isoformat()
        
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
    Note: Render filesystem is ephemeral; files exist only on the running instance.
    """
    _require_admin(request)
    n = max(1, min(int(n or 2), 10))

    log_dir = _get_log_dir()
    files = sorted(log_dir.glob("web_chat_*.log"), key=lambda x: x.stat().st_mtime, reverse=True)[:n]
    if not files:
        raise HTTPException(status_code=404, detail="No logs found")

    mem = io.BytesIO()
    with zipfile.ZipFile(mem, mode="w", compression=zipfile.ZIP_DEFLATED) as z:
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


# Pydantic 模型
class ChatRequest(BaseModel):
    message: str


class SessionInitRequest(BaseModel):
    bot_id: str


# 全局变量
_graph = None
_db_manager = None


def get_graph():
    """懒加载 graph"""
    global _graph
    if _graph is None:
        _graph = build_graph()
    return _graph


def get_db_manager():
    """懒加载 DBManager"""
    global _db_manager
    if _db_manager is None:
        if not os.getenv("DATABASE_URL"):
            raise RuntimeError("DATABASE_URL 未设置")
        _db_manager = DBManager.from_env()
    return _db_manager


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
        t_total = time.perf_counter()
        # 加载数据库状态
        db = get_db_manager()
        t0 = time.perf_counter()
        db_state = await db.load_state(user_id, bot_id)
        t_load_ms = (time.perf_counter() - t0) * 1000.0
        
        # 构建AgentState
        t0 = time.perf_counter()
        state = _make_initial_state(user_id, bot_id)
        state.update(db_state)  # 合并数据库状态
        t_state_ms = (time.perf_counter() - t0) * 1000.0
        
        # 业务语义：用户消息接收时间（进入流程之前）
        received_iso = datetime.now(timezone.utc).isoformat()
        state["messages"] = [
            HumanMessage(
                content=chat_data.message.strip(),
                additional_kwargs={"timestamp": received_iso},
            )
        ]
        state["current_time"] = received_iso
        state["user_received_at"] = received_iso
        state["user_input"] = chat_data.message.strip()
        state["external_user_text"] = state["user_input"]
        
        # Web：仅在显式设置环境变量时覆盖 LATS 配置（默认走系统原始策略/预算与评审）
        # 说明：用户可能希望在 Web 上也完整启用 LATS + LLM soft scorer。
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
        
        # 运行graph（重定向 stdout 到日志文件）
        log_file, log_path = get_or_create_log_file(session_id, user_id, bot_id)
        try:
            log_file.write(
                f"[WEB_PERF] db.load_state_ms={t_load_ms:.1f} make_state_ms={t_state_ms:.1f}\n"
            )
            log_file.flush()
        except Exception:
            pass
        original_stdout = sys.stdout
        sys.stdout = FileOnlyWriter(log_file)
        
        try:
            graph = get_graph()
            t0 = time.perf_counter()
            result = await graph.ainvoke(state, config={"recursion_limit": 50})
            t_graph_ms = (time.perf_counter() - t0) * 1000.0
        finally:
            sys.stdout = original_stdout
        
        # 注意：graph 末尾的 `memory_writer` 节点会负责写入 DB（Commit Late）。
        # Web 这里再写一次会导致同一轮 messages 被写入两次（历史显示重复）。
        t_save_ms = 0.0
        
        # 获取回复
        reply = result.get("final_response") or ""
        if not reply and result.get("final_segments"):
            reply = " ".join(result["final_segments"])
        if not reply:
            reply = result.get("draft_response") or "（无回复）"
        
        # 记录到日志文件
        try:
            t_total_ms = (time.perf_counter() - t_total) * 1000.0
            log_file, _ = get_or_create_log_file(session_id, user_id, bot_id)
            log_file.write(
                f"[WEB_PERF] graph_ms={t_graph_ms:.1f} save_turn_ms={t_save_ms:.1f} total_ms={t_total_ms:.1f} log={log_path}\n"
            )
            log_file.flush()
            log_web_chat(session_id, user_id, bot_id, chat_data.message.strip(), reply)
        except Exception as log_error:
            print(f"日志记录失败: {log_error}", file=sys.stderr)
        
        return {
            "reply": reply,
            "status": "success",
            # timestamps for UI
            "user_created_at": received_iso,
            "ai_created_at": (result.get("ai_sent_at") if isinstance(result, dict) else None),
        }
    except Exception as e:
        import traceback
        error_detail = traceback.format_exc()
        print(f"Chat error: {error_detail}")
        raise HTTPException(status_code=500, detail=f"处理消息失败: {str(e)}")


@app.post("/api/session/reset")
async def reset_session(
    request: Request, response: Response, session_id: Optional[str] = Cookie(None)
):
    """重置会话：清空对话历史"""
    if not session_id:
        raise HTTPException(status_code=401, detail="未找到会话")
    
    session = get_session(session_id)
    if not session:
        raise HTTPException(status_code=401, detail="会话无效")
    
    try:
        user_id = session["user_id"]
        bot_id = session["bot_id"]
        
        db = get_db_manager()
        await db.clear_messages_for(user_id, bot_id)
        
        return {"status": "success", "message": "对话历史已清空"}
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
                <button id="reset-btn" class="btn-secondary">重置会话</button>
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
