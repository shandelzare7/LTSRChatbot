"""
EmotionalChatBot V5.0 启动入口
支持控制台多轮对话；每次会话的 log 写入按时间命名的日志文件（不覆盖）。
"""
import os
import sys
from pathlib import Path
from datetime import datetime

# 加载 .env（若存在）
root = Path(__file__).resolve().parent
if str(root) not in sys.path:
    sys.path.insert(0, str(root))
try:
    from utils.env_loader import load_project_env
    load_project_env(root)
except Exception:
    pass

from langchain_core.messages import HumanMessage
from app.core.mode_base import PsychoMode
from app.graph import build_graph
from app.state import AgentState
from utils.yaml_loader import get_project_root, load_modes_from_dir


class TeeWriter:
    """同时写入控制台和日志文件，用于会话说明、你/Bot 对话等。"""
    def __init__(self, file_handle, original_stdout):
        self._file = file_handle
        self._out = original_stdout

    def write(self, s: str):
        if self._file:
            try:
                self._file.write(s)
                self._file.flush()
            except OSError:
                pass
        if self._out:
            self._out.write(s)

    def flush(self):
        if self._file:
            try:
                self._file.flush()
            except OSError:
                pass
        if self._out:
            self._out.flush()


def get_default_mode() -> PsychoMode:
    """加载默认模式（normal_mode）。"""
    proot = get_project_root()
    modes_dir = proot / "config" / "modes"
    raw = load_modes_from_dir(modes_dir)
    for m in raw:
        if m.get("id") == "normal_mode":
            return PsychoMode.model_validate(m)
    # Fallback: 如果找不到 normal_mode.yaml，创建一个最小化的默认模式
    from app.core.mode_base import BehaviorContract, LatsBudget, RequirementsPolicy, CriticCriteria, StyleBias
    return PsychoMode(
        id="normal_mode",
        name="Normal",
        description="默认模式",
        behavior_contract=BehaviorContract(),
        lats_budget=LatsBudget(),
        requirements_policy=RequirementsPolicy(),
        critic_criteria=CriticCriteria(),
        style_bias=StyleBias(),
        disallowed=[],
    )


def _make_initial_state(user_id: str, bot_id: str) -> AgentState:
    """构造每轮可复用的初始 state 骨架（只差 messages / current_time）。"""
    default_mode = get_default_mode()
    now = datetime.now().isoformat()
    return {
        "messages": [],
        "current_time": now,
        "user_id": user_id,
        "bot_id": bot_id,
        "current_mode": default_mode,
        "user_profile": {},
        "memories": "",
        "deep_reasoning_trace": {},
        "style_analysis": "",
        "draft_response": "",
        "critique_feedback": "",
        "retry_count": 0,
        "final_segments": [],
        "final_delay": 0.0,
        # LATS defaults:
        # Let the LATS node pick stage-aware rollouts/expand_k by default.
        # (So initiating/experimenting won't be forced into an overly small fixed budget.)
        # 默认启用：否则你新增的 LLM 逐条对齐/记忆一致性/关系拟人评审不会运行
        "lats_enable_llm_soft_scorer": (str(os.getenv("LATS_ENABLE_LLM_SOFT_SCORER", "1")).lower() not in ("0", "false", "no", "off")),
    }


def _open_session_log():
    """在项目 logs 目录下创建按时间命名的日志文件（不覆盖），返回 (文件句柄, 路径)。"""
    proot = get_project_root()
    log_dir = proot / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    ts = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    path = log_dir / f"chat_{ts}.log"
    f = open(path, "w", encoding="utf-8")
    return f, path


class FileOnlyWriter:
    """仅写入日志文件，不输出到控制台。用于 graph 内部节点 log（Detection、Reasoner 等）。"""
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


async def run_console_chat_async(
    user_id: str = "user_console_demo",
    bot_id: str = "default_bot",
):
    """控制台多轮对话（async）：用 ainvoke 跑图；所有 log 同时写入按时间命名的日志文件。"""
    import asyncio
    from app.nodes.loader import _get_db_manager

    loop = asyncio.get_running_loop()
    app = build_graph()

    log_file, log_path = _open_session_log()
    original_stdout = sys.stdout
    tee = TeeWriter(log_file, original_stdout)
    sys.stdout = tee

    def log_line(msg: str):
        """写一行到日志文件并打印到控制台。"""
        print(msg)

    # 每次启动控制台视为「重新开始会话」：
    # - 过去这里只清 messages，但 summary/transcripts/notes 仍会被召回，导致“助手模板记忆”反复污染生成
    # - 默认清空全部记忆资产（可用环境变量关闭）
    clear_all = str(os.getenv("CONSOLE_CLEAR_ALL_MEMORY_ON_START", "1")).lower() not in ("0", "false", "no", "off")
    db = _get_db_manager()
    if db:
        try:
            if clear_all and hasattr(db, "clear_all_memory_for"):
                counts = await db.clear_all_memory_for(user_id, bot_id, reset_profile=True)
                log_line(f"[Session] 已清空全部记忆资产（DB）: {counts}，当前为全新对话。")
            else:
                n = await db.clear_messages_for(user_id, bot_id)
                if n > 0:
                    log_line(f"[Session] 已清空本会话历史（共 {n} 条），当前为全新对话。")
        except Exception as e:
            log_line(f"[Session] 清空历史失败（继续运行）: {e}")
    else:
        # 无 DB 时清理本地 local_data（可用环境变量关闭）
        if clear_all:
            try:
                from app.core.local_store import LocalStoreManager
                store = LocalStoreManager()
                ok = store.clear_relationship(user_id, bot_id)
                if ok:
                    log_line("[Session] 已清空全部记忆资产（LocalStore），当前为全新对话。")
            except Exception as e:
                log_line(f"[Session] 清空本地记忆失败（继续运行）: {e}")

    try:
        log_line("=" * 50)
        log_line("EmotionalChatBot V5.0 控制台对话")
        log_line(f"   user_id: {user_id}   bot_id: {bot_id}")
        log_line("   输入一行内容回车发送，空行或 Ctrl+C 退出")
        log_line(f"   日志文件: {log_path}")
        log_line("=" * 50)

        while True:
            try:
                line = await loop.run_in_executor(None, lambda: input("\n你: ").strip())
            except (KeyboardInterrupt, EOFError):
                log_line("\n再见。")
                break
            if not line:
                log_line("（空输入退出）")
                break

            now_iso = datetime.now().isoformat()
            log_line("")
            log_line(f"[{now_iso}] === 你: {line}")
            log_line("-" * 40)

            state = _make_initial_state(user_id, bot_id)
            state["messages"] = [HumanMessage(content=line, additional_kwargs={"timestamp": now_iso})]
            state["current_time"] = now_iso

            try:
                # graph 内部所有 print 只写日志文件，不输出到控制台
                sys.stdout = FileOnlyWriter(log_file)
                try:
                    result = await app.ainvoke(state, config={"recursion_limit": 50})
                finally:
                    sys.stdout = tee
            except Exception as e:
                log_line(f"Bot: [出错] {e}")
                continue

            reply = result.get("final_response") or ""
            if not reply and result.get("final_segments"):
                reply = " ".join(result["final_segments"])
            if not reply:
                reply = result.get("draft_response") or "（无回复）"

            log_line(f"=== Bot: {reply}")
            log_line("")
            print("Bot:", reply)
    finally:
        sys.stdout = original_stdout
        try:
            log_file.close()
            print(f"日志已保存: {log_path}")
        except OSError:
            pass


def run_console_chat(
    user_id: str = "user_console_demo",
    bot_id: str = "default_bot",
):
    """入口：在单一事件循环内跑多轮对话。"""
    import asyncio
    asyncio.run(run_console_chat_async(user_id=user_id, bot_id=bot_id))


# 默认使用数据库里已 seed 的 bot/user，保证加载真人设；无 DB 时会 get-or-create
DEFAULT_USER_ID = "local_user_5128d1c1"
DEFAULT_BOT_ID = "4d803b5a-cb30-4d14-89eb-88d259564610"


if __name__ == "__main__":
    user_id = os.environ.get("USER_ID", DEFAULT_USER_ID)
    bot_id = os.environ.get("BOT_ID", DEFAULT_BOT_ID)
    if "--gui" in sys.argv or "-g" in sys.argv:
        from main_gui import run_gui
        run_gui(user_id=user_id, bot_id=bot_id)
    else:
        run_console_chat(user_id=user_id, bot_id=bot_id)
