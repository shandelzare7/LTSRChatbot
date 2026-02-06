"""
EmotionalChatBot V5.0 启动入口
支持控制台单轮示例；后续可接微信等。
"""
import os
from pathlib import Path
from datetime import datetime

# 加载 .env（若存在）：优先用 python-dotenv，缺失则使用内置 fallback 解析器
root = Path(__file__).resolve().parent
try:
    from utils.env_loader import load_project_env

    load_project_env(root)
except Exception:
    # 即使 env loader 异常，也不阻塞启动（只会影响 tracing/keys）
    pass

from langchain_core.messages import HumanMessage

from app.core.mode_base import PsychoMode
from app.graph import build_graph
from app.state import AgentState
from utils.yaml_loader import get_project_root, load_modes_from_dir


def get_default_mode() -> PsychoMode:
    """加载默认模式（normal_mode）。"""
    root = get_project_root()
    modes_dir = root / "config" / "modes"
    raw = load_modes_from_dir(modes_dir)
    for m in raw:
        if m.get("id") == "normal_mode":
            return PsychoMode.model_validate(m)
    return PsychoMode(
        id="normal_mode",
        name="正常模式",
        trigger_description="用户语气平和",
        system_prompt_template="你是一个陪伴型 Bot。",
        monologue_instruction="理性分析用户意图。",
        critic_criteria=["回复自然"],
        split_strategy="normal",
        typing_speed_multiplier=1.0,
    )


def run_console_example():
    """单轮控制台示例：构建图并跑一条消息。"""
    app = build_graph()
    default_mode = get_default_mode()

    initial_state: AgentState = {
        "messages": [HumanMessage(content="你好，今天心情不太好，想和你聊聊。")],
        # 为了让宏观门控可控：控制台示例固定在白天，避免“刚好在睡觉”导致长延迟看起来像卡住
        "current_time": datetime.now().replace(hour=12, minute=0, second=0, microsecond=0).isoformat(),
        "user_id": "user_console_demo",
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
    }

    print("=" * 50)
    print("EmotionalChatBot V5.0 控制台示例")
    print("=" * 50)
    print("输入:", initial_state["messages"][0].content)
    print()

    result = app.invoke(initial_state, config={"recursion_limit": 50})

    mode = result.get("current_mode")
    print("当前模式:", getattr(mode, "name", mode))
    print("最终回复片段:", result.get("final_segments", []))
    print("最终延迟系数:", result.get("final_delay"))
    print()
    if result.get("final_segments"):
        print("完整回复:", " ".join(result["final_segments"]))
    print("=" * 50)


if __name__ == "__main__":
    run_console_example()
