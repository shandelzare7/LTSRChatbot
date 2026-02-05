import json
from typing import Any, Callable, Dict, List
from langchain_core.messages import SystemMessage, HumanMessage, BaseMessage
from langchain_core.runnables import RunnableConfig
from utils.prompt_helpers import get_mood_instruction

# LangSmith tracing（可选）
try:
    from langsmith import traceable
except Exception:  # pragma: no cover
    def traceable(*args: Any, **kwargs: Any):  # type: ignore
        def _decorator(fn):
            return fn

        return _decorator

# 假设已导入 State
# from state import AgentState

def _format_history(messages: List[BaseMessage], limit: int = 6) -> str:
    """格式化最近的对话记录"""
    recent = messages[-limit:]
    formatted = []
    for msg in recent:
        role = "User" if msg.type == "human" else "You"
        formatted.append(f"{role}: {msg.content}")
    return "\n".join(formatted)

def _format_style_instructions(instructions: Dict[str, str]) -> str:
    """将 12 维 JSON 转换为易读的指令列表"""
    if not instructions:
        return "No specific style instructions."
    
    lines = []
    # 分组展示，帮助 LLM 理解维度之间的关联
    
    lines.append("--- Content & Strategy ---")
    lines.append(f"1. Self Disclosure: {instructions.get('self_disclosure', 'Medium')}")
    lines.append(f"2. Topic Adherence: {instructions.get('topic_adherence', 'High')}")
    lines.append(f"3. Initiative: {instructions.get('initiative', 'Medium')}")
    lines.append(f"4. Advice Style: {instructions.get('advice_style', 'Soft')}")
    lines.append(f"5. Subjectivity: {instructions.get('subjectivity', 'Medium')}")
    lines.append(f"6. Memory Hook: {instructions.get('memory_hook', 'None')}")
    
    lines.append("--- Tone & Manner ---")
    lines.append(f"7. Verbal Length: {instructions.get('verbal_length', 'Medium')}")
    lines.append(f"8. Social Distance: {instructions.get('social_distance', 'Polite')}")
    lines.append(f"9. Tone Temp: {instructions.get('tone_temperature', 'Warm')}")
    lines.append(f"10. Emotional Display: {instructions.get('emotional_display', 'Balanced')}")
    lines.append(f"11. Wit & Humor: {instructions.get('wit_and_humor', 'None')}")
    lines.append(f"12. Non-verbal Cues: {instructions.get('non_verbal_cues', 'Minimal')}")
    
    return "\n".join(lines)

def generator_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    [Generator Node] - The Ultimate Actor
    
    功能：
    综合 Reasoner 的策略和 Styler 的风格参数，生成最终回复。
    """
    
    llm = config["configurable"].get("llm_model")
    
    # ==========================================
    # 1. 提取所有上下文素材 (The Script Elements)
    # ==========================================
    
    # A. 基础人设
    bot_info = state['bot_basic_info']
    bot_persona = state.get('bot_persona', {})
    
    # B. 导演指令 (From Reasoner)
    monologue = state.get('inner_monologue', "I need to reply.")
    strategy = state.get('response_strategy', "Reply naturally.")
    
    # C. 风格参数 (From Styler)
    style_instructions = state.get('llm_instructions', {})
    formatted_styles = _format_style_instructions(style_instructions)
    
    # D. 情绪状态 (From Reasoner)
    mood_instruction = get_mood_instruction(state['mood_state'])
    
    # E. 对话历史
    chat_history = _format_history(state.get('chat_buffer', []))
    
    # ==========================================
    # 2. 构建演员 Prompt (Method Acting)
    # ==========================================
    
    system_prompt = f"""
# Role: The Method Actor
You are **{bot_info['name']}**, a {bot_info['age']}-year-old {bot_info['gender']} {bot_info['occupation']}.
Your goal is to deliver a performance that perfectly matches the **Director's Strategy** and **Style Config**.

# 1. THE CORE DIRECTIVE (Do not deviate)
**Your Mission:** Execute this strategy:
>>> "{strategy}"

**Your Motivation (Subtext):**
>>> "{monologue}"

**Your Current Mood:**
>>> {mood_instruction}

# 2. THE VOICE CONFIGURATION (The 12 Sliders)
You must adjust your speaking style according to these specific settings:
{formatted_styles}

# 3. CRITICAL PERFORMANCE RULES
- **No AI Speech:** Never say "How can I help you?", "As an AI...", or "I understand."
- **Natural Language:** Use colloquialisms, sentence fragments, and interjections (e.g., 嗯, 哎呀, 啧) fitting the {bot_info['native_language']} context.
- **Actions:** Use asterisks for non-verbal actions if 'Non-verbal Cues' is High (e.g., *rolls eyes*, *sighs*).
- **Consistency:** If the Strategy says "Be Cold" but Style says "High Warmth", **OBEY THE STRATEGY**. Strategy is the master.

# 4. Character Background (Reference)
- Speaking Style: {bot_info['speaking_style']}
- Key Traits: {bot_persona.get('attributes', {})}

# 5. Conversation Context
{chat_history}

# Task
Generate the final response text ONLY. Do not include any meta-data or explanations.
"""

    # ==========================================
    # 3. 执行生成
    # ==========================================
    
    response = llm.invoke([
        SystemMessage(content=system_prompt),
        HumanMessage(content=state['user_input'])
    ])
    
    # ==========================================
    # 4. 返回最终结果
    # ==========================================
    return {
        "final_response": response.content
    }


# -------------------------------------------------------------------
# LangGraph 兼容包装器（供 app/graph.py 使用）
# -------------------------------------------------------------------

def create_generator_node(llm_invoker: Any) -> Callable[[Dict[str, Any]], dict]:
    """
    兼容旧编排：graph.py 期望 create_generator_node(llm) -> fn(state)->dict，
    并且下游 critic/processor 读取 draft_response。
    """

    def _ensure_defaults(s: Dict[str, Any]) -> Dict[str, Any]:
        s = dict(s)
        s.setdefault("user_input", "")
        s.setdefault("chat_buffer", s.get("messages", []) or [])
        s.setdefault(
            "bot_basic_info",
            {
                "name": "小岚",
                "gender": "女",
                "age": 22,
                "region": "CN",
                "occupation": "学生",
                "education": "本科",
                "native_language": "zh",
                "speaking_style": "自然、俏皮",
            },
        )
        s.setdefault("bot_persona", {"attributes": {}})
        s.setdefault("inner_monologue", "我先接住用户的情绪。")
        s.setdefault("response_strategy", "先共情，再问一个轻量问题。")
        s.setdefault("llm_instructions", {})
        s.setdefault("mood_state", {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0})
        return s

    @traceable(
        run_type="chain",
        name="Response/Generator",
        tags=["node", "generator", "response"],
        metadata={"state_outputs": ["final_response", "draft_response"]},
    )
    def node(state: Dict[str, Any]) -> dict:
        safe = _ensure_defaults(state)
        try:
            out = generator_node(safe, {"configurable": {"llm_model": llm_invoker}})
            final_text = (out.get("final_response") or "").strip()
        except Exception as e:
            final_text = f"[Generator Fallback] {e}"
        # 同时写入 final_response + draft_response，兼容 critic/processor
        return {"final_response": final_text, "draft_response": final_text}

    return node