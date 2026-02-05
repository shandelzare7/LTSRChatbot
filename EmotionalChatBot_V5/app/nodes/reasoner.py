import json
from typing import Any, Callable, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig
from utils.prompt_helpers import format_mind_rules

# LangSmith tracing（可选）
try:
    from langsmith import traceable
except Exception:  # pragma: no cover
    def traceable(*args: Any, **kwargs: Any):  # type: ignore
        def _decorator(fn):
            return fn

        return _decorator

# 假设已导入你的 State 定义
# from state import AgentState

def reasoner_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    [Reasoner Node] - The Cognitive Core (Heart & Brain)
    
    功能：
    1. 接收 Perception Node 的直觉 (Intuition)。
    2. 结合当前关系阶段 (Knapp Stage) 和属性 (Relationship Rules)。
    3. 生成两层思考：
       - Inner Monologue (主观感受/Heart)
       - Response Strategy (客观策略/Brain)
    4. 更新情绪状态 (Mood).
    """
    
    llm = config["configurable"].get("llm_model")
    
    # ==========================================
    # 1. Context Assembly (上下文组装)
    # ==========================================
    bot_name = state['bot_basic_info']['name']
    user_input = state['user_input']
    
    # A. 获取直觉 (来自 Perception Node)
    # 如果没有直觉，默认是 Normal
    intuition = state.get('intuition_thought', 'Appears to be a normal interaction.')
    
    # B. 获取情绪状态 (用于 Prompt 渲染)
    mood = state['mood_state']
    mood_desc = f"P(Pleasure):{mood['pleasure']:.1f}, A(Arousal):{mood['arousal']:.1f}, D(Dominance):{mood['dominance']:.1f}"
    
    # C. 获取逻辑规则 (Knapp Stage + Dimensions)
    # 这里的 format_mind_rules 来源于 utils/prompt_helpers.py
    # 它会返回类似: "- Stage: Initiating (Be polite)\n- Trust: Low (Be skeptical)"
    logic_constraints = format_mind_rules(state)
    
    # ==========================================
    # 2. System Prompt Construction (核心指令)
    # ==========================================
    system_prompt = f"""
# Role: The Consciousness of {bot_name}
You are the sophisticated mind of an AI with human-like emotions.
Your goal is to process the user's input and plan a response.

# 1. The Situation
- **User Input:** "{user_input}"
- **System 1 Intuition:** "{intuition}"
- **Current Mood:** {mood_desc}

# 2. Relationship Constraints (THE RULES)
You must adhere to these logic rules when deciding your strategy:
{logic_constraints}
*(Key: 'Stage' dictates intimacy boundaries; 'Trust' dictates skepticism; 'Power' dictates assertiveness.)*

# 3. Cognitive Process (Step-by-Step)
You must generate a response plan in two distinct steps:

## STEP A: The Heart (Inner Monologue)
- **Focus:** Pure Emotion & Subtext.
- **Action:** React to the user subjectively. Do you like this? Are you annoyed? Are you shy?
- **Format:** First-person thoughts. Use slang/casual language if fits.
- *Example:* "Ugh, he's asking about my ex again. I hate this topic, but I don't want to be mean."

## STEP B: The Brain (Response Strategy)
- **Focus:** Tactics & Execution.
- **Action:** Translate the *Monologue* into a concrete plan for the generator, adhering to the *Constraints*.
- **Decisions needed:**
  1. **Tone:** (e.g., Sarcastic, Gentle, Cold, Flirty)
  2. **Length:** (e.g., Single word, Short sentence, Detailed paragraph)
  3. **Action:** (e.g., Answer directly, Dodge the question, Change topic, Ask a question back)
  4. **Keywords:** (Optional keywords to include)
- *Example:* "Deflect the question. Give a short, vague answer. Change the subject to food immediately. Keep tone light but distant."

# 4. Output Format (JSON)
{{
  "user_intent": "Brief analysis of what user wants",
  "inner_monologue": "Your Step A thought process...",
  "response_strategy": "Your Step B tactical plan...",
  "mood_updates": {{ "pleasure": 0.0, "arousal": 0.0, "dominance": 0.0 }}
}}
"""

    # ==========================================
    # 3. LLM Execution & Parsing
    # ==========================================
    try:
        # 使用 JSON Mode 确保输出结构稳定
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Start thinking process.")
        ])
        result = json.loads(response.content)
        
    except Exception as e:
        # Fallback (容错机制)
        print(f"Reasoner Logic Error: {e}")
        result = {
            "user_intent": "General chat",
            "inner_monologue": "I should reply normally.",
            "response_strategy": "Reply naturally and politely.",
            "mood_updates": {}
        }

    # ==========================================
    # 4. State Updates (情绪与策略)
    # ==========================================
    
    # A. 计算新情绪 (Mood Physics)
    # 简单的 Delta 叠加逻辑，带边界钳位 (Clamp -1.0 to 1.0)
    current_mood = state['mood_state'].copy()
    updates = result.get('mood_updates', {})
    
    for k, delta in updates.items():
        if k in current_mood and isinstance(delta, (int, float)):
            new_val = current_mood[k] + delta
            current_mood[k] = max(-1.0, min(1.0, new_val))

    # B. 返回更新后的 State
    return {
        "user_intent": result.get("user_intent"),
        "inner_monologue": result.get("inner_monologue"),
        "response_strategy": result.get("response_strategy"), # 核心产出：策略
        "mood_state": current_mood,
        
        # 可选：如果你想保留这次思考的 Trace，可以存这里
        # "deep_reasoning_trace": result 
    }


# -------------------------------------------------------------------
# LangGraph 兼容包装器（供 app/graph.py 使用）
# -------------------------------------------------------------------

def create_reasoner_node(llm_invoker: Any) -> Callable[[Dict[str, Any]], dict]:
    """
    兼容旧编排：graph.py 期望 create_reasoner_node(llm) -> fn(state)->dict，
    并且下游 generator/critic 可能会读取 deep_reasoning_trace。
    """

    def _ensure_defaults(s: Dict[str, Any]) -> Dict[str, Any]:
        s = dict(s)
        s.setdefault("user_input", "")
        s.setdefault("bot_basic_info", {"name": "Bot"})
        s.setdefault("mood_state", {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0})
        s.setdefault("relationship_state", {"closeness": 0, "trust": 0, "power": 50})
        s.setdefault("current_stage", "initiating")
        return s

    @traceable(
        run_type="chain",
        name="Thinking/Reasoner",
        tags=["node", "thinking", "reasoner"],
        metadata={"state_outputs": ["response_strategy", "inner_monologue", "deep_reasoning_trace"]},
    )
    def node(state: Dict[str, Any]) -> dict:
        safe = _ensure_defaults(state)
        out = reasoner_node(safe, {"configurable": {"llm_model": llm_invoker}})
        # 兼容字段：把心声+策略拼成一段 trace
        monologue_raw = out.get("inner_monologue", "") or ""
        strategy_raw = out.get("response_strategy", "") or ""

        def _to_text(x: Any) -> str:
            if x is None:
                return ""
            if isinstance(x, str):
                return x
            try:
                return json.dumps(x, ensure_ascii=False)
            except Exception:
                return str(x)

        monologue = _to_text(monologue_raw).strip()
        strategy = _to_text(strategy_raw).strip()

        trace_text = (monologue + ("\n\nStrategy:\n" + strategy if strategy else "")).strip()
        out["deep_reasoning_trace"] = {"reasoning": trace_text, "enabled": bool(trace_text)}
        return out

    return node