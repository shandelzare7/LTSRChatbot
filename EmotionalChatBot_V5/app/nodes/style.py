import json
from typing import Any, Callable, Dict
from langchain_core.messages import SystemMessage, HumanMessage
from langchain_core.runnables import RunnableConfig

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

def styler_node(state: Dict[str, Any], config: RunnableConfig) -> Dict[str, Any]:
    """
    [Styler Node] - The Nuance Engine (The Mixing Console)
    
    功能：
    根据 6 维关系属性 (The Essential 6) 和 Reasoner 确定的策略，
    计算出 Generator 需要的 12 个具体的表达风格参数 (The 12 Instructions)。
    
    对应关系图：
    A[Closeness] -> OUT1, OUT6, OUT7, OUT8, OUT12
    B[Trust]     -> OUT1, OUT10
    C[Liking]    -> OUT3, OUT4, OUT9, OUT11
    D[Respect]   -> OUT2, OUT5, OUT8
    E[Warmth]    -> OUT7, OUT9
    F[Power]     -> OUT3, OUT4, OUT5, OUT8
    """
    
    llm = config["configurable"].get("llm_model")
    
    # ==========================================
    # 1. 提取输入数据 (Input Extraction)
    # ==========================================
    
    # 1.1 核心关系属性 (The Essential 6)
    rel = state['relationship_state']
    
    # 辅助函数：将 0-100 的数值转化为 [Low/Medium/High] 便于 LLM 理解
    def get_level(val):
        if val <= 35: return "LOW"
        if val >= 65: return "HIGH"
        return "MEDIUM"

    # 1.2 上游策略 (Override Context)
    # 必须参考 Reasoner 的策略，因为策略可能要求“暂时冷漠”，即使平时关系很好
    strategy = state.get('response_strategy', 'Respond naturally.')
    
    # ==========================================
    # 2. 构建核心 Prompt (The Mixing Console)
    # ==========================================
    
    system_prompt = f"""
# Role: The Interaction Stylist
You are the "Mixing Engineer" for an AI's personality. 
Your job is to translate abstract Relationship Metrics into concrete Speaking Instructions (The 12 Output Sliders).

# 1. Input Signals (The Essential 6)
Current Relationship State:
- [A] Closeness ({rel['closeness']}): {get_level(rel['closeness'])}
- [B] Trust     ({rel['trust']}):     {get_level(rel['trust'])}
- [C] Liking    ({rel['liking']}):    {get_level(rel['liking'])}
- [D] Respect   ({rel['respect']}):   {get_level(rel['respect'])}
- [E] Warmth    ({rel['warmth']}):    {get_level(rel['warmth'])}
- [F] Power     ({rel['power']}):     {get_level(rel['power'])} (Bot's Dominance)

# 2. Context Override
**Current Strategy:** "{strategy}"
*(Note: If the Strategy explicitly demands a specific tone like "Be cold" or "Be brief", it OVERRIDES the relationship metrics below.)*

# 3. Calculation Rules (The Mapping Logic)
You must determine the setting for each of the 12 outputs based strictly on the input dependencies:

## Group 1: Intimacy & Openness (Driven by A, B)
1. **self_disclosure** (A+B): 
   - High Closeness + High Trust = High (Share secrets/feelings).
   - Low Trust = Low (Deflect personal questions).
6. **memory_hook** (A): 
   - High Closeness = Frequent (Reference past shared events "Remember when...").
   - Low Closeness = None.
10. **emotional_display** (B): 
    - High Trust = Raw/Vulnerable. 
    - Low Trust = Masked/Professional.
12. **non_verbal_cues** (A): 
    - High Closeness = High (Use asterisks for intimate actions *hugs*, *pokes*).

## Group 2: Tone & Energy (Driven by C, E)
9. **tone_temperature** (C+E): 
   - High Liking + High Warmth = Hot/Affectionate.
   - Low Liking = Cold/Distant.
11. **wit_and_humor** (C): 
    - High Liking = Playful/Teasing. 
    - Low Liking = Serious.
7. **verbal_length** (A+E): 
    - High Closeness = Short/Casual fragments. 
    - High Warmth = Longer/Chatty sentences.
    - *Balance these two based on context.*

## Group 3: Power & Dynamics (Driven by F, D)
3. **initiative** (C+F): 
   - High Power = Lead the conversation. 
   - High Liking = Ask questions to keep chat going.
4. **advice_style** (C+F): 
   - High Power = Directive/Commanding ("You should..."). 
   - Low Power = Tentative/Suggestive ("Maybe you could...").
5. **subjectivity** (D+F): 
   - High Power = Strong Opinions. 
   - High Respect = Balanced/Objective view.
2. **topic_adherence** (D): 
   - High Respect = Listen carefully, stick to their topic. 
   - Low Respect = Feel free to change topic randomly.
8. **social_distance** (A+D+F): 
   - High Closeness = Zero distance (Intimate). 
   - High Power + Low Closeness = Authoritative distance.

# 4. Task
Generate the configuration for the 12 outputs in JSON format.
For each field, provide a short, specific instruction string (e.g., "High. Use nicknames.", "Low. Stay formal.").

# Output Format (JSON)
{{
  "self_disclosure": "...",
  "topic_adherence": "...",
  "initiative": "...",
  "advice_style": "...",
  "subjectivity": "...",
  "memory_hook": "...",
  "verbal_length": "...",
  "social_distance": "...",
  "tone_temperature": "...",
  "emotional_display": "...",
  "wit_and_humor": "...",
  "non_verbal_cues": "..."
}}
"""

    # ==========================================
    # 3. 执行与解析 (Execution)
    # ==========================================
    
    try:
        response = llm.invoke([
            SystemMessage(content=system_prompt),
            HumanMessage(content="Configure the style parameters now.")
        ])
        # 解析 JSON
        instructions = json.loads(response.content)
        
    except Exception as e:
        print(f"Styler Node Error: {e}")
        # 容错默认值
        instructions = {
            "self_disclosure": "Medium",
            "topic_adherence": "High",
            "initiative": "Medium",
            "advice_style": "Soft",
            "subjectivity": "Medium",
            "memory_hook": "None",
            "verbal_length": "Medium",
            "social_distance": "Polite",
            "tone_temperature": "Warm",
            "emotional_display": "Balanced",
            "wit_and_humor": "None",
            "non_verbal_cues": "Minimal"
        }

    # ==========================================
    # 4. 返回状态更新
    # ==========================================
    return {
        "llm_instructions": instructions
    }


# -------------------------------------------------------------------
# LangGraph 兼容包装器（供 app/graph.py 使用）
# -------------------------------------------------------------------

def create_style_node(llm_invoker: Any) -> Callable[[Dict[str, Any]], dict]:
    """
    兼容旧编排：graph.py 期望 create_style_node(llm) -> fn(state)->dict。
    同时补齐 relationship_state 中 The Essential 6，避免 KeyError。
    """

    def _ensure_defaults(s: Dict[str, Any]) -> Dict[str, Any]:
        s = dict(s)
        rel = dict(s.get("relationship_state") or {})
        # Styler Node 依赖的 6 维指标
        rel.setdefault("closeness", 0)
        rel.setdefault("trust", 0)
        rel.setdefault("liking", 50)
        rel.setdefault("respect", 50)
        rel.setdefault("warmth", 50)
        rel.setdefault("power", 50)
        s["relationship_state"] = rel
        return s

    @traceable(
        run_type="chain",
        name="Thinking/Styler",
        tags=["node", "thinking", "styler", "style"],
        metadata={"state_outputs": ["llm_instructions", "style_analysis"]},
    )
    def node(state: Dict[str, Any]) -> dict:
        safe = _ensure_defaults(state)
        out = styler_node(safe, {"configurable": {"llm_model": llm_invoker}})
        # 兼容字段：给旧 generator 一个简单可读的 style_analysis
        instr = out.get("llm_instructions") or {}
        out["style_analysis"] = "12D instructions ready" if instr else ""
        return out

    return node