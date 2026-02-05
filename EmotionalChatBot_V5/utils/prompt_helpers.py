"""
Prompt Helper Functions

被 Reasoner / Generator 等节点用于把 state 中的数值转成更易读的约束/指令文本。
尽量保持“无依赖、可在 Python3.9 运行”的实现。
"""

from typing import Any, Dict


def _level_0_100(v: Any) -> str:
    """把 0-100 数值映射到 LOW / MEDIUM / HIGH（容错）。"""
    try:
        x = float(v)
    except Exception:
        return "UNKNOWN"
    if x <= 35:
        return "LOW"
    if x >= 65:
        return "HIGH"
    return "MEDIUM"


def format_mind_rules(state: Dict[str, Any]) -> str:
    """
    把关系阶段 + 关键关系属性变成 Mind Node 可用的“约束规则”文本。
    这是一个轻量版本：先能跑通链路，再按你的理论模型慢慢加规则。
    """
    stage = state.get("current_stage", "initiating")
    rel = state.get("relationship_state", {}) or {}

    closeness = rel.get("closeness", 0)
    trust = rel.get("trust", 0)
    power = rel.get("power", rel.get("dominance", 50))

    lines = [
        f"- Stage: {stage} (intimacy boundary anchor)",
        f"- Closeness: {closeness}/100 ({_level_0_100(closeness)})",
        f"- Trust: {trust}/100 ({_level_0_100(trust)})",
        f"- Power: {power}/100 ({_level_0_100(power)})",
    ]
    return "\n".join(lines)


def get_mood_instruction(mood_state: Dict[str, Any]) -> str:
    """
    把 PAD 情绪变成一句话，给 Generator 做“表演指令”。
    """
    mood_state = mood_state or {}
    p = mood_state.get("pleasure", 0.0)
    a = mood_state.get("arousal", 0.0)
    d = mood_state.get("dominance", 0.0)
    return f"PAD={{P:{p:.2f}, A:{a:.2f}, D:{d:.2f}}}（数值越高代表越愉悦/越激动/越强势）"

