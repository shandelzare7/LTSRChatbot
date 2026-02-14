"""
Prompt Helper Functions

被 Reasoner / Generator 等节点用于把 state 中的数值转成更易读的约束/指令文本。
含 6 维关系属性的详细数值说明加载与 LLM 提示词格式化。
"""

from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from utils.yaml_loader import get_project_root, load_yaml, load_stage_by_id
except Exception:
    get_project_root = None
    load_yaml = None
    load_stage_by_id = None

# 6 维 key 顺序（与 state 一致）
REL_DIM_KEYS = ("closeness", "trust", "liking", "respect", "warmth", "power")

# 未加载 YAML 时的默认中文名与简要说明（0–1 客观语义）
REL_DIM_DEFAULTS: Dict[str, Dict[str, str]] = {
    "closeness": {"label_zh": "亲密度", "brief": "关系距离/熟悉程度"},
    "trust": {"label_zh": "信任", "brief": "可靠性/善意/可预测性确信"},
    "liking": {"label_zh": "喜爱", "brief": "好感/亲近偏好"},
    "respect": {"label_zh": "尊重", "brief": "认可/认真对待/边界承认"},
    "warmth": {"label_zh": "暖意", "brief": "情感温度底色（冷淡↔温暖）"},
    "power": {"label_zh": "权力/主导", "brief": "互动姿态的强势程度"},
}

# 关系值 0–1 语义锚定：重要提示（防止误读 0）
REL_NOTE_FOR_LLM = (
    "重要提示：0 不是“中性起点”，而是“几乎不存在/极弱”。0.5 附近才可视为“中等/一般”。"
    "各维度可相互独立（例如可能出现 高尊重但低喜爱，或 高喜爱但低暖意）。"
)

_dimensions_config: Optional[Dict[str, Any]] = None


def load_relationship_dimensions() -> Dict[str, Any]:
    """加载 config/relationship_dimensions.yaml，返回 dimensions 字典。失败时返回空结构。"""
    global _dimensions_config
    if _dimensions_config is not None:
        return _dimensions_config
    if get_project_root is None or load_yaml is None:
        _dimensions_config = {}
        return _dimensions_config
    try:
        root = get_project_root()
        path = root / "config" / "relationship_dimensions.yaml"
        if path.exists():
            data = load_yaml(path)
            _dimensions_config = data.get("dimensions") or {}
        else:
            _dimensions_config = {}
    except Exception:
        _dimensions_config = {}
    return _dimensions_config


def _to_01(value: Any) -> float:
    """将值钳位到 0-1 范围（系统内部统一使用 0-1）。"""
    try:
        v = float(value)
        return max(0.0, min(1.0, v))
    except (TypeError, ValueError):
        return 0.0


def get_relationship_tier(value_01: float, dim_key: str) -> Dict[str, str]:
    """根据 0–1 数值与维度 key，返回当前区间的 name 与 desc（客观语义，无指导）。区间左闭右开，末段含 1.0。"""
    dims = load_relationship_dimensions()
    dim = dims.get(dim_key) if isinstance(dims, dict) else None
    if not dim or not isinstance(dim.get("tiers"), list):
        return {"name": "", "desc": ""}
    v = max(0.0, min(1.0, float(value_01)))
    tiers = dim["tiers"]
    for i, t in enumerate(tiers):
        r = t.get("range")
        if not isinstance(r, (list, tuple)) or len(r) < 2:
            continue
        lo, hi = float(r[0]), float(r[1])
        is_last = i == len(tiers) - 1
        if is_last:
            if lo <= v <= hi:
                return {"name": str(t.get("name") or ""), "desc": str(t.get("desc") or "")}
        else:
            if lo <= v < hi:
                return {"name": str(t.get("name") or ""), "desc": str(t.get("desc") or "")}
    return {"name": "", "desc": ""}


def format_relationship_for_llm(relationship_state: Dict[str, Any]) -> str:
    """
    将 6 维关系状态格式化为供 LLM 阅读的客观语义说明（0–1 区间锚定，无指导性表述）。
    系统内部统一使用 0-1 范围。
    """
    rel = relationship_state or {}
    dims = load_relationship_dimensions()
    lines: List[str] = []
    for key in REL_DIM_KEYS:
        val_raw = rel.get(key, 0)
        val_01 = _to_01(val_raw)
        dim = dims.get(key) if isinstance(dims, dict) else {}
        default = REL_DIM_DEFAULTS.get(key) or {}
        label_zh = dim.get("label_zh") or default.get("label_zh") or key
        brief = dim.get("brief") or default.get("brief") or ""
        tier = get_relationship_tier(val_01, key)
        name, desc = tier.get("name") or "", tier.get("desc") or ""
        val_str = f"{val_01:.2f}" if val_01 < 1.0 else "1.00"
        if name or desc:
            lines.append(f"- **{label_zh}**（{brief}）: {val_str} — {name}。{desc}")
        else:
            lines.append(f"- **{label_zh}**（{brief}）: {val_str}")
    if not lines:
        return "（暂无关系维度数据）"
    return "\n".join(lines) + "\n\n" + REL_NOTE_FOR_LLM


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

    closeness = rel.get("closeness", 0.0)
    trust = rel.get("trust", 0.0)
    power = rel.get("power", rel.get("dominance", 0.5))

    lines = [
        f"- Stage: {stage} (intimacy boundary anchor)",
        f"- Closeness: {closeness:.2f} (0-1 range)",
        f"- Trust: {trust:.2f} (0-1 range)",
        f"- Power: {power:.2f} (0-1 range)",
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


def format_stage_for_llm(stage_id: str) -> str:
    """
    将 Knapp 阶段 ID 格式化为供 LLM 阅读的描述信息。
    包含：阶段名称、角色、目标、策略等。
    """
    if load_stage_by_id is None:
        return f"阶段ID: {stage_id}"
    
    try:
        stage_config = load_stage_by_id(stage_id)
        if not stage_config:
            return f"阶段ID: {stage_id}"
        
        lines = []
        lines.append(f"**阶段ID**: {stage_id}")
        
        stage_name = stage_config.get("stage_name") or ""
        stage_number = stage_config.get("stage_number")
        phase = stage_config.get("phase") or ""
        role = stage_config.get("role") or ""
        stage_goal = stage_config.get("stage_goal") or ""
        strategy = stage_config.get("strategy") or []
        
        if stage_name:
            lines.append(f"**阶段名称**: {stage_name}")
        if stage_number:
            lines.append(f"**阶段编号**: {stage_number}")
        if phase:
            phase_zh = "关系上升期" if phase == "coming_together" else "关系解体期"
            lines.append(f"**所属阶段**: {phase_zh}")
        if role:
            lines.append(f"**角色**: {role}")
        if stage_goal:
            lines.append(f"**阶段目标**: {stage_goal}")
        if strategy:
            lines.append("**策略要点**:")
            if isinstance(strategy, list):
                for i, s in enumerate(strategy[:5], 1):  # 最多5条
                    lines.append(f"  {i}. {s}")
            else:
                lines.append(f"  {strategy}")
        
        return "\n".join(lines)
    except Exception as e:
        # 降级：只返回阶段ID
        return f"阶段ID: {stage_id}（加载描述失败: {e}）"

