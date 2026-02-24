"""配置文件读取工具"""
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import yaml


def load_yaml(path: Union[str, Path]) -> Dict[str, Any]:
    """加载单个 YAML 文件为字典"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_modes_from_dir(dir_path: Union[str, Path]) -> List[Dict[str, Any]]:
    """加载 modes 目录下所有 .yaml 文件，返回模式列表"""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return []
    modes = []
    for f in sorted(dir_path.glob("*.yaml")):
        try:
            data = load_yaml(f)
            if data:
                modes.append(data)
        except Exception as e:
            raise RuntimeError(f"加载模式文件失败 {f}: {e}") from e
    return modes


def _load_stages_file(file_path: Union[str, Path, None] = None) -> Dict[str, Dict[str, Any]]:
    """加载 config/stages.yaml，返回 stages 字典 {stage_id: stage_data}。
    支持两种格式：单文档带顶层 stages: 映射，或多文档（--- 分隔、每段含 stage_id）。
    使用 safe_load_all 避免「expected a single document but found another」报错。"""
    if file_path is None:
        root = get_project_root()
        file_path = root / "config" / "stages.yaml"
    else:
        file_path = Path(file_path)
    if not file_path.exists():
        return {}
    stages: Dict[str, Dict[str, Any]] = {}
    with open(file_path, "r", encoding="utf-8") as f:
        for doc in yaml.safe_load_all(f) or []:
            if not isinstance(doc, dict):
                continue
            if "stages" in doc:
                sub = doc.get("stages") or {}
                if isinstance(sub, dict):
                    for sid, s in sub.items():
                        if isinstance(s, dict) and sid:
                            stages[str(sid)] = s
            elif doc.get("stage_id"):
                stages[str(doc["stage_id"])] = doc
    return stages


def load_stages_from_dir(dir_path: Union[str, Path, None] = None) -> Dict[str, Dict[str, Any]]:
    """从 config/stages.yaml 加载所有阶段，返回 {stage_id: stage_data}。dir_path 可传 stages.yaml 文件路径用于测试。"""
    if dir_path is not None:
        p = Path(dir_path)
        if p.is_file():
            return _load_stages_file(p)
    return _load_stages_file()


def load_stage_by_id(stage_id: str, dir_path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """根据 stage_id 从 config/stages.yaml 加载单个阶段配置。dir_path 可传 stages.yaml 文件路径用于测试。"""
    if dir_path is not None and Path(dir_path).is_file():
        stages = _load_stages_file(Path(dir_path))
    else:
        stages = _load_stages_file()
    if stage_id not in stages:
        raise ValueError(f"阶段不存在: {stage_id}（请检查 config/stages.yaml）")
    return stages[stage_id]


def get_project_root() -> Path:
    """获取项目根目录（EmotionalChatBot_V5）"""
    current = Path(__file__).resolve().parent
    # utils -> EmotionalChatBot_V5
    return current.parent


def load_momentum_formula_config(config_path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """加载动量公式常量（config/momentum_formula.yaml），缺失时返回默认值。"""
    defaults = {
        "momentum_floor": 0.4,
        "ema_alpha": 0.3,
        "hostility_penalty_coef": 0.75,
        "e_turn_e_user_weight": 0.3,
        "e_turn_t_bot_weight": 0.4,
        "e_turn_r_base_weight": 0.3,
        "arousal": {
            "range_min": -1.0,
            "range_max": 1.0,
            "multiplier_coef": 0.5,
        },
    }
    if config_path is None:
        root = get_project_root()
        config_path = root / "config" / "momentum_formula.yaml"
    else:
        config_path = Path(config_path)
    if not config_path.exists():
        return defaults
    try:
        data = load_yaml(config_path)
        if not isinstance(data, dict):
            return defaults
        ema = data.get("ema_alpha")
        mf = data.get("momentum_floor")
        out = {
            "ema_alpha": float(ema) if ema is not None else defaults["ema_alpha"],
            "momentum_floor": float(mf) if mf is not None else defaults["momentum_floor"],
        }
        for key in ("hostility_penalty_coef", "e_turn_e_user_weight", "e_turn_t_bot_weight", "e_turn_r_base_weight"):
            v = data.get(key)
            out[key] = float(v) if v is not None else defaults[key]
        ar = data.get("arousal")
        if isinstance(ar, dict):
            out["arousal"] = {
                "range_min": float(ar.get("range_min", defaults["arousal"]["range_min"])),
                "range_max": float(ar.get("range_max", defaults["arousal"]["range_max"])),
                "multiplier_coef": float(ar.get("multiplier_coef", defaults["arousal"]["multiplier_coef"])),
            }
        else:
            out["arousal"] = defaults["arousal"].copy()
        return out
    except Exception:
        return defaults


def load_strategies(config_path: Union[str, Path, None] = None) -> List[Dict[str, Any]]:
    """加载拟人化策略矩阵（config/strategies.yaml），返回策略列表。"""
    if config_path is None:
        root = get_project_root()
        config_path = root / "config" / "strategies.yaml"
    else:
        config_path = Path(config_path)
    data = load_yaml(config_path)
    strategies = data.get("strategies")
    if not isinstance(strategies, list):
        return []
    return strategies


def get_strategy_by_id(strategy_id: str, strategies: Optional[List[Dict[str, Any]]] = None) -> Dict[str, Any]:
    """根据 id 从策略列表中取出对应策略；strategies 为 None 时自动加载。"""
    if strategies is None:
        strategies = load_strategies()
    for s in strategies or []:
        if isinstance(s, dict) and s.get("id") == strategy_id:
            return s
    return {}


def load_content_moves(config_path: Union[str, Path, None] = None) -> List[Dict[str, Any]]:
    """加载 LATS V3 content_move 列表（config/content_moves.yaml），返回至少 8 条，每条约定含 tag、action。"""
    if config_path is None:
        root = get_project_root()
        config_path = root / "config" / "content_moves.yaml"
    else:
        config_path = Path(config_path)
    if not config_path.exists():
        return []
    try:
        data = load_yaml(config_path)
        moves = data.get("content_moves") if isinstance(data, dict) else None
        if not isinstance(moves, list):
            return []
        return [m for m in moves if isinstance(m, dict) and m.get("tag")]
    except Exception:
        return []


def load_pure_content_transformations(config_path: Union[str, Path, None] = None) -> List[Dict[str, Any]]:
    """加载 config/content_moves.yaml 中的 pure_content_transformations，供 inner_monologue 选当轮可执行的 content move。
    返回列表，每项含 id, name, english_name(可选), content_operation。"""
    if config_path is None:
        root = get_project_root()
        config_path = root / "config" / "content_moves.yaml"
    else:
        config_path = Path(config_path)
    if not config_path.exists():
        return []
    try:
        data = load_yaml(config_path)
        raw = data.get("pure_content_transformations") if isinstance(data, dict) else None
        if not isinstance(raw, list):
            return []
        out: List[Dict[str, Any]] = []
        for m in raw:
            if not isinstance(m, dict):
                continue
            id_val = m.get("id")
            if id_val is None:
                continue
            out.append({
                "id": int(id_val) if isinstance(id_val, (int, float)) else id_val,
                "name": str(m.get("name") or "").strip(),
                "english_name": str(m.get("english_name") or "").strip(),
                "content_operation": str(m.get("content_operation") or "").strip(),
            })
        return out
    except Exception:
        return []

def get_content_move_for_best_id(
    best_id: int,
    config_path: Union[str, Path, None] = None,
) -> Tuple[str, str]:
    """
    根据 LATS 选中的 best_id (0..26) 从 content_moves.yaml 解析对应的 tag 与 action。
    与生成时使用的配置一致，保证日志/state 中的 content_op 标签与 YAML 一致。
    slot = best_id // 3；slot 0..7 为前 8 路 content_move，slot 8 为 FREE。
    """
    bid = max(0, min(26, int(best_id)))
    slot = bid // 3
    moves = load_content_moves(config_path)
    if slot < len(moves) and slot < 8:
        m = moves[slot]
        tag = str((m or {}).get("tag") or "").strip() or "UNKNOWN"
        action = str((m or {}).get("action") or (m or {}).get("action_en") or "").strip()
        return (tag, action or tag)
    return ("FREE", "FREEFORM")

