"""配置文件读取工具"""
import os
from pathlib import Path
from typing import Any, Dict, List, Union

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


def load_stages_from_dir(dir_path: Union[str, Path]) -> Dict[str, Dict[str, Any]]:
    """加载 stages 目录下所有 .yaml 文件，返回阶段字典 {stage_id: stage_data}"""
    dir_path = Path(dir_path)
    if not dir_path.is_dir():
        return {}
    stages = {}
    for f in sorted(dir_path.glob("*.yaml")):
        try:
            data = load_yaml(f)
            if data and "stage_id" in data:
                stages[data["stage_id"]] = data
        except Exception as e:
            raise RuntimeError(f"加载阶段文件失败 {f}: {e}") from e
    return stages


def load_stage_by_id(stage_id: str, dir_path: Union[str, Path, None] = None) -> Dict[str, Any]:
    """根据 stage_id 加载单个阶段配置"""
    if dir_path is None:
        root = get_project_root()
        dir_path = root / "config" / "stages"
    else:
        dir_path = Path(dir_path)
    
    stage_file = dir_path / f"{stage_id}.yaml"
    if not stage_file.exists():
        raise ValueError(f"阶段文件不存在: {stage_file}")
    
    return load_yaml(stage_file)


def get_project_root() -> Path:
    """获取项目根目录（EmotionalChatBot_V5）"""
    current = Path(__file__).resolve().parent
    # utils -> EmotionalChatBot_V5
    return current.parent
