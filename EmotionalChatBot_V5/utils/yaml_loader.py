"""配置文件读取工具"""
import os
from pathlib import Path
from typing import Any, Dict, List

import yaml


def load_yaml(path: str | Path) -> Dict[str, Any]:
    """加载单个 YAML 文件为字典"""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_modes_from_dir(dir_path: str | Path) -> List[Dict[str, Any]]:
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


def get_project_root() -> Path:
    """获取项目根目录（EmotionalChatBot_V5）"""
    current = Path(__file__).resolve().parent
    # utils -> EmotionalChatBot_V5
    return current.parent
