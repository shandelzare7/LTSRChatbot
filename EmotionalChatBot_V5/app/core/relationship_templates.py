"""
关系维度初始值模板

定义3种不同的初始关系模板，在创建新关系时随机选择其中一个。
"""
from __future__ import annotations

import random
from typing import Dict, List, Literal

# 3个关系维度初始值模板
RELATIONSHIP_TEMPLATES: List[Dict[str, float]] = [
    {
        "name": "neutral_stranger",
        "description": "中性陌生人（更克制、少热情）",
        "closeness": 0.15,
        "trust": 0.18,
        "liking": 0.40,
        "respect": 0.52,
        "warmth": 0.45,
        "power": 0.50,
    },
    {
        "name": "friendly_icebreaker",
        "description": "友好破冰（更像聚会上主动搭话）",
        "closeness": 0.22,
        "trust": 0.25,
        "liking": 0.55,
        "respect": 0.58,
        "warmth": 0.60,
        "power": 0.48,
    },
    {
        "name": "moderate_acquaintance",
        "description": "中等熟悉度（平衡的初始关系）",
        "closeness": 0.18,
        "trust": 0.22,
        "liking": 0.45,
        "respect": 0.55,
        "warmth": 0.52,
        "power": 0.50,
    },
]


def get_random_relationship_template() -> Dict[str, float]:
    """
    随机选择一个关系维度模板
    
    Returns:
        包含关系维度的字典，格式：
        {
            "closeness": float,
            "trust": float,
            "liking": float,
            "respect": float,
            "warmth": float,
            "power": float,
        }
    """
    template = random.choice(RELATIONSHIP_TEMPLATES)
    return {
        "closeness": template["closeness"],
        "trust": template["trust"],
        "liking": template["liking"],
        "respect": template["respect"],
        "warmth": template["warmth"],
        "power": template["power"],
    }


def get_relationship_template_by_name(name: Literal["neutral_stranger", "friendly_icebreaker", "moderate_acquaintance"]) -> Dict[str, float]:
    """
    根据名称获取指定的关系维度模板
    
    Args:
        name: 模板名称
        
    Returns:
        包含关系维度的字典
    """
    for template in RELATIONSHIP_TEMPLATES:
        if template["name"] == name:
            return {
                "closeness": template["closeness"],
                "trust": template["trust"],
                "liking": template["liking"],
                "respect": template["respect"],
                "warmth": template["warmth"],
                "power": template["power"],
            }
    # fallback to random if not found
    return get_random_relationship_template()
