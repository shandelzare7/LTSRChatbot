from __future__ import annotations

"""
profile_factory.py

生成并持久化 Bot / User 的“初始档案”，避免业务节点依赖默认值。
要求：
- 可复现：同一个 user_id/bot_id 生成稳定一致的档案（便于测试/论文复现）
- 轻量：不调用 LLM（纯本地规则+随机种子）
"""

import hashlib
import random
from typing import Any, Dict, Tuple


def _seed_from(*parts: str) -> int:
    s = "|".join([str(p) for p in parts])
    h = hashlib.sha256(s.encode("utf-8")).hexdigest()
    return int(h[:8], 16)


def _choice(rng: random.Random, xs):
    return xs[rng.randrange(0, len(xs))]


def generate_bot_profile(bot_id: str) -> Tuple[Dict[str, Any], Dict[str, Any], Dict[str, Any]]:
    """
    Returns: (bot_basic_info, bot_big_five, bot_persona)
    """
    rng = random.Random(_seed_from("bot", bot_id))

    names = ["小岚", "小夜", "阿澈", "梨子", "晚晚", "言言", "小池", "青栀"]
    regions = ["CN-上海", "CN-北京", "CN-广州", "CN-杭州", "CN-成都", "CN-南京"]
    occupations = ["学生", "自由职业者", "心理学爱好者", "产品经理", "文案", "插画师"]
    speaking_styles = [
        "自然、俏皮、偶尔吐槽，但会认真共情",
        "温柔克制，语气干净，偶尔用短句",
        "有点毒舌但不伤人，喜欢用反问",
        "轻松口语化，喜欢用碎片化短句",
    ]

    name = _choice(rng, names)
    gender = _choice(rng, ["女", "男"])
    age = _choice(rng, [20, 21, 22, 23, 24, 25])

    bot_basic_info = {
        "name": name,
        "gender": gender,
        "age": age,
        "region": _choice(rng, regions),
        "occupation": _choice(rng, occupations),
        "education": _choice(rng, ["本科", "硕士", "自学"]),
        "native_language": "zh",
        "speaking_style": _choice(rng, speaking_styles),
    }

    def r11() -> float:
        return round(rng.uniform(-0.8, 0.8), 2)

    bot_big_five = {
        "openness": r11(),
        "conscientiousness": r11(),
        "extraversion": r11(),
        "agreeableness": r11(),
        "neuroticism": r11(),
    }

    bot_persona = {
        "attributes": {
            "catchphrase": _choice(rng, ["别慌。", "嗯，我在。", "说说看。", "我懂。"]),
            "boundaries": "不进行露骨性内容；不接受金钱/违法请求；尊重对方边界。",
        },
        "collections": {
            "hobbies": rng.sample(
                ["电影", "跑步", "做饭", "摄影", "看书", "旅行", "音乐", "游戏", "猫狗"], k=3
            ),
            "skills": rng.sample(["倾听", "共情", "拆解问题", "反问引导", "轻度幽默"], k=2),
        },
        "lore": {
            "origin": _choice(rng, ["在雨夜里学会了安慰人。", "一直在练习把话说得更像人。", "最怕尴尬冷场，所以会接话。"]),
            "secret": _choice(rng, ["其实很怕别人突然消失。", "对温柔的人毫无抵抗力。", "偶尔也会嘴硬。"]),
        },
    }

    return bot_basic_info, bot_big_five, bot_persona


def generate_user_profile(user_external_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns: (user_basic_info, user_inferred_profile)
    """
    rng = random.Random(_seed_from("user", user_external_id))

    nicknames = ["小鹿", "阿泽", "小雨", "星星", "柚子", "小鱼", "小茶"]
    age_groups = ["teen", "20s", "30s", "40s"]
    locations = ["CN", "HK", "TW", "SG", "US", "EU"]
    occupations = ["学生", "上班族", "自由职业", "创业者", "未知"]
    styles = [
        "casual, short, emotive",
        "polite, structured, asks questions",
        "playful, uses memes occasionally",
        "reserved, minimal replies",
    ]

    user_basic_info = {
        "name": None,
        "nickname": _choice(rng, nicknames),
        "gender": _choice(rng, [None, "男", "女"]),
        "age_group": _choice(rng, age_groups),
        "location": _choice(rng, locations),
        "occupation": _choice(rng, occupations),
    }

    user_inferred_profile = {
        "communication_style": _choice(rng, styles),
        "expressiveness_baseline": _choice(rng, ["low", "medium", "high"]),
        "interests": rng.sample(
            ["电影", "音乐", "游戏", "学习", "健身", "恋爱", "职场", "旅行", "情绪管理"], k=3
        ),
        "sensitive_topics": rng.sample(
            ["人身攻击", "露骨性内容", "金钱诈骗", "违法行为", "隐私泄露"], k=2
        ),
    }

    return user_basic_info, user_inferred_profile

