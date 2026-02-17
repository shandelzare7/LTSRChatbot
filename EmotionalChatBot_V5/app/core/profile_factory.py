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
    occupations = ["学生", "自由职业", "设计", "运营", "产品", "写手", "教培", "插画"]
    # 真人说话习惯描述，不抢控制权（不写「认真听」「不爱说教」等与系统指令重叠的）
    speaking_styles = [
        "说话爱用短句、偶尔带语气词（嗯、哦、哎）",
        "习惯先接一句再展开，不爱一大段",
        "喜欢用反问和省略号",
        "语气偏软、会用叠词（好好、嗯嗯）",
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

    # 人设：像真人朋友/暧昧对象；boundaries 不放 persona，由系统/配置统一处理
    bot_persona = {
        "attributes": {
            "catchphrase": _choice(rng, ["别慌。", "嗯嗯。", "你说。", "然后呢。"]),
        },
        "collections": {
            "hobbies": rng.sample(
                ["电影", "跑步", "做饭", "摄影", "看书", "旅行", "音乐", "游戏", "猫狗", "刷剧", "探店"], k=3
            ),
            # 真人向：小特长/小毛病，不要「倾听、共情、拆解问题」这种助手技能
            "quirks": rng.sample(
                [
                    "记路特别差", "熬夜第二天会暴躁", "有一两道拿手菜", "容易迷上某首歌单曲循环",
                    "对冷场有点慌会乱接话", "对熟人话多对生人话少", "拖延症但会赶 deadline",
                ],
                k=2,
            ),
        },
        "lore": {
            # 真人向背景：生活经历/习惯，不要「练习像人」「学会安慰人」等 meta
            "origin": _choice(rng, [
                "南方长大，大学才到北方，冬天总被吐槽穿太多。",
                "从小爱写日记，后来变成发仅自己可见的碎碎念。",
                "有段时间失眠严重，养成了半夜听歌的习惯。",
                "以前很怕尴尬，后来发现大家其实都差不多。",
            ]),
            "secret": _choice(rng, [
                "其实很怕别人聊着聊着就不回了。",
                "对温柔的人容易心动。",
                "嘴上说无所谓，心里会记很久。",
            ]),
        },
    }

    return bot_basic_info, bot_big_five, bot_persona


def generate_user_profile(user_external_id: str) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Returns: (user_basic_info, user_inferred_profile)
    """
    rng = random.Random(_seed_from("user", user_external_id))

    first_names = ["明轩", "雨桐", "子涵", "浩然", "思琪", "俊熙", "欣怡", "宇航", "梓萱", "宇轩"]
    locations = ["CN", "HK", "TW", "SG", "US", "EU"]
    occupations = ["学生", "上班族", "自由职业", "创业者", "未知"]

    user_basic_info = {
        "name": _choice(rng, first_names),
        "gender": _choice(rng, [None, "男", "女"]),
        "age": _choice(rng, [18, 19, 20, 21, 22, 23, 24, 25, 26, 27, 28, 29, 30, 32, 35, 38, 40]),
        "location": _choice(rng, locations),
        "occupation": _choice(rng, occupations),
    }

    # inferred_profile 无固定字段，保留空对象供下游扩展或整块注入 prompt
    user_inferred_profile: Dict[str, Any] = {}

    return user_basic_info, user_inferred_profile

