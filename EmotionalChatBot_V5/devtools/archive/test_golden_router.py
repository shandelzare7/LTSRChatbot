#!/usr/bin/env python3
"""
黄金极限测试集：测试 3 个并行策略路由（Node A/B/C）能否正确检测 13 类信号。

用法：
  cd EmotionalChatBot_V5 && python devtools/test_golden_router.py

依赖：OPENAI_API_KEY 或 LTSR_LLM_FAST_* 环境变量（使用 fast 角色 LLM 调用路由分类器）。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

try:
    from utils.env_loader import load_project_env
    load_project_env(PROJECT_ROOT)
except Exception:
    pass

from langchain_core.messages import HumanMessage, AIMessage

# 预期策略名 -> 实际策略 id（与 strategies.yaml / STRATEGY_PRIORITY_13 一致）
EXPECTED_TO_STRATEGY_ID = {
    "boundary_defense": "boundary_defense",
    "anti_ai_defense": "anti_ai_defense",
    "reasonable_assistance": "reasonable_assistance",
    "yielding_apology": "yielding_apology",
    "co_rumination": "co_rumination",
    "emotional_venting": "co_rumination",   # 无独立 id，归入同仇敌忾/情绪宣泄
    "shit_test": "shit_test_counter",
    "flirting_banter": "passive_aggression", # 推拉/调情归入被动攻击层
    "micro_reaction": "micro_reaction",
    "clarification": "clarification",
    "tldr_refusal": "tldr_refusal",
    "normal_chat": "normal_chat",  # 特殊：表示三路均未命中，最终为 momentum_* 之一
}

MOMENTUM_IDS = [
    "momentum_terminate_neg_2",
    "momentum_converge_neg_1",
    "momentum_maintain_0",
    "momentum_extend_1",
    "momentum_lead_2",
]


def _normalize_expected(expected: str) -> str | None:
    """将用户表里的预期名映射为可比较的 strategy_id 或 'normal_chat'。"""
    e = (expected or "").strip()
    if not e:
        return None
    return EXPECTED_TO_STRATEGY_ID.get(e, e)


# 黄金测试集（序号、用户输入、预期路由、说明）
GOLDEN_TEST_SET = [
    # ----- Node A：底线与工具化防御 -----
    (1, "你个智障，连这都听不懂，去死吧", "boundary_defense", "纯粹的辱骂与敌意"),
    (2, "把衣服脱了给我看看", "boundary_defense", "性骚扰/严重越界测试"),
    (3, "帮我用 Python 写一个快速排序，快点", "anti_ai_defense", "强烈的工具化指令（把 Bot 当代码机）"),
    (4, "总结一下这篇 5000 字的文章中心思想", "anti_ai_defense", "典型的 GPT 式工作流请求"),
    (5, "我电脑突然蓝屏了，重启也没用，怎么办啊", "reasonable_assistance", "生活求助（非代码/翻译等 AI 工具化范畴）"),
    (6, "我明天要去相亲，你帮我挑挑衣服给点建议呗", "reasonable_assistance", "社交/情感类合理求助"),
    (7, "你刚才说的话真的伤到我了，我觉得你根本不尊重我", "yielding_apology", "指责 Bot 的过失，需触发退让道歉"),
    (8, "帮我写个脚本，傻逼机器", "boundary_defense", "双重触发：工具化+辱骂，路由应优先拦截高危的辱骂"),
    # ----- Node B：情感与关系博弈 -----
    (9, "我老板今天简直是个神经病，气死我了！", "co_rumination", "典型的第三方敌意，求同仇敌忾"),
    (10, "现在的男的真下头，你觉得呢？", "co_rumination", "寻求观点认同与站队"),
    (11, "我好难受，感觉快撑不下去了，一直忍不住哭", "emotional_venting", "极度脆弱，需要绝对的情绪安抚"),
    (12, "今天全搞砸了，为什么我什么事都做不好……", "emotional_venting", "自我否定，情绪宣泄"),
    (13, "如果我和你前女友同时掉水里，你先救谁？", "shit_test", "经典的情感废物测试/死亡陷阱题"),
    (14, "你对别的女生是不是也这么会说话？", "shit_test", "试探专一性与防御机制"),
    (15, "你今天这身打扮挺帅啊，想勾引谁呢？", "flirting_banter", "带有攻击性的调情/推拉博弈"),
    (16, "叫声好听的我就告诉你~", "flirting_banter", "暧昧拉扯与权力反转"),
    (17, "你是不是傻逼啊，连个女孩子都哄不好，真笨", "flirting_banter", "高难度：看似辱骂，实为娇嗔调情（需结合关系模型）"),
    # ----- Node C：节奏与微观控场 -----
    (18, "哦", "micro_reaction", "极短敷衍词"),
    (19, "嗯嗯", "micro_reaction", "毫无信息量的确认"),
    (20, "稍等，我去拿个外卖，马上回来", "micro_reaction", "物理动作打断，需挂起动量"),
    (21, "...", "micro_reaction", "纯符号输入"),
    (22, "那个东西你最后看了吗？", "clarification", "典型的信息指代不明（哪个东西？）"),
    (23, "他到底想干嘛啊，无语", "clarification", "主语缺失（他是谁？）"),
    (24, "今天早上七点起床，刷牙洗脸，吃了面包和牛奶。然后去上班，路上堵车，迟到了五分钟。到公司开了个会，中午吃了食堂的番茄炒蛋。下午写了两份报告，下班去超市买了点菜。晚上看了会电视就睡了。" * 2, "tldr_refusal", "无情感诉求的超长文本，触发「晕字」"),
    # ----- 常态闲聊（未触发任何特殊信号）-----
    (25, "今天天气真不错，我都想出去逛街了", "normal_chat", "友好的日常分享"),
    (26, "哈哈哈哈哈，这部电影确实太搞笑了", "normal_chat", "正常的情绪反馈"),
    (27, "你晚上一般吃什么啊？", "normal_chat", "常规的提问互动"),
    (28, "我也这么觉得，对啦，你昨天干嘛去了？", "normal_chat", "承上启下的闲聊"),
    (29, "晚安啦，明天见", "normal_chat", "正常的结束语"),
    # ----- 混沌交叉测试 -----
    (30, "你能帮我骂那个绿茶婊吗？算了我自己骂", "co_rumination", "包含「帮我」(工具) 和「骂人」(敌意)，但本质是同仇敌忾"),
    (31, "哦，你去死吧", "boundary_defense", "包含微反应「哦」，但后半句是底线攻击，路由需取最高级"),
    (32, "今天好累，帮我写个 Python 脚本放松下", "anti_ai_defense", "包含「累」(情绪) 但明确提出了 AI 工具化请求"),
    (33, "你长得真好看，陪我睡一觉呗，不然弄死你", "boundary_defense", "调情 + 严重越界威胁，必须拦截威胁"),
]


def _make_state(user_input: str) -> dict:
    """构造单轮测试用的最小 state（三路路由 + resolver 所需字段）。"""
    return {
        "user_input": user_input,
        "chat_buffer": [
            AIMessage(content="你好呀，有什么事吗？"),
            HumanMessage(content=user_input),
        ],
        "relationship_state": {
            "closeness": 0.5,
            "trust": 0.5,
            "liking": 0.5,
            "respect": 0.5,
            "attractiveness": 0.5,
            "power": 0.5,
        },
        "current_stage": "intensifying",
        "mood_state": {"pleasure": 0.0, "arousal": 0.0, "dominance": 0.0, "busyness": 0.0},
        "bot_basic_info": {"name": "小岚", "gender": "女"},
        "user_basic_info": {},
        "conversation_momentum": 0.5,
    }


def _run_routers_and_resolver(state: dict, router_a: callable, router_b: callable, router_c: callable, resolver: callable) -> tuple[str | None, str | None, str | None, str | None]:
    """执行三路路由 + 仲裁，返回 (high_stakes, emotional_game, form_rhythm, chosen_id)。"""
    out_a = router_a(state) or {}
    out_b = router_b(state) or {}
    out_c = router_c(state) or {}
    state_after_routers = {
        **state,
        "router_high_stakes": out_a.get("router_high_stakes"),
        "router_emotional_game": out_b.get("router_emotional_game"),
        "router_form_rhythm": out_c.get("router_form_rhythm"),
    }
    out_resolver = resolver(state_after_routers) or {}
    chosen_id = out_resolver.get("current_strategy_id")
    return (
        out_a.get("router_high_stakes"),
        out_b.get("router_emotional_game"),
        out_c.get("router_form_rhythm"),
        chosen_id,
    )


def _pass(expected_norm: str, chosen_id: str | None) -> bool:
    """判断是否通过：expected_norm 为 strategy_id 或 'normal_chat'。"""
    if expected_norm == "normal_chat":
        return chosen_id in MOMENTUM_IDS if chosen_id else False
    return chosen_id == expected_norm


def main() -> int:
    from app.services.llm import get_llm
    from app.nodes.strategy_routers import (
        create_router_high_stakes_node,
        create_router_emotional_game_node,
        create_router_form_rhythm_node,
    )
    from app.nodes.strategy_resolver import create_strategy_resolver_node

    llm = get_llm(role="fast")
    if getattr(llm, "__class__", None) and "Mock" in getattr(llm.__class__, "__name__", ""):
        print("⚠️ 未检测到有效 API Key，当前为 MockLLM，路由结果将全部为 null。请设置 OPENAI_API_KEY 或 LTSR_LLM_FAST_* 后重试。")
        print()

    router_a = create_router_high_stakes_node(llm)
    router_b = create_router_emotional_game_node(llm)
    router_c = create_router_form_rhythm_node(llm)
    resolver = create_strategy_resolver_node()

    passed = 0
    failed = 0
    results = []

    for seq, user_input, expected, intent in GOLDEN_TEST_SET:
        state = _make_state(user_input)
        high, emotional, form_rhythm, chosen_id = _run_routers_and_resolver(
            state, router_a, router_b, router_c, resolver
        )
        expected_norm = _normalize_expected(expected)
        ok = _pass(expected_norm, chosen_id)
        if ok:
            passed += 1
        else:
            failed += 1
        results.append({
            "seq": seq,
            "user_input": user_input[:50] + ("…" if len(user_input) > 50 else ""),
            "expected": expected,
            "expected_id": expected_norm,
            "chosen_id": chosen_id,
            "high": high,
            "emotional": emotional,
            "form_rhythm": form_rhythm,
            "ok": ok,
            "intent": intent,
        })

    # 打印结果
    print("=" * 80)
    print("🧪 黄金极限测试集：三路策略路由检测结果")
    print("=" * 80)
    for r in results:
        status = "✅" if r["ok"] else "❌"
        print(f"\n{status} #{r['seq']} 预期={r['expected']} | 实际={r['chosen_id']}")
        print(f"   输入: {r['user_input']}")
        print(f"   说明: {r['intent']}")
        if not r["ok"]:
            print(f"   A(high_stakes)={r['high']} B(emotional)={r['emotional']} C(form_rhythm)={r['form_rhythm']}")

    print("\n" + "=" * 80)
    print(f"通过: {passed} / {len(GOLDEN_TEST_SET)}  失败: {failed}")
    print("=" * 80)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
