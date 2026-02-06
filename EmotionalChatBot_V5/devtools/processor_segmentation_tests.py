"""
Processor 拆句/延迟系统测试脚本（论文复现友好：每组用固定随机种子）。

用法：
  cd EmotionalChatBot_V5
  python3 devtools/processor_segmentation_tests.py
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Dict, List, Optional

from langchain_core.messages import HumanMessage

import sys
from pathlib import Path

# allow running from devtools/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from app.nodes.behavior_processor import (
    STAGE_DELAY_FACTORS,
    calculate_human_dynamics,
    create_behavior_processor_node,
)
from app.state import AgentState


@dataclass
class Case:
    name: str
    seed: int
    stage: str
    user_input: str
    final_response: str
    # Big Five (repo: [-1, 1])
    extraversion: float
    conscientiousness: float
    neuroticism: float
    # Relationship (0-100)
    closeness: float
    trust: float = 50.0
    # Mood (PAD + busyness)
    arousal: float = 0.0
    pleasure: float = 0.0
    dominance: float = 0.0
    busyness: float = 0.0


def _mk_state(c: Case) -> AgentState:
    return {
        "messages": [HumanMessage(content=c.user_input)],
        "user_input": c.user_input,
        "current_stage": c.stage,  # type: ignore
        "bot_big_five": {
            "openness": 0.0,
            "agreeableness": 0.0,
            "extraversion": c.extraversion,
            "conscientiousness": c.conscientiousness,
            "neuroticism": c.neuroticism,
        },
        "relationship_state": {
            "closeness": c.closeness,
            "trust": c.trust,
            "liking": 50.0,
            "respect": 50.0,
            "warmth": 50.0,
            "power": 50.0,
        },
        "mood_state": {
            "pleasure": c.pleasure,
            "arousal": c.arousal,
            "dominance": c.dominance,
            "busyness": c.busyness,
        },
        # processor 会优先取 final_response
        "final_response": c.final_response,
        "draft_response": c.final_response,
    }


def _split_threshold_from_dynamics(dyn: Dict[str, float]) -> int:
    # 与 processor.py 当前实现保持一致：20 - frag*15
    thr = 20 - (dyn["fragmentation_tendency"] * 15)
    thr = max(5.0, min(30.0, float(thr)))
    return int(thr)


def _print_case_result(c: Case, out: dict, dyn: Dict[str, float]) -> None:
    bubbles: List[str] = out.get("final_segments") or []
    ho = out.get("humanized_output") or {}
    breakdown = ho.get("latency_breakdown") or {}
    timeline = ho.get("segments") or []

    print("\n" + "=" * 90)
    print(f"CASE: {c.name}")
    print("-" * 90)
    print(f"stage={c.stage} (factor={STAGE_DELAY_FACTORS.get(c.stage, 1.0)})")
    print(
        "Big5: "
        f"E={c.extraversion:+.2f}, C={c.conscientiousness:+.2f}, N={c.neuroticism:+.2f} | "
        f"Mood: arousal={c.arousal:+.2f}, busyness={c.busyness:.2f} | "
        f"Rel: closeness={c.closeness:.0f}"
    )
    print(f"user_input_len={len(c.user_input)} | response_len={len(c.final_response)}")
    print(
        "dynamics: "
        f"speed_factor={dyn['speed_factor']:.3f}, noise_level={dyn['noise_level']:.3f}, "
        f"frag_tendency={dyn['fragmentation_tendency']:.3f}"
    )
    print(f"split_threshold(chars)={_split_threshold_from_dynamics(dyn)}")

    print("\nlatency_breakdown:", breakdown)
    print(f"final_delay(first bubble)={out.get('final_delay')}")

    print(f"\nsegments={len(bubbles)}")
    for i, b in enumerate(bubbles, 1):
        preview = b.replace("\n", "\\n")
        if len(preview) > 60:
            preview = preview[:60] + "..."
        print(f"  [{i:02d}] len={len(b):3d}  {preview}")

    print("\ntimeline (first 5):")
    for seg in timeline[:5]:
        content = str(seg.get("content", "")).replace("\n", "\\n")
        if len(content) > 50:
            content = content[:50] + "..."
        print(f"  delay={seg.get('delay'):>5}s  action={seg.get('action'):>6}  {content}")


def main() -> None:
    processor = create_behavior_processor_node()

    base_text = (
        "我懂你。今天那种“喘不过气”的感觉，很像是压力一直堆着没地方放。"
        "要不你先告诉我，是工作、关系、还是身体状态在拖你后腿？"
        "我们先把最重的那一块拎出来。"
    )

    cases: List[Case] = [
        Case(
            name="High E + High closeness + High arousal (more fragmented / faster typing)",
            seed=1,
            stage="intensifying",
            user_input="我真的有点撑不住了……",
            final_response=base_text,
            extraversion=0.8,
            conscientiousness=-0.2,
            neuroticism=0.0,
            closeness=85,
            arousal=0.7,
            busyness=0.2,
        ),
        Case(
            name="Low E + Low closeness + Low arousal (less fragmented / slower)",
            seed=2,
            stage="initiating",
            user_input="你好。",
            final_response=base_text,
            extraversion=-0.6,
            conscientiousness=0.5,
            neuroticism=0.0,
            closeness=10,
            arousal=-0.4,
            busyness=0.1,
        ),
        Case(
            name="Stagnating stage (large cognitive delay)",
            seed=3,
            stage="stagnating",
            user_input="在吗？",
            final_response=base_text,
            extraversion=0.2,
            conscientiousness=0.2,
            neuroticism=0.1,
            closeness=55,
            arousal=0.0,
            busyness=0.4,
        ),
        Case(
            name="Avoiding stage + high busyness (very slow reply)",
            seed=4,
            stage="avoiding",
            user_input="你是不是不想理我？",
            final_response=base_text,
            extraversion=0.0,
            conscientiousness=0.3,
            neuroticism=0.2,
            closeness=40,
            arousal=-0.1,
            busyness=0.9,
        ),
        Case(
            name="High neuroticism (hesitation noise affects t_cog)",
            seed=5,
            stage="experimenting",
            user_input="我不知道该不该继续坚持。",
            final_response=base_text,
            extraversion=0.2,
            conscientiousness=0.0,
            neuroticism=0.9,
            closeness=50,
            arousal=0.2,
            busyness=0.2,
        ),
        Case(
            name="Short response (should become 1-2 bubbles)",
            seed=6,
            stage="integrating",
            user_input="嗯。",
            final_response="嗯，我在。慢慢说。",
            extraversion=0.4,
            conscientiousness=0.1,
            neuroticism=0.0,
            closeness=70,
            arousal=0.1,
            busyness=0.0,
        ),
    ]

    for c in cases:
        random.seed(c.seed)  # 保证每次跑一致（gauss/uniform）
        state = _mk_state(c)
        dyn = calculate_human_dynamics(state)
        out = processor(state)
        _print_case_result(c, out, dyn)

    print("\n✅ All processor cases done.")


if __name__ == "__main__":
    main()

