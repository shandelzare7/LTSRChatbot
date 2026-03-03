#!/usr/bin/env python3
"""
scripts/export_annotation_samples.py

从数据库导出三类人工标注任务样本：

  实验3  Style 感知验证：40 条 bot 回复 + 重建的 6D 风格参数
         → annotation_samples_style.csv

  实验4  Move 人机理解一致性：40 条对话 + 近期上下文 + 8 种 move 描述
         → annotation_samples_move.csv

  实验5  Judge 选优验证：40 条 bot 回复 + 上下文（供标注员评分）
         → annotation_samples_judge.csv

使用说明：
  export DATABASE_URL="postgresql+asyncpg://..."
  python scripts/export_annotation_samples.py [--limit 40] [--seed 42]

注意：
  - Move 的 selected_content_move_ids 仅存在于运行日志，数据库中不持久化。
    本脚本标注包含「系统实际选取的 move（如可从日志回填）」列，默认为空，
    需在导出后人工或通过日志解析填入。
  - Judge 候选池同样在日志中。本脚本导出最终 bot 回复 + 上下文，
    供标注员评估回复质量（相对于「随机/固定替代」基线）。
"""
from __future__ import annotations

import argparse
import asyncio
import csv
import json
import os
import random
import sys
from typing import Any, Dict, List, Optional, Tuple

# ── 路径初始化 ──────────────────────────────────────────────────────────────
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_ROOT)

OUTPUT_DIR = os.path.join(PROJECT_ROOT, "scripts", "output")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── 依赖（可选延迟导入）────────────────────────────────────────────────────
try:
    from sqlalchemy import text
    from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
    from sqlalchemy.orm import sessionmaker
    _SA_AVAILABLE = True
except ImportError:
    _SA_AVAILABLE = False

try:
    from app.nodes.pipeline.style import Inputs, compute_style_keys
    _STYLE_AVAILABLE = True
except Exception:
    _STYLE_AVAILABLE = False

try:
    from utils.yaml_loader import load_pure_content_transformations
    _MOVES_AVAILABLE = True
except Exception:
    _MOVES_AVAILABLE = False


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  数据库查询                                                             ║
# ╚══════════════════════════════════════════════════════════════════════════╝

FETCH_SQL = """
SELECT
    t.id                                              AS transcript_id,
    t.created_at                                      AS created_at,
    t.user_text                                       AS user_text,
    t.bot_text                                        AS bot_text,
    t.topic                                           AS topic,
    t.importance                                      AS importance,
    t.short_context                                   AS short_context,
    t.session_id                                      AS session_id,
    t.turn_index                                      AS turn_index,
    u.id                                              AS user_id,
    u.current_stage                                   AS current_stage,
    u.dimensions                                      AS dimensions,
    u.inferred_profile                                AS inferred_profile,
    b.id                                              AS bot_id,
    b.name                                            AS bot_name,
    b.big_five                                        AS big_five,
    b.mood_state                                      AS mood_state
FROM transcripts  t
JOIN users  u ON t.user_id = u.id
JOIN bots   b ON u.bot_id  = b.id
WHERE
    LENGTH(TRIM(t.user_text)) > 0
    AND LENGTH(TRIM(t.bot_text)) > 0
ORDER BY t.created_at DESC
LIMIT :pool_limit
"""

CONTEXT_SQL = """
SELECT user_text, bot_text, created_at
FROM transcripts
WHERE user_id = :uid
  AND created_at < :before
ORDER BY created_at DESC
LIMIT 5
"""


async def fetch_pool(db_url: str, pool_limit: int = 400) -> List[Dict[str, Any]]:
    """从 DB 拉取候选池（最新 pool_limit 条有效对话轮次）。"""
    engine = create_async_engine(db_url, echo=False, future=True)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    rows = []
    async with AsyncSessionLocal() as session:
        result = await session.execute(text(FETCH_SQL), {"pool_limit": pool_limit})
        for row in result.mappings():
            rows.append(dict(row))
    await engine.dispose()
    return rows


async def fetch_context(db_url: str, user_id: str, before_dt: Any) -> List[Dict]:
    """获取某用户在 before_dt 之前的最近 5 轮对话（用于上下文窗口）。"""
    engine = create_async_engine(db_url, echo=False, future=True)
    AsyncSessionLocal = sessionmaker(engine, class_=AsyncSession, expire_on_commit=False)
    rows = []
    async with AsyncSessionLocal() as session:
        result = await session.execute(
            text(CONTEXT_SQL),
            {"uid": user_id, "before": before_dt},
        )
        for r in result.mappings():
            rows.append(dict(r))
    await engine.dispose()
    return list(reversed(rows))  # 时间正序


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Style 参数重建                                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def _f(d: Dict, key: str, default: float = 0.5) -> float:
    try:
        v = d.get(key)
        if v is None:
            return default
        return min(1.0, max(0.0, float(v)))
    except Exception:
        return default


def reconstruct_style(row: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    """从 DB 行重建 Style 6D 参数（当前状态近似）。"""
    if not _STYLE_AVAILABLE:
        return None
    bf = row.get("big_five") or {}
    ms = row.get("mood_state") or {}
    rel = row.get("dimensions") or {}

    pad_scale = str(ms.get("pad_scale") or "m1_1")

    def _pad(v: Any) -> float:
        if v is None:
            return 0.5
        try:
            x = float(v)
            if pad_scale == "0_1":
                return min(1.0, max(0.0, x))
            return min(1.0, max(0.0, (x + 1.0) / 2.0))
        except Exception:
            return 0.5

    inp = Inputs(
        E=_f(bf, "extraversion"),
        A=_f(bf, "agreeableness"),
        C=_f(bf, "conscientiousness"),
        O=_f(bf, "openness"),
        N=_f(bf, "neuroticism"),
        P=_pad(ms.get("pleasure")),
        Ar=_pad(ms.get("arousal")),
        D=_pad(ms.get("dominance")),
        busy=_f(ms, "busyness"),
        momentum=0.5,       # DB 中无持久化，用中性值
        topic_appeal=0.5,   # DB 中无持久化，用中性值
        closeness=_f(rel, "closeness"),
        trust=_f(rel, "trust"),
        liking=_f(rel, "liking"),
        respect=_f(rel, "respect"),
        attractiveness=_f(rel, "attractiveness", default=_f(rel, "liking")),
        power=_f(rel, "power"),
        evidence=None,
    )
    try:
        return compute_style_keys(inp)
    except Exception:
        return None


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  Content Move 描述加载                                                  ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def load_moves_desc() -> List[Dict[str, Any]]:
    """从 content_moves.yaml 加载 8 种 move 的描述，供标注任务使用。"""
    if not _MOVES_AVAILABLE:
        return []
    try:
        raw = load_pure_content_transformations()
        if isinstance(raw, dict):
            raw = raw.get("pure_content_transformations") or []
        if not isinstance(raw, list):
            return []
        return [m for m in raw if isinstance(m, dict) and m.get("id") is not None]
    except Exception:
        return []


def moves_to_reference_str(moves: List[Dict]) -> str:
    """将 move 列表转换为标注员可读的参考文本。"""
    lines = []
    for m in moves:
        mid = m.get("id", "?")
        name = str(m.get("name") or "").strip()
        op = str(m.get("content_operation") or "").strip()[:120]
        lines.append(f"Move {mid}: {name} —— {op}")
    return "\n".join(lines) if lines else "(未能加载 move 描述)"


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  导出函数                                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝

def export_style_samples(samples: List[Dict], moves_desc: str) -> str:
    """
    导出实验3：Style 感知验证标注任务（研究用 Likert 格式）。
    标注员对 bot_text 在 5 个维度评分（1-7 Likert）。
    """
    path = os.path.join(OUTPUT_DIR, "annotation_samples_style_research.csv")
    fieldnames = [
        "sample_id", "created_at", "current_stage",
        "user_text", "bot_text",
        # 重建的 Style 参数（供研究者分析，标注员不看）
        "FORMALITY", "POLITENESS", "WARMTH", "CERTAINTY",
        "EXPRESSION_MODE", "CHAT_MARKERS",
        # 关系维度快照
        "closeness", "trust", "liking",
        # 标注员填写列（1-7 Likert）
        "ann_formality_1to7",    # 1=非常随意 7=非常正式
        "ann_politeness_1to7",   # 1=非常直接 7=非常礼貌
        "ann_warmth_1to7",       # 1=非常冷淡 7=非常温暖
        "ann_certainty_1to7",    # 1=非常犹豫 7=非常笃定
        "ann_chat_markers_1to7", # 1=非常书面 7=非常口语化
        "ann_notes",             # 备注
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            style = s.get("_style") or {}
            rel = s.get("dimensions") or {}
            writer.writerow({
                "sample_id": str(s.get("transcript_id", ""))[:8],
                "created_at": str(s.get("created_at", ""))[:16],
                "current_stage": s.get("current_stage", ""),
                "user_text": str(s.get("user_text", "")).replace("\n", " "),
                "bot_text": str(s.get("bot_text", "")).replace("\n", " "),
                "FORMALITY": round(float(style.get("FORMALITY", 0)), 3) if style else "",
                "POLITENESS": round(float(style.get("POLITENESS", 0)), 3) if style else "",
                "WARMTH": round(float(style.get("WARMTH", 0)), 3) if style else "",
                "CERTAINTY": round(float(style.get("CERTAINTY", 0)), 3) if style else "",
                "EXPRESSION_MODE": style.get("EXPRESSION_MODE", "") if style else "",
                "CHAT_MARKERS": round(float(style.get("CHAT_MARKERS", 0)), 3) if style else "",
                "closeness": round(_f(rel, "closeness"), 3),
                "trust": round(_f(rel, "trust"), 3),
                "liking": round(_f(rel, "liking"), 3),
                "ann_formality_1to7": "",
                "ann_politeness_1to7": "",
                "ann_warmth_1to7": "",
                "ann_certainty_1to7": "",
                "ann_chat_markers_1to7": "",
                "ann_notes": "",
            })
    print(f"  → Style 标注样本：{path}  ({len(samples)} 条)")
    return path


def export_move_samples(samples: List[Dict], moves_desc: str) -> str:
    """
    导出实验4：Move 人机理解一致性验证标注任务（研究用完整上下文格式）。
    标注员阅读对话上下文，选择最适合的 1-3 个 move。
    """
    path = os.path.join(OUTPUT_DIR, "annotation_samples_move_research.csv")
    fieldnames = [
        "sample_id", "created_at", "current_stage",
        # 近期上下文（最多 3 轮）
        "context_t_minus_3", "context_bot_minus_3",
        "context_t_minus_2", "context_bot_minus_2",
        "context_t_minus_1", "context_bot_minus_1",
        # 当轮用户消息
        "user_text",
        # 系统实际回复（参考）
        "bot_text",
        # 系统选取的 move（需从日志回填；此处默认空）
        "system_move_ids",
        # 标注员选择列
        "ann_move_ids",   # 标注员认为最适合的 move id（1-8，可多选，逗号分隔）
        "ann_notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        # 在文件开头嵌入 Move 参考表（作为 CSV 注释行）
        for line in moves_desc.split("\n"):
            f.write(f"# {line}\n")
        f.write("#\n# 标注说明：请在 ann_move_ids 列填写你认为本轮最适合的 move id\n# （1-8，可多选，逗号分隔；例如：1,3）\n#\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            ctx = s.get("_context", [])

            def _ctx(i: int, role: str) -> str:
                idx = len(ctx) - 3 + i  # i=0,1,2 → ctx[-3], ctx[-2], ctx[-1]
                if idx < 0 or idx >= len(ctx):
                    return ""
                return str(ctx[idx].get("user_text" if role == "user" else "bot_text", "")).replace("\n", " ")

            writer.writerow({
                "sample_id": str(s.get("transcript_id", ""))[:8],
                "created_at": str(s.get("created_at", ""))[:16],
                "current_stage": s.get("current_stage", ""),
                "context_t_minus_3": _ctx(0, "user"),
                "context_bot_minus_3": _ctx(0, "bot"),
                "context_t_minus_2": _ctx(1, "user"),
                "context_bot_minus_2": _ctx(1, "bot"),
                "context_t_minus_1": _ctx(2, "user"),
                "context_bot_minus_1": _ctx(2, "bot"),
                "user_text": str(s.get("user_text", "")).replace("\n", " "),
                "bot_text": str(s.get("bot_text", "")).replace("\n", " "),
                "system_move_ids": "",  # 需从日志回填
                "ann_move_ids": "",
                "ann_notes": "",
            })
    print(f"  → Move 标注样本：{path}  ({len(samples)} 条)")
    return path


def export_judge_samples(samples: List[Dict]) -> str:
    """
    导出实验5：Judge 选优验证标注任务。
    标注员对 bot 回复在三个维度评分，最终比较 Judge 选 vs 随机选 vs 固定路由选。

    注意：Judge 候选池（5路×4条=20候选）不在 DB 中；
    本脚本仅导出 Judge 实际选取的回复（transcripts.bot_text）供质量评分。
    若需完整候选池对照，需从日志中提取后手动补入 candidate_A/B 列。
    """
    path = os.path.join(OUTPUT_DIR, "annotation_samples_judge.csv")
    fieldnames = [
        "sample_id", "created_at", "current_stage",
        # 近期上下文（最多 2 轮）
        "context_t_minus_2", "context_bot_minus_2",
        "context_t_minus_1", "context_bot_minus_1",
        # 当轮
        "user_text",
        # Judge 实际选取的回复（Response C）
        "response_judge",
        # 需从日志补充的对照回复（待研究者填入）
        "response_random",   # 候选池中随机选1条（从日志提取后填入）
        "response_free",     # FREE 路由固定选第0条（从日志提取后填入）
        # 标注员评分列（1-5 Likert）
        "ann_judge_context_fit",   # 情景契合度
        "ann_judge_naturalness",   # 自然/人味
        "ann_judge_emotion_fit",   # 情绪贴合
        "ann_random_context_fit",
        "ann_random_naturalness",
        "ann_random_emotion_fit",
        "ann_free_context_fit",
        "ann_free_naturalness",
        "ann_free_emotion_fit",
        "ann_notes",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        f.write("# 标注说明：请对 response_judge / response_random / response_free\n")
        f.write("# 在情景契合度/自然人味/情绪贴合三个维度各打 1-5 分（1最低 5最高）\n")
        f.write("# response_random 和 response_free 需由研究者从日志中补入后再发给标注员\n#\n")
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            ctx = s.get("_context", [])

            def _ctx(i: int, role: str) -> str:
                idx = len(ctx) - 2 + i  # i=0,1 → ctx[-2], ctx[-1]
                if idx < 0 or idx >= len(ctx):
                    return ""
                return str(ctx[idx].get("user_text" if role == "user" else "bot_text", "")).replace("\n", " ")

            writer.writerow({
                "sample_id": str(s.get("transcript_id", ""))[:8],
                "created_at": str(s.get("created_at", ""))[:16],
                "current_stage": s.get("current_stage", ""),
                "context_t_minus_2": _ctx(0, "user"),
                "context_bot_minus_2": _ctx(0, "bot"),
                "context_t_minus_1": _ctx(1, "user"),
                "context_bot_minus_1": _ctx(1, "bot"),
                "user_text": str(s.get("user_text", "")).replace("\n", " "),
                "response_judge": str(s.get("bot_text", "")).replace("\n", " "),
                "response_random": "",   # 待从日志补入
                "response_free": "",     # 待从日志补入
                "ann_judge_context_fit": "",
                "ann_judge_naturalness": "",
                "ann_judge_emotion_fit": "",
                "ann_random_context_fit": "",
                "ann_random_naturalness": "",
                "ann_random_emotion_fit": "",
                "ann_free_context_fit": "",
                "ann_free_naturalness": "",
                "ann_free_emotion_fit": "",
                "ann_notes": "",
            })
    print(f"  → Judge 标注样本：{path}  ({len(samples)} 条)")
    return path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  标注网页专用 CSV 导出（简洁格式，供 /annotation/ 页面使用）              ║
# ╚══════════════════════════════════════════════════════════════════════════╝

_WEBAPP_STYLE_DIMS = ["FORMALITY", "POLITENESS", "WARMTH", "CERTAINTY", "EMOTIONAL_INTENSITY"]
_WEBAPP_DIM_KEY: Dict[str, str] = {
    "FORMALITY":           "formality",
    "POLITENESS":          "politeness",
    "WARMTH":              "warmth",
    "CERTAINTY":           "certainty",
    "EMOTIONAL_INTENSITY": "emotional_intensity",
}


# 与 prompt_utils._STYLE_VALUE_TO_LABEL 保持一致的五档体系
_STYLE_TIERS = [
    (0.86, 1.01, "extremely_high"),
    (0.61, 0.86, "high"),
    (0.41, 0.61, "mid"),
    (0.16, 0.41, "low"),
    (0.00, 0.16, "extremely_low"),
]
# 档位编号：extremely_high=0, high=1, mid=2, low=3, extremely_low=4
_TIER_RANK: Dict[str, int] = {label: i for i, (_, _, label) in enumerate(_STYLE_TIERS)}
# 档位距离 → diff_bucket 标签（沿用相同五档词汇中的4个）
_TIER_DIST_LABEL: Dict[int, str] = {1: "low", 2: "mid", 3: "high", 4: "extremely_high"}
_DIFF_BUCKETS = ["low", "mid", "high", "extremely_high"]


def _value_to_tier(v: float) -> str:
    """将 0-1 的 style 值映射到五档标签（与 prompt_utils._STYLE_VALUE_TO_LABEL 一致）。"""
    for lo, hi, label in _STYLE_TIERS:
        if lo <= v < hi:
            return label
    return "mid"


def _diff_bucket(tier_a: str, tier_b: str) -> Optional[str]:
    """
    计算两个档位标签的距离，返回 diff_bucket 标签。
    距离为 0（同档）时返回 None（调用方应跳过该配对）。
    """
    dist = abs(_TIER_RANK[tier_a] - _TIER_RANK[tier_b])
    if dist == 0:
        return None
    return _TIER_DIST_LABEL.get(dist, "high")


def export_webapp_move_samples(samples: List[Dict]) -> str:
    """标注网页 Move 任务 CSV：sample_id / context_user_text / bot_text / system_move_ids。"""
    path = os.path.join(OUTPUT_DIR, "annotation_samples_move.csv")
    fieldnames = ["sample_id", "context_user_text", "bot_text", "system_move_ids"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            writer.writerow({
                "sample_id":          str(s.get("transcript_id", ""))[:8],
                "context_user_text":  str(s.get("user_text", "")).replace("\n", " "),
                "bot_text":           str(s.get("bot_text", "")).replace("\n", " "),
                "system_move_ids":    "",  # 需从日志回填
            })
    print(f"  → [Webapp] Move 任务 CSV：{path}  ({len(samples)} 条)")
    return path


def export_webapp_expr_mode_samples(samples: List[Dict]) -> str:
    """标注网页 ExprMode 任务 CSV：sample_id / context_user_text / bot_text / system_expr_mode。"""
    path = os.path.join(OUTPUT_DIR, "annotation_samples_expr_mode.csv")
    fieldnames = ["sample_id", "context_user_text", "bot_text", "system_expr_mode"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for s in samples:
            style = s.get("_style") or {}
            writer.writerow({
                "sample_id":         str(s.get("transcript_id", ""))[:8],
                "context_user_text": str(s.get("user_text", "")).replace("\n", " "),
                "bot_text":          str(s.get("bot_text", "")).replace("\n", " "),
                "system_expr_mode":  int(style.get("EXPRESSION_MODE", 0)) if style else "",
            })
    print(f"  → [Webapp] ExprMode 任务 CSV：{path}  ({len(samples)} 条)")
    return path


def export_webapp_style_pairs(
    samples_with_style: List[Dict],
    rng: random.Random,
    n_pairs: int,
) -> str:
    """
    标注网页 Style 配对任务 CSV。
    每行 = 一道对比题（两组对话对 + 一个维度问题）。

    约束：
      - 两组该维度取值不同（diff ≥ 0.05）
      - diff_bucket（extremely_low/low/medium/high）尽量均匀分配
      - 五个维度尽量均匀分配
      - A/B 位置 50% 随机打乱
    """
    valid = [s for s in samples_with_style if s.get("_style")]
    if len(valid) < 2:
        print("  [Webapp] Style 配对：有效样本不足 2 条，跳过")
        return ""

    # 按 (维度, 档位距离标签) 收集候选配对
    candidates: Dict[Tuple[str, str], List[Dict]] = {}
    for i in range(len(valid)):
        for j in range(i + 1, len(valid)):
            a, b = valid[i], valid[j]
            sa, sb = a["_style"], b["_style"]
            for dim in _WEBAPP_STYLE_DIMS:
                va = float(sa.get(dim, 0.5))
                vb = float(sb.get(dim, 0.5))
                tier_a = _value_to_tier(va)
                tier_b = _value_to_tier(vb)
                bucket = _diff_bucket(tier_a, tier_b)
                if bucket is None:
                    continue  # 同档位，差异不可辨，跳过
                row = {
                    "task_id":        None,
                    "dimension":      _WEBAPP_DIM_KEY[dim],
                    "text_a_user":    str(a.get("user_text", "")).replace("\n", " "),
                    "text_a_bot":     str(a.get("bot_text", "")).replace("\n", " "),
                    "text_b_user":    str(b.get("user_text", "")).replace("\n", " "),
                    "text_b_bot":     str(b.get("bot_text", "")).replace("\n", " "),
                    "system_value_a": round(va, 3),
                    "system_label_a": tier_a,
                    "system_value_b": round(vb, 3),
                    "system_label_b": tier_b,
                    "diff_bucket":    bucket,
                }
                candidates.setdefault((dim, bucket), []).append(row)

    for rows in candidates.values():
        rng.shuffle(rows)

    # 每个 (dim×bucket) 格子均匀抽取
    per_bin = max(1, n_pairs // (len(_WEBAPP_STYLE_DIMS) * len(_DIFF_BUCKETS)))
    selected: List[Dict] = []
    used_pairs: set = set()

    for dim in _WEBAPP_STYLE_DIMS:
        for bucket in _DIFF_BUCKETS:
            pool = candidates.get((dim, bucket), [])
            for r in pool:
                if len([x for x in selected if x["dimension"] == _WEBAPP_DIM_KEY[dim] and x["diff_bucket"] == bucket]) >= per_bin:
                    break
                key = tuple(sorted([r["text_a_bot"][:30], r["text_b_bot"][:30]]))
                if key not in used_pairs:
                    used_pairs.add(key)
                    selected.append(r)

    # 不够 n_pairs 时从剩余候选中补充
    if len(selected) < n_pairs:
        all_cands = [r for rows in candidates.values() for r in rows]
        rng.shuffle(all_cands)
        for r in all_cands:
            if len(selected) >= n_pairs:
                break
            key = tuple(sorted([r["text_a_bot"][:30], r["text_b_bot"][:30]]))
            if key not in used_pairs:
                used_pairs.add(key)
                selected.append(r)

    rng.shuffle(selected)
    selected = selected[:n_pairs]

    # 随机打乱 A/B 顺序（50%）
    for row in selected:
        if rng.random() < 0.5:
            row["text_a_user"], row["text_b_user"] = row["text_b_user"], row["text_a_user"]
            row["text_a_bot"],  row["text_b_bot"]  = row["text_b_bot"],  row["text_a_bot"]
            row["system_value_a"], row["system_value_b"] = row["system_value_b"], row["system_value_a"]

    # 分配 task_id
    for idx, row in enumerate(selected):
        row["task_id"] = f"ST{idx + 1:04d}"

    path = os.path.join(OUTPUT_DIR, "annotation_samples_style.csv")
    fieldnames = [
        "task_id", "dimension",
        "text_a_user", "text_a_bot",
        "text_b_user", "text_b_bot",
        "system_value_a", "system_label_a",
        "system_value_b", "system_label_b",
        "diff_bucket",
    ]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in selected:
            writer.writerow(row)

    dim_dist: Dict[str, int] = {}
    bucket_dist: Dict[str, int] = {}
    for row in selected:
        dim_dist[row["dimension"]] = dim_dist.get(row["dimension"], 0) + 1
        bucket_dist[row["diff_bucket"]] = bucket_dist.get(row["diff_bucket"], 0) + 1
    print(f"  → [Webapp] Style 配对 CSV：{path}  ({len(selected)} 对)")
    print(f"    维度分布：{dim_dist}")
    print(f"    差值桶分布：{bucket_dist}")
    return path


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  主流程                                                                 ║
# ╚══════════════════════════════════════════════════════════════════════════╝

async def main_async(db_url: str, n_samples: int, seed: int) -> None:
    rng = random.Random(seed)

    print("  拉取候选池……")
    pool = await fetch_pool(db_url, pool_limit=max(400, n_samples * 10))
    print(f"  候选池大小：{len(pool)} 条")

    if len(pool) < n_samples:
        print(f"  [警告] 候选池仅 {len(pool)} 条，少于请求的 {n_samples} 条，全部导出。")
        n_samples = len(pool)

    if n_samples == 0:
        print("  [错误] 数据库中无有效对话记录，请检查数据或重新运行系统以积累对话。")
        sys.exit(1)

    # ── 分层抽样：按关系阶段分层，尽量覆盖不同阶段 ──────────────────────
    stage_groups: Dict[str, List[Dict]] = {}
    for row in pool:
        s = str(row.get("current_stage") or "unknown")
        stage_groups.setdefault(s, []).append(row)

    print(f"  阶段分布：{ {k: len(v) for k, v in stage_groups.items()} }")

    # 先随机打乱各组，再按比例抽取
    for g in stage_groups.values():
        rng.shuffle(g)

    # 总样本池抽取 3×n_samples（Style/Move/Judge 各需 n_samples，允许重叠）
    # 策略：按阶段比例保证均匀性
    all_staged: List[Dict] = []
    stage_keys = sorted(stage_groups.keys())
    per_stage = max(1, (3 * n_samples) // max(1, len(stage_keys)))
    for k in stage_keys:
        all_staged.extend(stage_groups[k][:per_stage])

    # 补充到 3*n_samples
    rng.shuffle(all_staged)
    all_pool = all_staged + [r for r in pool if r not in all_staged]
    rng.shuffle(all_pool)

    style_samples = all_pool[:n_samples]
    move_samples  = all_pool[n_samples: 2 * n_samples]
    judge_samples = all_pool[2 * n_samples: 3 * n_samples]

    # 若总数不够三组，允许重叠
    if len(all_pool) < 3 * n_samples:
        rng.shuffle(all_pool)
        style_samples = all_pool[:min(n_samples, len(all_pool))]
        rng.shuffle(all_pool)
        move_samples  = all_pool[:min(n_samples, len(all_pool))]
        rng.shuffle(all_pool)
        judge_samples = all_pool[:min(n_samples, len(all_pool))]

    # ── 重建 Style 参数 ──────────────────────────────────────────────────
    print("  重建 Style 参数……")
    for row in style_samples:
        row["_style"] = reconstruct_style(row)

    # ── 拉取上下文 ────────────────────────────────────────────────────────
    print("  拉取对话上下文（Move + Judge）……")
    for row in move_samples + judge_samples:
        uid = str(row.get("user_id", ""))
        before = row.get("created_at")
        if uid and before:
            try:
                ctx = await fetch_context(db_url, uid, before)
                row["_context"] = ctx
            except Exception:
                row["_context"] = []
        else:
            row["_context"] = []

    # ── 加载 Move 描述 ────────────────────────────────────────────────────
    moves = load_moves_desc()
    moves_desc = moves_to_reference_str(moves)

    # ── 导出研究用 CSV（Likert 格式）──────────────────────────────────────
    print("\n  导出研究用标注文件……")
    path_style = export_style_samples(style_samples, moves_desc)
    path_move  = export_move_samples(move_samples, moves_desc)
    path_judge = export_judge_samples(judge_samples)

    # ── 导出 Webapp 标注任务 CSV ──────────────────────────────────────────
    print("\n  导出 Webapp 标注任务 CSV……")
    path_webapp_move      = export_webapp_move_samples(move_samples)
    path_webapp_expr_mode = export_webapp_expr_mode_samples(style_samples)
    path_webapp_style     = export_webapp_style_pairs(style_samples, rng, n_samples)

    # ── 元数据 JSON ───────────────────────────────────────────────────────
    meta = {
        "seed": seed,
        "n_samples_each": n_samples,
        "pool_size": len(pool),
        "stage_distribution": {k: len(v) for k, v in stage_groups.items()},
        "moves_reference": moves_desc,
        "files": {
            "style_research": path_style,
            "move_research": path_move,
            "judge": path_judge,
            "webapp_move": path_webapp_move,
            "webapp_expr_mode": path_webapp_expr_mode,
            "webapp_style_pairs": path_webapp_style,
        },
        "notes": {
            "move_system_ids": (
                "system_move_ids 列当前为空，需研究者从运行日志中提取"
                "「[Extract] emotion_tag=... move_ids=[...]」行回填。"
            ),
            "judge_baselines": (
                "response_random/response_free 列当前为空，需研究者从运行日志中提取"
                "「[Generate]」和「[Judge]」相关行回填候选文本后再发给标注员。"
            ),
            "style_momentum_topic_appeal": (
                "Style 重建时 momentum=0.5, topic_appeal=0.5（DB 中无持久化），"
                "实际值会有小幅偏差，主要影响 CHAT_MARKERS。"
            ),
        },
    }
    meta_path = os.path.join(OUTPUT_DIR, "annotation_metadata.json")
    with open(meta_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)
    print(f"  → 元数据：{meta_path}")

    print("\n  完成！下一步操作：")
    print("  1. 访问 /annotation/ 页面，使用 Webapp 标注任务 CSV 进行标注")
    print("  2. 向 annotation_samples_move_research.csv 的 system_move_ids 列回填系统实际选取的 move ID")
    print("  3. 向 annotation_samples_judge.csv 的 response_random/free 列补入候选文本")
    print("  4. 标注完成后使用 Spearman/Kappa 计算标注员间一致性")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="从数据库导出三类人工标注任务样本",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "--db-url",
        default=os.environ.get("DATABASE_URL", ""),
        help="PostgreSQL 连接 URL（默认读取 DATABASE_URL 环境变量）",
    )
    parser.add_argument("--limit", type=int, default=40, help="每类导出样本数（默认 40）")
    parser.add_argument("--seed", type=int, default=42, help="随机种子（默认 42）")
    args = parser.parse_args()

    if not _SA_AVAILABLE:
        print("错误：未安装 SQLAlchemy + asyncpg。请运行：pip install sqlalchemy asyncpg")
        sys.exit(1)

    if not args.db_url:
        print("错误：未提供数据库 URL。")
        print("请设置环境变量：export DATABASE_URL='postgresql+asyncpg://user:pass@host/db'")
        print("或传入参数：--db-url 'postgresql+asyncpg://...'")
        sys.exit(1)

    # asyncpg URL 格式归一化
    db_url = args.db_url.strip()
    if db_url.startswith("postgres://") or db_url.startswith("postgresql://"):
        if "+asyncpg" not in db_url:
            db_url = db_url.replace("postgresql://", "postgresql+asyncpg://", 1)
            db_url = db_url.replace("postgres://", "postgresql+asyncpg://", 1)

    print("=" * 70)
    print("  标注样本导出脚本")
    print(f"  每类样本数：{args.limit}  随机种子：{args.seed}")
    print("=" * 70)

    asyncio.run(main_async(db_url, args.limit, args.seed))
    print("\n  全部完成 ✓")


if __name__ == "__main__":
    main()
