from __future__ import annotations

import re
from typing import List, Tuple

# 这些模式一旦出现在“外部对话文本”（user_input / final_response），几乎可以判定为：
# - internal prompt 泄漏
# - debug dump 混入
# - LATS/evaluator 指令串入
#
# 命中后应当：拒写入 external 通道（DB/chat_buffer），并报警/中断压测。
_INTERNAL_LEAK_PATTERNS: List[re.Pattern] = [
    re.compile(r"\bjson schema\b", re.IGNORECASE),
    re.compile(r"请严格输出\s*json", re.IGNORECASE),
    re.compile(r"score_breakdown", re.IGNORECASE),
    re.compile(r"plan_alignment_details", re.IGNORECASE),
    re.compile(r"style_dim_report", re.IGNORECASE),
    re.compile(r"stage_act_report", re.IGNORECASE),
    re.compile(r"memory_report", re.IGNORECASE),
    re.compile(r"\breplyplan\b", re.IGNORECASE),
    re.compile(r"\breply_plan\b", re.IGNORECASE),
    re.compile(r"\bmust_cover_map\b", re.IGNORECASE),
    re.compile(r"\bstage_targets\b", re.IGNORECASE),
    re.compile(r"\bstyle_targets\b", re.IGNORECASE),
    re.compile(r"\bplan_goals\b", re.IGNORECASE),
    re.compile(r"解析结果\s*\(parsed\)", re.IGNORECASE),
    re.compile(r"llm\s*响应\s*\(raw\)", re.IGNORECASE),
    re.compile(r"\[Evaluator\b|\[LATS\b|\[ReplyPlanner\b", re.IGNORECASE),
]


def detect_internal_leak(text: str) -> Tuple[bool, List[str]]:
    """
    Returns: (is_leak, reasons)
    """
    s = str(text or "")
    reasons: List[str] = []
    for pat in _INTERNAL_LEAK_PATTERNS:
        if pat.search(s):
            reasons.append(pat.pattern)
    return (len(reasons) > 0), reasons


# LATS 多候选时模型可能把 "候选N：" 写进 reply 文本，需在写回 final_response 前去掉
_CANDIDATE_PREFIX = re.compile(r"^\s*候选\d+[：:]\s*", re.UNICODE)


def strip_candidate_prefix(text: str) -> str:
    """去掉句首「候选N：」或「候选N:」前缀，避免泄漏到用户可见回复。"""
    s = str(text or "").strip()
    return _CANDIDATE_PREFIX.sub("", s).strip() or s


def sanitize_external_text(text: str) -> str:
    """
    “外部文本”净化器：
    - 只做最小化清洗（strip）
    - 若命中 internal leak，则抛异常（压测/工具脚本应中止；线上可改为替换/降级策略）
    """
    s = str(text or "").strip()
    leak, reasons = detect_internal_leak(s)
    if leak:
        raise ValueError(f"external_text contaminated by internal patterns: {reasons[:3]}")
    return s

