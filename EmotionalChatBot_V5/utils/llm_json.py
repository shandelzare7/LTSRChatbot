"""从 LLM 返回的文本中稳健解析 JSON（支持纯 JSON、markdown 代码块、前后杂文）。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, List, Optional


def _normalize_parsed(obj: Any) -> Optional[Dict[str, Any]]:
    """
    将解析结果规范为 Dict，供各调用方使用：
    - 若已是 dict，原样返回（其他节点如 detection/task_planner 等依赖各自结构）；
    - 若为 list：单元素且含 "reply" 则转为 {"reply": ...}，多元素且每项含 "reply" 则转为 {"candidates": [...]}，
      否则无法规范为约定格式则返回 None。
    """
    if isinstance(obj, dict):
        return obj
    if isinstance(obj, list) and obj:
        if len(obj) == 1 and isinstance(obj[0], dict) and obj[0].get("reply") is not None:
            return {"reply": obj[0]["reply"]}
        if all(isinstance(x, dict) and "reply" in x for x in obj):
            return {"candidates": obj}
    return None


def _fix_smart_quotes(s: str) -> str:
    """将中文/智能引号替换为 ASCII 双引号，以便 JSON 解析。"""
    return s.replace("\u201c", '"').replace("\u201d", '"').replace("\u2018", "'").replace("\u2019", "'")


def _try_parse_one(s: str) -> Optional[Dict[str, Any]]:
    """对单段字符串尝试解析为 JSON 并规范为 Dict；支持去除尾部逗号、智能引号修复后重试。"""
    s = s.strip()
    if not s:
        return None
    # 1. 直接解析
    try:
        obj = json.loads(s)
        return _normalize_parsed(obj)
    except json.JSONDecodeError:
        pass
    # 2. 去除尾部逗号后重试（LLM 常犯）
    s2 = re.sub(r",\s*}", "}", s)
    s2 = re.sub(r",\s*]", "]", s2)
    try:
        obj = json.loads(s2)
        return _normalize_parsed(obj)
    except json.JSONDecodeError:
        pass
    # 3. 修复智能引号后重试（Qwen 常见）
    s3 = _fix_smart_quotes(s2)
    if s3 != s2:
        try:
            obj = json.loads(s3)
            return _normalize_parsed(obj)
        except json.JSONDecodeError:
            pass
    return None


def parse_json_from_llm(text: str) -> Optional[Dict[str, Any]]:
    """
    尝试从 LLM 输出中解析 JSON，并规范为含 "reply" 或 "candidates" 的 Dict。
    - 支持：纯 JSON、前后空白、markdown 代码块、首尾杂文。
    - 支持：尾部逗号、顶层为单元素数组 [{"reply":"..."}] 或多元素 [{"reply":"a"},...]。
    - 若内容像 HTML（如 API 返回 502/503 错误页），直接返回 None，避免 JSON 解析报错。
    """
    if not text or not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None
    # API 返回 HTML 错误页时不要尝试解析，直接返回 None，让调用方用默认值
    if raw.lstrip().upper().startswith("<!DOCTYPE") or (
        len(raw) > 10 and raw.lstrip().startswith("<") and "<?xml" not in raw[:20]
    ):
        return None

    # 1. 直接解析（含尾部逗号修复、顶层数组规范化）
    out = _try_parse_one(raw)
    if out is not None:
        return out

    # 2. 从 markdown 代码块提取（非贪婪，取第一个完整块）
    for pattern in (
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
    ):
        m = re.search(pattern, raw)
        if m:
            out = _try_parse_one(m.group(1))
            if out is not None:
                return out

    # 3. 从第一个 { 到最后一个 } 截取
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        out = _try_parse_one(raw[start : end + 1])
        if out is not None:
            return out

    # 4. 若整体像数组，尝试从 [ 到 ] 截取
    start_b = raw.find("[")
    end_b = raw.rfind("]")
    if start_b != -1 and end_b != -1 and end_b > start_b:
        out = _try_parse_one(raw[start_b : end_b + 1])
        if out is not None:
            return out

    return None
