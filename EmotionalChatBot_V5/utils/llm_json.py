"""从 LLM 返回的文本中稳健解析 JSON（支持纯 JSON、markdown 代码块、前后杂文）。"""
from __future__ import annotations

import json
import re
from typing import Any, Dict, Optional


def parse_json_from_llm(text: str) -> Optional[Dict[str, Any]]:
    """
    尝试从 LLM 输出中解析 JSON。
    - 先直接 json.loads
    - 再尝试 strip 后解析
    - 再尝试从 ```json ... ``` 或 ``` ... ``` 中提取
    - 再尝试从第一个 { 到最后一个 } 截取后解析
    """
    if not text or not isinstance(text, str):
        return None
    raw = text.strip()
    if not raw:
        return None

    # 1. 直接解析
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass

    # 2. 从 markdown 代码块提取
    for pattern in (
        r"```(?:json)?\s*([\s\S]*?)\s*```",
        r"```\s*([\s\S]*?)\s*```",
    ):
        m = re.search(pattern, raw)
        if m:
            try:
                return json.loads(m.group(1).strip())
            except json.JSONDecodeError:
                pass

    # 3. 从第一个 { 到最后一个 } 截取
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1 and end > start:
        try:
            return json.loads(raw[start : end + 1])
        except json.JSONDecodeError:
            pass

    return None
