"""
详细日志记录工具
用于记录每个环节的提示词、参数和运算过程
"""
import json
import os
from typing import Any, Dict, List, Optional

# 当 LTSR_FULL_PROMPT_LOG=1 或 BOT2BOT_FULL_LOGS=1 时，不截断提示词/响应，记录完整内容
def _full_logs() -> bool:
    return str(os.getenv("LTSR_FULL_PROMPT_LOG") or os.getenv("BOT2BOT_FULL_LOGS") or "").strip() in ("1", "true", "yes", "on")


def _truncate_limit() -> int:
    return 500_000 if _full_logs() else 500


def _truncate_limit_response() -> int:
    return 500_000 if _full_logs() else 1000


def _truncate_limit_params() -> int:
    return 500_000 if _full_logs() else 200


def log_prompt_and_params(
    node_name: str,
    system_prompt: Optional[str] = None,
    user_prompt: Optional[str] = None,
    messages: Optional[List[Any]] = None,
    params: Optional[Dict[str, Any]] = None,
    prefix: str = "",
):
    """记录提示词和参数"""
    indent = "  "
    print(f"{prefix}[{node_name}] ========== 提示词与参数 ==========")
    
    if system_prompt:
        print(f"{prefix}{indent}【System Prompt】")
        print(f"{prefix}{indent}{'=' * 60}")
        print(f"{prefix}{indent}{system_prompt}")
        print(f"{prefix}{indent}{'=' * 60}")
    
    if user_prompt:
        print(f"{prefix}{indent}【User Prompt / Task】")
        print(f"{prefix}{indent}{'=' * 60}")
        print(f"{prefix}{indent}{user_prompt}")
        print(f"{prefix}{indent}{'=' * 60}")
    
    if messages:
        print(f"{prefix}{indent}【Messages (Body)】")
        print(f"{prefix}{indent}{'=' * 60}")
        limit = _truncate_limit()
        for i, msg in enumerate(messages):
            msg_type = getattr(msg, "type", type(msg).__name__)
            content = getattr(msg, "content", str(msg))
            if len(content) > limit:
                content_preview = content[:limit] + f"\n{indent}... (截断，总长度: {len(content)} 字符)"
            else:
                content_preview = content
            print(f"{prefix}{indent}[{i+1}] {msg_type}: {content_preview}")
        print(f"{prefix}{indent}{'=' * 60}")
    
    if params:
        print(f"{prefix}{indent}【输入参数】")
        print(f"{prefix}{indent}{'=' * 60}")
        limit = _truncate_limit_params()
        for key, value in params.items():
            if isinstance(value, (dict, list)):
                try:
                    value_str = json.dumps(value, ensure_ascii=False, indent=2)
                    if len(value_str) > limit:
                        value_str = value_str[:limit] + "\n... (截断)"
                except Exception:
                    value_str = str(value)[:limit]
            else:
                value_str = str(value)
                if len(value_str) > limit:
                    value_str = value_str[:limit] + "... (截断)"
            print(f"{prefix}{indent}  {key}: {value_str}")
        print(f"{prefix}{indent}{'=' * 60}")
    
    print(f"{prefix}[{node_name}] ==========================================")


def log_llm_response(
    node_name: str,
    raw_response: Any,
    parsed_result: Optional[Any] = None,
    prefix: str = "",
):
    """记录 LLM 响应"""
    indent = "  "
    print(f"{prefix}[{node_name}] ========== LLM 响应 ==========")
    
    raw_content = getattr(raw_response, "content", str(raw_response))
    limit = _truncate_limit_response()
    print(f"{prefix}{indent}【原始响应 (Raw)】")
    print(f"{prefix}{indent}{'=' * 60}")
    if len(raw_content) > limit:
        print(f"{prefix}{indent}{raw_content[:limit]}")
        print(f"{prefix}{indent}... (截断，总长度: {len(raw_content)} 字符)")
    else:
        print(f"{prefix}{indent}{raw_content}")
    print(f"{prefix}{indent}{'=' * 60}")
    
    if parsed_result is not None:
        print(f"{prefix}{indent}【解析结果 (Parsed)】")
        print(f"{prefix}{indent}{'=' * 60}")
        try:
            if isinstance(parsed_result, (dict, list)):
                result_str = json.dumps(parsed_result, ensure_ascii=False, indent=2)
                if len(result_str) > limit:
                    print(f"{prefix}{indent}{result_str[:limit]}")
                    print(f"{prefix}{indent}... (截断，总长度: {len(result_str)} 字符)")
                else:
                    print(f"{prefix}{indent}{result_str}")
            else:
                result_str = str(parsed_result)
                if len(result_str) > limit:
                    print(f"{prefix}{indent}{result_str[:limit]}")
                    print(f"{prefix}{indent}... (截断)")
                else:
                    print(f"{prefix}{indent}{result_str}")
        except Exception as e:
            print(f"{prefix}{indent}[解析失败] {e}")
            print(f"{prefix}{indent}{str(parsed_result)[:limit]}")
        print(f"{prefix}{indent}{'=' * 60}")
    
    print(f"{prefix}[{node_name}] ==========================================")


def log_computation(
    node_name: str,
    step_name: str,
    inputs: Optional[Dict[str, Any]] = None,
    outputs: Optional[Dict[str, Any]] = None,
    intermediate_steps: Optional[List[Dict[str, Any]]] = None,
    prefix: str = "",
):
    """记录计算过程"""
    indent = "  "
    print(f"{prefix}[{node_name}] ========== 计算过程: {step_name} ==========")
    
    if inputs:
        print(f"{prefix}{indent}【输入】")
        print(f"{prefix}{indent}{'=' * 60}")
        for key, value in inputs.items():
            value_str = _format_value(value)
            print(f"{prefix}{indent}  {key}: {value_str}")
        print(f"{prefix}{indent}{'=' * 60}")
    
    if intermediate_steps:
        print(f"{prefix}{indent}【中间步骤】")
        print(f"{prefix}{indent}{'=' * 60}")
        for i, step in enumerate(intermediate_steps):
            print(f"{prefix}{indent}步骤 {i+1}:")
            for key, value in step.items():
                value_str = _format_value(value)
                print(f"{prefix}{indent}    {key}: {value_str}")
        print(f"{prefix}{indent}{'=' * 60}")
    
    if outputs:
        print(f"{prefix}{indent}【输出】")
        print(f"{prefix}{indent}{'=' * 60}")
        for key, value in outputs.items():
            value_str = _format_value(value)
            print(f"{prefix}{indent}  {key}: {value_str}")
        print(f"{prefix}{indent}{'=' * 60}")
    
    print(f"{prefix}[{node_name}] ==========================================")


def _format_value(value: Any, max_length: Optional[int] = None) -> str:
    """格式化值用于日志输出"""
    if max_length is None:
        max_length = 500_000 if _full_logs() else 300
    if isinstance(value, (dict, list)):
        try:
            value_str = json.dumps(value, ensure_ascii=False, indent=2)
            if len(value_str) > max_length:
                return value_str[:max_length] + f"\n... (截断，总长度: {len(value_str)} 字符)"
            return value_str
        except Exception:
            return str(value)[:max_length]
    else:
        value_str = str(value)
        if len(value_str) > max_length:
            return value_str[:max_length] + "... (截断)"
        return value_str
