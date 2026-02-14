"""
详细日志记录工具
用于记录每个环节的提示词、参数和运算过程
"""
import json
from typing import Any, Dict, List, Optional


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
        for i, msg in enumerate(messages):
            msg_type = getattr(msg, "type", type(msg).__name__)
            content = getattr(msg, "content", str(msg))
            # 截断过长的内容
            if len(content) > 500:
                content_preview = content[:500] + f"\n{indent}... (截断，总长度: {len(content)} 字符)"
            else:
                content_preview = content
            print(f"{prefix}{indent}[{i+1}] {msg_type}: {content_preview}")
        print(f"{prefix}{indent}{'=' * 60}")
    
    if params:
        print(f"{prefix}{indent}【输入参数】")
        print(f"{prefix}{indent}{'=' * 60}")
        for key, value in params.items():
            # 格式化复杂对象
            if isinstance(value, (dict, list)):
                try:
                    value_str = json.dumps(value, ensure_ascii=False, indent=2)
                    if len(value_str) > 500:
                        value_str = value_str[:500] + "\n... (截断)"
                except Exception:
                    value_str = str(value)[:200]
            else:
                value_str = str(value)
                if len(value_str) > 200:
                    value_str = value_str[:200] + "... (截断)"
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
    
    # 原始响应
    raw_content = getattr(raw_response, "content", str(raw_response))
    print(f"{prefix}{indent}【原始响应 (Raw)】")
    print(f"{prefix}{indent}{'=' * 60}")
    if len(raw_content) > 1000:
        print(f"{prefix}{indent}{raw_content[:1000]}")
        print(f"{prefix}{indent}... (截断，总长度: {len(raw_content)} 字符)")
    else:
        print(f"{prefix}{indent}{raw_content}")
    print(f"{prefix}{indent}{'=' * 60}")
    
    # 解析后的结果
    if parsed_result is not None:
        print(f"{prefix}{indent}【解析结果 (Parsed)】")
        print(f"{prefix}{indent}{'=' * 60}")
        try:
            if isinstance(parsed_result, (dict, list)):
                result_str = json.dumps(parsed_result, ensure_ascii=False, indent=2)
                if len(result_str) > 1000:
                    print(f"{prefix}{indent}{result_str[:1000]}")
                    print(f"{prefix}{indent}... (截断，总长度: {len(result_str)} 字符)")
                else:
                    print(f"{prefix}{indent}{result_str}")
            else:
                result_str = str(parsed_result)
                if len(result_str) > 500:
                    print(f"{prefix}{indent}{result_str[:500]}")
                    print(f"{prefix}{indent}... (截断)")
                else:
                    print(f"{prefix}{indent}{result_str}")
        except Exception as e:
            print(f"{prefix}{indent}[解析失败] {e}")
            print(f"{prefix}{indent}{str(parsed_result)[:200]}")
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


def _format_value(value: Any, max_length: int = 300) -> str:
    """格式化值用于日志输出"""
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
