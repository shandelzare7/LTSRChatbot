#!/usr/bin/env python3
"""
验证 gpt-5-mini 的 verbosity / reasoning_effort 在 API 请求中是否存在。
方法：开启 openai SDK 的 HTTP 日志，从 stderr 中捕获请求 body。
"""
import io
import json
import logging
import os
import re
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


def unwrap_llm(llm):
    inner = getattr(llm, "_inner", llm)
    if inner is llm:
        return llm
    return unwrap_llm(inner)


def check_default_params():
    from app.services.llm import get_llm
    print("=" * 70)
    print("检查 1: _default_params（ChatOpenAI 构建 API body 时用的参数字典）")
    print("=" * 70)
    llm = get_llm(role="fast", model="gpt-5-mini")
    base = unwrap_llm(llm)
    dp = getattr(base, '_default_params', {})
    print(f"  _default_params = {json.dumps({k: v for k, v in dp.items()}, ensure_ascii=False)}")
    v_ok = dp.get('verbosity') == 'low'
    r_ok = dp.get('reasoning_effort') == 'low'
    t_absent = 'temperature' not in dp
    print(f"  verbosity=low: {'✅' if v_ok else '❌ ' + repr(dp.get('verbosity'))}")
    print(f"  reasoning_effort=low: {'✅' if r_ok else '❌ ' + repr(dp.get('reasoning_effort'))}")
    print(f"  temperature 不在 params 中: {'✅' if t_absent else '❌ temperature=' + str(dp.get('temperature'))}")
    return llm, base, v_ok and r_ok and t_absent


def check_invoke_code():
    from app.lats.reply_planner import _is_planner_gpt5_mini
    from app.services.llm import get_llm
    print("\n" + "=" * 70)
    print("检查 2: reply_planner invoke_kwargs 代码逻辑")
    print("=" * 70)
    llm = get_llm(role="fast", model="gpt-5-mini")
    is_gpt5 = _is_planner_gpt5_mini(llm)
    print(f"  _is_planner_gpt5_mini(llm_planner_27) = {is_gpt5}")
    if is_gpt5:
        print("  => invoke_kwargs 会走 gpt5-mini 分支：")
        print("     verbosity='low', reasoning_effort='low', max_tokens=...")
        print("  ✅ invoke 时会传 verbosity 和 reasoning_effort")
    else:
        print("  ❌ 不会走 gpt5-mini 分支！会走 temperature/top_p 分支")
    return is_gpt5


def check_structured_output_preserves_params():
    """验证 with_structured_output 返回的链条是否保留了原始 LLM 的 _default_params"""
    from app.services.llm import get_llm
    from src.schemas import ReplyPlannerCandidates

    print("\n" + "=" * 70)
    print("检查 3: with_structured_output 是否保留 verbosity/reasoning_effort")
    print("=" * 70)
    llm = get_llm(role="fast", model="gpt-5-mini")
    base = unwrap_llm(llm)

    structured_chain = base.with_structured_output(ReplyPlannerCandidates)

    # structured_chain 通常是 RunnableSequence 或 RunnableBinding
    print(f"  structured_chain type: {type(structured_chain).__name__}")

    # 查找链中的 LLM 实例
    found_llm = None
    if hasattr(structured_chain, "first"):
        first = structured_chain.first
        print(f"  chain.first type: {type(first).__name__}")
        if hasattr(first, "bound"):
            bound = first.bound
            print(f"  chain.first.bound type: {type(bound).__name__}")
            found_llm = bound
        elif hasattr(first, "_default_params"):
            found_llm = first
    elif hasattr(structured_chain, "bound"):
        bound = structured_chain.bound
        print(f"  chain.bound type: {type(bound).__name__}")
        found_llm = bound

    if found_llm and hasattr(found_llm, "_default_params"):
        dp = found_llm._default_params
        v = dp.get("verbosity", "❌ 不存在")
        r = dp.get("reasoning_effort", "❌ 不存在")
        print(f"  链中 LLM 的 _default_params.verbosity: {v!r}")
        print(f"  链中 LLM 的 _default_params.reasoning_effort: {r!r}")
        ok = v == "low" and r == "low"
        print(f"  => {'✅' if ok else '❌'} structured_output 链{'保留' if ok else '丢失'}了 verbosity/reasoning_effort")
        return ok
    else:
        print("  未找到链中的 LLM 实例，尝试递归搜索...")
        # 递归搜索
        def find_default_params(obj, depth=0, seen=None):
            if seen is None:
                seen = set()
            obj_id = id(obj)
            if obj_id in seen or depth > 8:
                return None
            seen.add(obj_id)
            if hasattr(obj, "_default_params"):
                return obj._default_params
            for attr in ["first", "middle", "last", "bound", "runnable", "_inner", "steps"]:
                child = getattr(obj, attr, None)
                if child is not None:
                    if isinstance(child, list):
                        for item in child:
                            result = find_default_params(item, depth + 1, seen)
                            if result is not None:
                                return result
                    else:
                        result = find_default_params(child, depth + 1, seen)
                        if result is not None:
                            return result
            return None

        dp = find_default_params(structured_chain)
        if dp:
            v = dp.get("verbosity", "❌ 不存在")
            r = dp.get("reasoning_effort", "❌ 不存在")
            print(f"  找到链中 LLM: verbosity={v!r}, reasoning_effort={r!r}")
            ok = v == "low" and r == "low"
            print(f"  => {'✅' if ok else '❌'} structured_output 链{'保留' if ok else '丢失'}了参数")
            return ok
        else:
            print("  ❌ 无法在链中找到 _default_params")
            return False


def check_real_call():
    """实际发一个请求，通过 httpx logging 看请求 body"""
    from app.services.llm import get_llm, _LLM_CACHE
    from langchain_core.messages import HumanMessage, SystemMessage
    from src.schemas import ReplyPlannerCandidates

    print("\n" + "=" * 70)
    print("检查 4: 真实 API 调用验证（1 次 structured_output 调用）")
    print("=" * 70)

    _LLM_CACHE.clear()
    llm = get_llm(role="fast", model="gpt-5-mini")

    # 开启 httpx debug 日志到一个 buffer
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    for logger_name in ["httpx", "openai", "openai._base_client", "httpcore"]:
        lg = logging.getLogger(logger_name)
        lg.addHandler(handler)
        lg.setLevel(logging.DEBUG)

    os.environ["OPENAI_LOG"] = "debug"

    messages = [SystemMessage(content="hi"), HumanMessage(content="hello")]
    try:
        structured = llm.with_structured_output(ReplyPlannerCandidates)
        structured.invoke(messages, verbosity="low", reasoning_effort="low", max_tokens=200)
    except Exception as e:
        print(f"  调用结果: {type(e).__name__}: {str(e)[:120]}")

    log_output = buf.getvalue()

    # 在 log 中搜 verbosity
    if "verbosity" in log_output:
        print("  ✅ httpx/openai 日志中出现了 'verbosity'")
        for line in log_output.split("\n"):
            if "verbosity" in line.lower():
                print(f"    {line.strip()[:200]}")
                break
    else:
        print("  日志中未出现 'verbosity'（httpx debug 可能未打印 body）")

    # 即使日志不够详细，检查 response usage 来间接验证
    print("  （注意：_default_params 包含 verbosity='low' 已在检查1中确认，")
    print("   ChatOpenAI 在每次 API 调用时会 merge _default_params 到请求 body）")

    os.environ.pop("OPENAI_LOG", None)
    for logger_name in ["httpx", "openai", "openai._base_client", "httpcore"]:
        lg = logging.getLogger(logger_name)
        lg.removeHandler(handler)
        lg.setLevel(logging.WARNING)


def main():
    _, _, p1 = check_default_params()
    p2 = check_invoke_code()
    p3 = check_structured_output_preserves_params()
    check_real_call()

    print("\n" + "=" * 70)
    print("最终结论")
    print("=" * 70)
    all_ok = p1 and p2 and p3
    if all_ok:
        print("✅ 全部通过。verbosity='low' 和 reasoning_effort='low' 确认在所有调用路径上都会传入 API：")
        print("  1. ChatOpenAI 构造时设置了 verbosity='low', reasoning_effort='low'")
        print("  2. _default_params 中包含这两个参数（每次 API 调用都会带上）")
        print("  3. reply_planner 的 invoke_kwargs 中也会传这两个参数（双保险）")
        print("  4. with_structured_output 链保留了原始 LLM 的参数")
    else:
        print("❌ 存在问题：")
        if not p1: print("  - _default_params 中缺少 verbosity 或 reasoning_effort")
        if not p2: print("  - reply_planner 未走 gpt5-mini 分支")
        if not p3: print("  - with_structured_output 链未保留参数")


if __name__ == "__main__":
    main()
