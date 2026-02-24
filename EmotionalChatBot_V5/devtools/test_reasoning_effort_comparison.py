"""
对比 gpt-5-mini 在 reasoning_effort=low vs medium 下的表现差异。
测量指标：tokens (input/output/reasoning)、延迟、回复内容。
"""
from __future__ import annotations

import os
import sys
import time
import json
from typing import Any, Dict

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
sys.path.insert(0, PROJECT_ROOT)

from pathlib import Path
try:
    from utils.env_loader import load_project_env
    load_project_env(Path(PROJECT_ROOT))
except Exception:
    pass

from langchain_openai import ChatOpenAI
from langchain_core.messages import SystemMessage, HumanMessage
from pydantic import BaseModel, Field


class ReplyCandidate(BaseModel):
    reply: str = Field(description="候选回复")


class ReplyCandidates(BaseModel):
    candidates: list[ReplyCandidate] = Field(description="候选回复列表")


SYSTEM_PROMPT = """你是一个温暖有共情力的聊天伙伴「阿澈」，男性，25岁。
你正在和用户聊天，对方刚分享了最近的心情。
请根据对话给出3条候选回复，每条30字以内，语气自然、口语化。"""

USER_MSG = """用户说：最近工作压力好大，每天加班到很晚，感觉整个人都快撑不住了。而且领导还总是在群里@我，周末也没法好好休息。"""


def run_one_test(effort: str, run_id: int) -> Dict[str, Any]:
    """用指定 reasoning_effort 跑一次，返回统计数据。"""
    llm = ChatOpenAI(
        model="gpt-5-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        verbosity="low",
        reasoning_effort=effort,
    )

    messages = [SystemMessage(content=SYSTEM_PROMPT), HumanMessage(content=USER_MSG)]

    structured = llm.with_structured_output(ReplyCandidates)

    t0 = time.perf_counter()
    result = structured.invoke(messages, max_tokens=800)
    elapsed = time.perf_counter() - t0

    data = result.model_dump() if hasattr(result, "model_dump") else result.dict()

    usage = {}
    if hasattr(result, "response_metadata"):
        usage = result.response_metadata.get("token_usage", {})
    if not usage and hasattr(structured, "last_response"):
        meta = getattr(structured.last_response, "response_metadata", {})
        usage = meta.get("token_usage", {})

    return {
        "effort": effort,
        "run_id": run_id,
        "elapsed_s": round(elapsed, 3),
        "usage": usage,
        "candidates": data.get("candidates", []),
        "num_candidates": len(data.get("candidates", [])),
        "avg_reply_len": round(
            sum(len(c.get("reply", "")) for c in data.get("candidates", []))
            / max(len(data.get("candidates", [])), 1),
            1,
        ),
    }


def run_direct_test(effort: str, run_id: int) -> Dict[str, Any]:
    """直接 invoke（非 structured_output），便于拿到 token usage。"""
    llm = ChatOpenAI(
        model="gpt-5-mini",
        api_key=os.environ.get("OPENAI_API_KEY"),
        verbosity="low",
        reasoning_effort=effort,
    )

    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(
            content=USER_MSG
            + '\n\n请用JSON格式回复：{"candidates": [{"reply": "..."}, {"reply": "..."}, {"reply": "..."}]}'
        ),
    ]

    t0 = time.perf_counter()
    resp = llm.invoke(messages, max_tokens=800)
    elapsed = time.perf_counter() - t0

    content = resp.content or ""
    usage = {}
    if hasattr(resp, "response_metadata"):
        usage = resp.response_metadata.get("token_usage", {})

    return {
        "effort": effort,
        "run_id": run_id,
        "elapsed_s": round(elapsed, 3),
        "usage": usage,
        "content": content,
        "content_len": len(content),
    }


def print_separator(title: str):
    print(f"\n{'='*70}")
    print(f"  {title}")
    print(f"{'='*70}\n")


def main():
    num_runs = int(os.environ.get("TEST_RUNS", "3"))
    print(f"gpt-5-mini reasoning_effort 对比测试")
    print(f"每种 effort 跑 {num_runs} 次，取平均值\n")

    # ── Part 1: direct invoke ──
    print_separator("Part 1: Direct invoke (可获取完整 token usage)")

    results: Dict[str, list] = {"low": [], "medium": []}

    for effort in ["low", "medium"]:
        print(f"--- reasoning_effort={effort} ---")
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs} ...", end=" ", flush=True)
            r = run_direct_test(effort, i + 1)
            results[effort].append(r)
            u = r["usage"]
            print(
                f"耗时 {r['elapsed_s']}s | "
                f"prompt_tokens={u.get('prompt_tokens','?')} "
                f"completion_tokens={u.get('completion_tokens','?')} "
                f"reasoning_tokens={u.get('completion_tokens_details',{}).get('reasoning_tokens','?')} "
                f"total={u.get('total_tokens','?')}"
            )
        print()

    print_separator("Direct invoke 汇总对比")
    for effort in ["low", "medium"]:
        rs = results[effort]
        avg_time = sum(r["elapsed_s"] for r in rs) / len(rs)
        avg_prompt = sum(r["usage"].get("prompt_tokens", 0) for r in rs) / len(rs)
        avg_completion = sum(r["usage"].get("completion_tokens", 0) for r in rs) / len(rs)
        avg_reasoning = sum(
            (r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
            for r in rs
        ) / len(rs)
        avg_total = sum(r["usage"].get("total_tokens", 0) for r in rs) / len(rs)
        avg_content_len = sum(r["content_len"] for r in rs) / len(rs)

        print(f"  [{effort:6s}] 平均耗时: {avg_time:.2f}s")
        print(f"           prompt_tokens: {avg_prompt:.0f}")
        print(f"           completion_tokens: {avg_completion:.0f}")
        print(f"           reasoning_tokens: {avg_reasoning:.0f}")
        print(f"           total_tokens: {avg_total:.0f}")
        print(f"           回复字符数: {avg_content_len:.0f}")
        print()

    # ── Part 2: structured output ──
    print_separator("Part 2: Structured output (with_structured_output)")

    struct_results: Dict[str, list] = {"low": [], "medium": []}

    for effort in ["low", "medium"]:
        print(f"--- reasoning_effort={effort} ---")
        for i in range(num_runs):
            print(f"  Run {i+1}/{num_runs} ...", end=" ", flush=True)
            r = run_one_test(effort, i + 1)
            struct_results[effort].append(r)
            print(
                f"耗时 {r['elapsed_s']}s | "
                f"candidates={r['num_candidates']} | "
                f"avg_reply_len={r['avg_reply_len']}"
            )
            for j, c in enumerate(r["candidates"]):
                print(f"    [{j+1}] {c.get('reply','')}")
        print()

    print_separator("Structured output 汇总对比")
    for effort in ["low", "medium"]:
        rs = struct_results[effort]
        avg_time = sum(r["elapsed_s"] for r in rs) / len(rs)
        avg_candidates = sum(r["num_candidates"] for r in rs) / len(rs)
        avg_reply_len = sum(r["avg_reply_len"] for r in rs) / len(rs)

        print(f"  [{effort:6s}] 平均耗时: {avg_time:.2f}s")
        print(f"           候选数: {avg_candidates:.1f}")
        print(f"           平均回复长度: {avg_reply_len:.1f} 字")
        print()

    # ── 最终对比 ──
    print_separator("最终对比摘要")
    low_time = sum(r["elapsed_s"] for r in results["low"]) / len(results["low"])
    med_time = sum(r["elapsed_s"] for r in results["medium"]) / len(results["medium"])
    low_total = sum(r["usage"].get("total_tokens", 0) for r in results["low"]) / len(results["low"])
    med_total = sum(r["usage"].get("total_tokens", 0) for r in results["medium"]) / len(results["medium"])
    low_reasoning = sum(
        (r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        for r in results["low"]
    ) / len(results["low"])
    med_reasoning = sum(
        (r["usage"].get("completion_tokens_details") or {}).get("reasoning_tokens", 0)
        for r in results["medium"]
    ) / len(results["medium"])

    print(f"  耗时:     low={low_time:.2f}s  medium={med_time:.2f}s  差异={((med_time-low_time)/low_time)*100:+.1f}%")
    print(f"  总tokens: low={low_total:.0f}    medium={med_total:.0f}    差异={((med_total-low_total)/max(low_total,1))*100:+.1f}%")
    print(f"  推理tokens: low={low_reasoning:.0f}    medium={med_reasoning:.0f}    差异={((med_reasoning-low_reasoning)/max(low_reasoning,1))*100:+.1f}%")


if __name__ == "__main__":
    main()
