"""
Knowledge Fetcher 节点：当 detection 判断 knowledge_gap=True 时触发，
执行外部搜索并将摘要写入 state.retrieved_external_knowledge。

搜索优先级：
  1. Tavily（设置了 TAVILY_API_KEY）：质量高，适合生产
  2. DuckDuckGo（无需 API key）：开发/备用

结果格式化后注入 inner_monologue 上下文（作为「你刚好知道的背景」）。
"""
from __future__ import annotations

import os
import re
import time
from datetime import datetime, timezone
from typing import Callable

from app.state import AgentState


_MAX_SNIPPET_CHARS = 200   # 每条搜索结果截断长度
_MAX_RESULTS = 3            # 最多保留几条结果

_TIME_QUALIFIERS = re.compile(r"(最新|现任|当前|今年|\d{4})")


def _ensure_time_qualifier(keywords: str) -> str:
    """如果搜索关键词不包含任何时间限定词，自动追加当前年份。"""
    if _TIME_QUALIFIERS.search(keywords):
        return keywords
    year = datetime.now(timezone.utc).year
    return f"{keywords} {year}最新"


def _search_duckduckgo(keywords: str) -> str:
    """使用 DuckDuckGo 搜索（优先 ddgs 包，否则 duckduckgo_search），返回拼接摘要。"""
    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            # 不使用 region 或使用 wt-wt，避免 cn-zh 经常无结果
            results = list(ddgs.text(keywords, max_results=_MAX_RESULTS))
        if not results:
            return ""
        lines = []
        for r in results:
            title = (r.get("title") or "").strip()
            body = (r.get("body") or "").strip()[:_MAX_SNIPPET_CHARS]
            if body:
                lines.append(f"- {title}：{body}" if title else f"- {body}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[KnowledgeFetcher] DuckDuckGo 搜索失败: {e}")
        return ""


def _search_tavily(keywords: str, api_key: str) -> str:
    """使用 Tavily 搜索，返回拼接摘要。"""
    try:
        from tavily import TavilyClient
        client = TavilyClient(api_key=api_key)
        resp = client.search(keywords, max_results=_MAX_RESULTS, search_depth="basic")
        results = resp.get("results") or []
        if not results:
            return ""
        lines = []
        for r in results:
            title = (r.get("title") or "").strip()
            content = (r.get("content") or "").strip()[:_MAX_SNIPPET_CHARS]
            if content:
                lines.append(f"- {title}：{content}" if title else f"- {content}")
        return "\n".join(lines)
    except Exception as e:
        print(f"[KnowledgeFetcher] Tavily 搜索失败: {e}")
        return ""


def create_knowledge_fetcher_node() -> Callable[[AgentState], dict]:
    """创建 Knowledge Fetcher 节点（无 LLM，纯外部搜索）。"""

    tavily_key = (os.getenv("TAVILY_API_KEY") or "").strip()

    def knowledge_fetcher_node(state: AgentState) -> dict:
        detection = state.get("detection") or {}
        knowledge_gap = detection.get("knowledge_gap") or state.get("knowledge_gap") or False
        search_keywords = (
            detection.get("search_keywords")
            or state.get("search_keywords")
            or ""
        ).strip()

        if not knowledge_gap or not search_keywords:
            return {"retrieved_external_knowledge": ""}

        search_keywords = _ensure_time_qualifier(search_keywords)

        t0_search = time.perf_counter()
        print(f"[KnowledgeFetcher] 触发搜索：{search_keywords!r}")

        # 有 TAVILY_API_KEY 时用 Tavily，否则用 DuckDuckGo
        snippet = ""
        if tavily_key:
            snippet = _search_tavily(search_keywords, tavily_key)
        if not snippet:
            snippet = _search_duckduckgo(search_keywords)

        elapsed_search = time.perf_counter() - t0_search
        if snippet:
            result = f"【外部搜索：{search_keywords}】\n{snippet}"
            print(f"[KnowledgeFetcher] 搜索成功，{len(snippet)} 字符，用时 {elapsed_search:.2f}s")
        else:
            result = ""
            print(f"[KnowledgeFetcher] 搜索无结果，用时 {elapsed_search:.2f}s")

        out = {
            "retrieved_external_knowledge": result,
            "knowledge_gap": knowledge_gap,
            "search_keywords": search_keywords,
        }
        # 供测试/监控：搜索步骤耗时（秒）
        if os.getenv("LTSR_SEARCH_TIMING") or os.getenv("BOT2BOT_FULL_LOGS"):
            out["retrieved_external_knowledge_elapsed_seconds"] = elapsed_search
        return out

    return knowledge_fetcher_node
