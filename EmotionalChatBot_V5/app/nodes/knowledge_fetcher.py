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
from typing import Callable

from app.state import AgentState


_MAX_SNIPPET_CHARS = 200   # 每条搜索结果截断长度
_MAX_RESULTS = 3            # 最多保留几条结果


def _search_duckduckgo(keywords: str) -> str:
    """使用 duckduckgo-search 搜索，返回拼接摘要。"""
    try:
        from duckduckgo_search import DDGS
        with DDGS() as ddgs:
            results = list(ddgs.text(keywords, max_results=_MAX_RESULTS, region="cn-zh"))
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

        print(f"[KnowledgeFetcher] 触发搜索：{search_keywords!r}")

        # 优先 Tavily，否则 DuckDuckGo
        if tavily_key:
            snippet = _search_tavily(search_keywords, tavily_key)
        else:
            snippet = _search_duckduckgo(search_keywords)

        if snippet:
            result = f"【外部搜索：{search_keywords}】\n{snippet}"
            print(f"[KnowledgeFetcher] 搜索成功，{len(snippet)} 字符")
        else:
            result = ""
            print(f"[KnowledgeFetcher] 搜索无结果")

        return {
            "retrieved_external_knowledge": result,
            "knowledge_gap": knowledge_gap,
            "search_keywords": search_keywords,
        }

    return knowledge_fetcher_node
