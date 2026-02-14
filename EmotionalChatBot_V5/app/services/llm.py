"""LLM 客户端封装：支持普通调用与 Structured Output。"""
import os
import time
from typing import Any, Optional, Type, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

T = TypeVar("T")


def _truthy(v: Optional[str]) -> bool:
    if v is None:
        return False
    return v.strip().lower() in {"1", "true", "yes", "y", "on"}


def _approx_input_size(input: Any) -> dict[str, int]:
    """
    Best-effort sizing for perf logs (avoid heavy tokenization).
    Returns {"messages": n, "chars": c}
    """
    try:
        if isinstance(input, str):
            return {"messages": 1, "chars": len(input)}
        if isinstance(input, list):
            chars = 0
            for m in input:
                c = getattr(m, "content", None)
                if c is None:
                    c = str(m)
                chars += len(str(c))
            return {"messages": len(input), "chars": chars}
        c = getattr(input, "content", None)
        if c is None:
            c = str(input)
        return {"messages": 1, "chars": len(str(c))}
    except Exception:
        return {"messages": 0, "chars": 0}


class _TimedInvoker:
    """Wrap an object with .invoke(...) and log elapsed time."""

    def __init__(self, inner: Any, *, label: str):
        self._inner = inner
        self._label = label

    def invoke(self, input: Any, **kwargs) -> Any:
        enabled = _truthy(os.getenv("LTSR_LLM_PERF"))
        min_ms_raw = os.getenv("LTSR_LLM_PERF_MIN_MS", "").strip()
        try:
            min_ms = float(min_ms_raw) if min_ms_raw else 0.0
        except Exception:
            min_ms = 0.0

        size = _approx_input_size(input)
        t0 = time.perf_counter()
        try:
            return self._inner.invoke(input, **kwargs)
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            if enabled and dt_ms >= min_ms:
                model_name = getattr(self._inner, "model_name", None) or getattr(self._inner, "model", None) or "unknown"
                print(
                    f"[LLM_PERF] {self._label} model={model_name} "
                    f"dt_ms={dt_ms:.1f} messages={size['messages']} chars={size['chars']}"
                )


class TimedLLM:
    """
    Lightweight wrapper for perf logging.
    Keeps the small surface this repo uses:
    - invoke(...)
    - with_structured_output(...).invoke(...)
    """

    def __init__(self, inner: Any):
        self._inner = inner
        self.model_name = getattr(inner, "model_name", None) or getattr(inner, "model", None)

    def invoke(self, input: Any, **kwargs) -> Any:
        return _TimedInvoker(self._inner, label="invoke").invoke(input, **kwargs)

    def with_structured_output(self, schema: Type[T], **kwargs) -> Any:
        structured = self._inner.with_structured_output(schema, **kwargs)
        return _TimedInvoker(structured, label=f"structured<{getattr(schema, '__name__', 'schema')}>")


def _build_httpx_client_for_trace() -> Any:
    """
    Build an httpx.Client with request/response hooks that print:
    - status_code
    - request id (best-effort)
    - elapsed ms

    If the SDK retries internally, these hooks will run multiple times, which
    becomes direct evidence of retry/backoff.
    """
    try:
        import httpx  # type: ignore
    except Exception:
        return None

    enabled = _truthy(os.getenv("LTSR_HTTP_TRACE"))
    if not enabled:
        return None

    min_ms_raw = os.getenv("LTSR_HTTP_TRACE_MIN_MS", "").strip()
    try:
        min_ms = float(min_ms_raw) if min_ms_raw else 0.0
    except Exception:
        min_ms = 0.0

    def on_request(request: "httpx.Request") -> None:
        request.extensions["ltsr_t0"] = time.perf_counter()

    def on_response(response: "httpx.Response") -> None:
        t0 = response.request.extensions.get("ltsr_t0")
        if not isinstance(t0, (int, float)):
            return
        dt_ms = (time.perf_counter() - t0) * 1000.0
        if dt_ms < min_ms:
            return

        h = response.headers
        request_id = (
            h.get("x-request-id")
            or h.get("request-id")
            or h.get("x-dashscope-request-id")
            or h.get("x-amzn-requestid")
            or ""
        )
        method = response.request.method
        url = str(response.request.url)
        # Avoid printing any auth headers (never print request headers).
        print(f"[HTTP_TRACE] {method} {url} status={response.status_code} dt_ms={dt_ms:.1f} request_id={request_id}")

    return httpx.Client(event_hooks={"request": [on_request], "response": [on_response]})


def get_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.3,
) -> BaseChatModel:
    """获取配置好的 LLM 实例。未配置 API Key 时返回 MockLLM。"""
    key = api_key or os.getenv("OPENAI_API_KEY")
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o")
    base_url = os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE")
    if key:
        kwargs: dict[str, Any] = {
            "model": model_name,
            "temperature": temperature,
            "api_key": key,
        }
        # Best-effort timeout/retries (provider-dependent). Helps identify "卡死" vs "慢"。
        timeout_s_raw = os.getenv("OPENAI_TIMEOUT_SECONDS", "").strip()
        retries_raw = os.getenv("OPENAI_MAX_RETRIES", "").strip()
        try:
            timeout_s = float(timeout_s_raw) if timeout_s_raw else None
        except Exception:
            timeout_s = None
        try:
            max_retries = int(retries_raw) if retries_raw else None
        except Exception:
            max_retries = None
        if timeout_s is not None:
            kwargs["timeout"] = timeout_s
        if max_retries is not None:
            kwargs["max_retries"] = max_retries
        # Allow OpenAI-compatible providers (e.g., DashScope) via base_url.
        # If the underlying client doesn't accept base_url, fall back silently.
        if base_url:
            kwargs["base_url"] = base_url
        # Optional: enable HTTP-level trace (status_code, request_id, elapsed).
        http_client = _build_httpx_client_for_trace()
        if http_client is not None:
            kwargs["http_client"] = http_client
            # Let langchain-openai include response headers in message metadata (best-effort).
            kwargs["include_response_headers"] = True
        try:
            llm = ChatOpenAI(**kwargs)  # type: ignore[arg-type]
        except TypeError:
            kwargs.pop("base_url", None)
            # Some versions may not accept timeout/max_retries either.
            kwargs.pop("timeout", None)
            kwargs.pop("max_retries", None)
            kwargs.pop("http_client", None)
            kwargs.pop("include_response_headers", None)
            llm = ChatOpenAI(**kwargs)  # type: ignore[arg-type]
        return TimedLLM(llm) if _truthy(os.getenv("LTSR_LLM_PERF")) else llm
    llm = MockLLM(model=model_name, temperature=temperature)
    return TimedLLM(llm) if _truthy(os.getenv("LTSR_LLM_PERF")) else llm


class MockLLM(BaseChatModel):
    """无 API 时的占位 LLM：返回固定/简单逻辑，便于本地跑通流程。"""

    model: str = "mock"
    temperature: float = 0.3

    @property
    def _llm_type(self) -> str:
        return "mock"

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        from langchain_core.outputs import ChatGeneration, ChatResult
        from langchain_core.callbacks.manager import CallbackManagerForLLMRun

        content = "[Mock] 已根据输入生成占位回复。请配置 OPENAI_API_KEY 使用真实模型。"
        for m in messages:
            if hasattr(m, "content") and m.content:
                s = str(m.content)
                if "STATIC KNOWLEDGE (The Signal Rubric)" in s or "relationship across 6 dimensions" in s:
                    # Relationship Analyzer：返回可解析 JSON（含 thought_process / detected_signals / deltas）
                    content = (
                        '{"thought_process":"用户表达想聊聊，属于轻度自我暴露与求助信号。总体符合当前阶段。",'
                        '"detected_signals":["展示脆弱性 (在无助时求助)"],'
                        '"deltas":{"closeness":1,"trust":1,"liking":1,"respect":0,"warmth":1,"power":0}}'
                    )
                    break
                if "侧写" in s:
                    content = '{"reasoning":"用户语气正常","target_mode_id":"normal_mode"}'
                    break
                if "read the room" in s or "Intuition & Social Radar" in s or "intuition_thought" in s:
                    # 感知节点：给一个可解析的 JSON
                    content = (
                        '{"intuition_thought":"语境整体正常，用户只是表达想聊聊。","category":"NORMAL","reason":"与摘要情绪一致，未见越界或胡言乱语","risk_score":1}'
                    )
                    break
                if "去死" in s or "滚" in s:
                    content = '{"reasoning":"检测到攻击","target_mode_id":"stress_mode"}'
                    break
                if "质检" in s or "标准" in s:
                    content = "通过"
                    break
                if "内心独白" in s or "思考方向" in s:
                    content = "用户需要情绪支持，应共情并简短回应。"
                    break
                if "风格" in s and "模式" in s:
                    content = "自然、友好、适度共情。"
                    break
        gen = ChatGeneration(message=HumanMessage(content=content))
        return ChatResult(generations=[gen])

    def invoke(self, input: Any, **kwargs) -> Any:
        if isinstance(input, str):
            messages = [HumanMessage(content=input)]
        else:
            messages = input if isinstance(input, list) else [input]
        result = self._generate(messages, **kwargs)
        return result.generations[0].message

    def with_structured_output(self, schema: Type[T], **kwargs) -> Any:
        """返回一个可调用的对象，invoke(prompt) -> schema 实例。"""
        return _StructuredMock(self, schema)


class _StructuredMock:
    """Mock 的 with_structured_output 返回的调用器。"""

    def __init__(self, llm: MockLLM, schema: Type[T]):
        self.llm = llm
        self.schema = schema

    def invoke(self, prompt: Any) -> T:
        # 支持传入 messages(list[BaseMessage]) 或字符串
        msg = self.llm.invoke(prompt)
        content = getattr(msg, "content", str(msg))
        # 尝试从 content 里解析出 target_mode_id / reasoning
        import json
        try:
            # 可能是 JSON 字符串
            d = json.loads(content.strip())
        except Exception:
            d = {"reasoning": content[:100], "target_mode_id": "normal_mode"}
        return self.schema(**d)
