"""LLM 客户端封装：支持普通调用与 Structured Output，并支持按角色路由多模型。

Profiling notes:
- When enabled (LTSR_LLM_STATS=1 or LTSR_PROFILE_STEPS=1), we collect per-(role,base_url,model)
  call counts and total elapsed time. This is used by devtools/bot_to_bot_chat.py to produce
  step-by-step performance reports.
"""
import os
import time
import contextvars
from typing import Any, Optional, Type, TypeVar, Literal
from datetime import datetime

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

T = TypeVar("T")

# ----------------------------
# Current node context (optional)
# ----------------------------

_CURRENT_NODE: contextvars.ContextVar[str] = contextvars.ContextVar("ltsr_current_node", default="")


def set_current_node(name: str) -> contextvars.Token:
    """Set current graph node name (for per-call logs)."""
    return _CURRENT_NODE.set(str(name or ""))


def reset_current_node(token: contextvars.Token) -> None:
    """Reset current node name back to previous value."""
    try:
        _CURRENT_NODE.reset(token)
    except Exception:
        pass


def get_current_node() -> str:
    try:
        return str(_CURRENT_NODE.get() or "")
    except Exception:
        return ""


def _call_log_enabled() -> bool:
    return _truthy(os.getenv("LTSR_LLM_CALL_LOG"))


def _dump_file_path() -> str:
    """If set, append every LLM call's prompt+timing to this file."""
    return (os.getenv("LTSR_LLM_DUMP_FILE") or "").strip()


def _service_tier_for_call(*, model: str, base_url: str) -> Optional[str]:
    """
    Best-effort OpenAI service tier routing.
    Default behavior in this repo: all `gpt-4o-mini` calls use `priority` on the official
    OpenAI endpoint (unless explicitly disabled).
    """
    m = str(model or "")
    if m != "gpt-4o-mini":
        return None
    # Only for OpenAI official endpoint (or empty base_url which defaults to it in this repo)
    b = (base_url or "").strip()
    if b and "api.openai.com" not in b:
        return None
    # Explicit disable switch (debug/perf comparisons)
    disable = (os.getenv("LTSR_DISABLE_4OMINI_PRIORITY") or "").strip().lower() in {
        "1",
        "true",
        "yes",
        "y",
        "on",
    }
    if disable:
        return None
    # Allow override via env; default to priority for gpt-4o-mini
    tier = (os.getenv("LTSR_OPENAI_SERVICE_TIER") or os.getenv("OPENAI_SERVICE_TIER") or "").strip()
    return tier or "priority"


_LLM_DUMP_SEQ = 0


def _as_text(x: Any) -> str:
    if x is None:
        return ""
    if isinstance(x, str):
        return x
    try:
        return str(x)
    except Exception:
        return repr(x)


def _format_messages_for_dump(input: Any) -> str:
    """Best-effort: turn input/messages into readable text."""
    if isinstance(input, list):
        lines: list[str] = []
        for i, m in enumerate(input):
            role = getattr(m, "type", None) or getattr(m, "role", None) or m.__class__.__name__
            content = _as_text(getattr(m, "content", None) if hasattr(m, "content") else m)
            lines.append(f"- [{i}] {role}: {content}")
        return "\n".join(lines)
    # single message or string
    role = getattr(input, "type", None) or getattr(input, "role", None) or input.__class__.__name__
    content = _as_text(getattr(input, "content", None) if hasattr(input, "content") else input)
    return f"- [0] {role}: {content}"


def _dump_llm_call(*, node: str, role: str, kind: str, model: str, dt_ms: float, input: Any) -> None:
    path = _dump_file_path()
    if not path:
        return
    global _LLM_DUMP_SEQ
    _LLM_DUMP_SEQ += 1
    try:
        ts = datetime.now().isoformat(timespec="seconds")
    except Exception:
        ts = ""
    block = "\n".join(
        [
            "============================================================",
            f"seq={_LLM_DUMP_SEQ}",
            f"time={ts}",
            f"node={node}",
            f"role={role}",
            f"kind={kind}",
            f"model={model}",
            f"dt_ms={dt_ms:.1f}",
            "messages:",
            _format_messages_for_dump(input),
            "",
        ]
    )
    try:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "a", encoding="utf-8") as f:
            f.write(block)
    except Exception:
        # Never fail the main call due to dump issues.
        return


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


# ----------------------------
# LLM call stats (optional)
# ----------------------------

_LLM_STATS: dict[str, dict[str, Any]] = {}


def _stats_enabled() -> bool:
    return _truthy(os.getenv("LTSR_LLM_STATS")) or _truthy(os.getenv("LTSR_PROFILE_STEPS"))


def reset_llm_stats() -> None:
    """Reset in-memory counters for a fresh run."""
    global _LLM_STATS
    _LLM_STATS = {}


def get_llm_stats() -> dict[str, dict[str, Any]]:
    """Return a shallow copy of the current stats dict."""
    return {k: dict(v) for k, v in _LLM_STATS.items()}


def llm_stats_snapshot() -> dict[str, dict[str, Any]]:
    """Alias for get_llm_stats(), kept for clarity at call sites."""
    return get_llm_stats()


def llm_stats_diff(before: dict[str, dict[str, Any]], after: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    """Compute (after - before) for each key."""
    out: dict[str, dict[str, Any]] = {}
    keys = set(before.keys()) | set(after.keys())
    for k in keys:
        b = before.get(k) or {}
        a = after.get(k) or {}
        dcalls = int(a.get("calls", 0) or 0) - int(b.get("calls", 0) or 0)
        dms = float(a.get("total_ms", 0.0) or 0.0) - float(b.get("total_ms", 0.0) or 0.0)
        if dcalls or abs(dms) > 0.001:
            out[k] = {"calls": dcalls, "total_ms": round(dms, 2)}
    return out


def _stats_key(*, role: str, base_url: str, model: str, kind: str) -> str:
    return f"role={role}|kind={kind}|base_url={base_url}|model={model}"


def _record_llm_call(*, role: str, base_url: str, model: str, kind: str, dt_ms: float) -> None:
    if not _stats_enabled():
        return
    k = _stats_key(role=role, base_url=base_url or "", model=model or "unknown", kind=kind)
    rec = _LLM_STATS.get(k)
    if rec is None:
        rec = {"calls": 0, "total_ms": 0.0}
        _LLM_STATS[k] = rec
    rec["calls"] = int(rec.get("calls", 0) or 0) + 1
    rec["total_ms"] = float(rec.get("total_ms", 0.0) or 0.0) + float(dt_ms or 0.0)


class InstrumentedLLM:
    """Wrapper that records call counts and elapsed time."""

    def __init__(self, inner: Any, *, role: str, base_url: str, model: str):
        self._inner = inner
        self._role = role
        self._base_url = base_url or ""
        self.model_name = getattr(inner, "model_name", None) or getattr(inner, "model", None) or model
        self._model = model or self.model_name or "unknown"

    def invoke(self, input: Any, **kwargs) -> Any:
        t0 = time.perf_counter()
        try:
            tier = _service_tier_for_call(model=str(self._model), base_url=str(self._base_url))
            if tier and "service_tier" not in kwargs:
                kwargs["service_tier"] = tier
            return self._inner.invoke(input, **kwargs)
        finally:
            dt_ms = (time.perf_counter() - t0) * 1000.0
            _record_llm_call(role=self._role, base_url=self._base_url, model=str(self._model), kind="invoke", dt_ms=dt_ms)
            node = get_current_node() or "?"
            _dump_llm_call(node=node, role=self._role, kind="invoke", model=str(self._model), dt_ms=dt_ms, input=input)
            if _call_log_enabled():
                print(
                    f"[LLM_CALL] node={node} role={self._role} kind=invoke "
                    f"model={self._model} dt_ms={dt_ms:.1f}"
                )

    def with_structured_output(self, schema: Type[T], **kwargs) -> Any:
        structured = self._inner.with_structured_output(schema, **kwargs)

        role = self._role
        base_url = self._base_url
        model = str(self._model)

        class _StructuredInstrumented:
            def __init__(self, inner_struct: Any):
                self._inner_struct = inner_struct

            def invoke(self, input: Any, **kw) -> Any:
                t0 = time.perf_counter()
                try:
                    tier = _service_tier_for_call(model=str(model), base_url=str(base_url))
                    if tier and "service_tier" not in kw:
                        kw["service_tier"] = tier
                    return self._inner_struct.invoke(input, **kw)
                finally:
                    dt_ms = (time.perf_counter() - t0) * 1000.0
                    _record_llm_call(
                        role=role,
                        base_url=base_url,
                        model=model,
                        kind=f"structured<{getattr(schema, '__name__', 'schema')}>",
                        dt_ms=dt_ms,
                    )
                    node = get_current_node() or "?"
                    _dump_llm_call(
                        node=node,
                        role=role,
                        kind=f"structured<{getattr(schema, '__name__', 'schema')}>",
                        model=str(model),
                        dt_ms=dt_ms,
                        input=input,
                    )
                    if _call_log_enabled():
                        print(
                            f"[LLM_CALL] node={node} role={role} kind=structured<{getattr(schema, '__name__', 'schema')}> "
                            f"model={model} dt_ms={dt_ms:.1f}"
                        )

        return _StructuredInstrumented(structured)


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
    role: Optional[str] = None,
    model: Optional[str] = None,
    api_key: Optional[str] = None,
    temperature: float = 0.3,
) -> BaseChatModel:
    """
    获取配置好的 LLM 实例。未配置 API Key 时返回 MockLLM。

    支持按角色路由多模型（role: main/fast/judge），并支持预设：
    - LTSR_LLM_PRESET=openai
    - LTSR_LLM_PRESET=deepseek_route_a  (main=DeepSeek, fast/judge=OpenAI)
    - LTSR_LLM_PRESET=deepseek_route_b  (all=DeepSeek)

    角色级覆盖（优先级最高）：
    - LTSR_LLM_<ROLE>_API_KEY
    - LTSR_LLM_<ROLE>_BASE_URL
    - LTSR_LLM_<ROLE>_MODEL
    - LTSR_LLM_<ROLE>_TEMPERATURE
    """

    r = (role or "main").strip().lower()

    def _env(role_key: str, name: str) -> str:
        return os.getenv(f"LTSR_LLM_{role_key.upper()}_{name}", "").strip()

    # Role-level overrides (highest priority)
    role_api_key = _env(r, "API_KEY")
    role_base_url = _env(r, "BASE_URL")
    role_model = _env(r, "MODEL")
    role_temp = _env(r, "TEMPERATURE")

    preset = (os.getenv("LTSR_LLM_PRESET") or "").strip().lower() or "openai"

    # Resolve config by preset (when role overrides absent)
    def _resolve_by_preset() -> tuple[Optional[str], Optional[str], Optional[str], Optional[float]]:
        # Defaults
        if preset == "deepseek_route_a":
            if r == "main":
                k = (os.getenv("DEEPSEEK_API_KEY") or "").strip() or (os.getenv("OPENAI_API_KEY") or "").strip()
                return k or None, "https://api.deepseek.com/v1", "deepseek-chat", None
            # fast/judge use OpenAI official by default (can override via LTSR_LLM_* envs)
            k = (os.getenv("OPENAI_API_KEY_OPENAI") or "").strip() or (os.getenv("OPENAI_API_KEY") or "").strip()
            if r == "fast":
                return k or None, "https://api.openai.com/v1", "gpt-4o", None
            if r == "judge":
                # Judge needs structured JSON stability; default to gpt-4o (override to mini if cost-sensitive)
                return k or None, "https://api.openai.com/v1", "gpt-4o", None
            return k or None, "https://api.openai.com/v1", "gpt-4o", None

        if preset == "deepseek_route_b":
            # All roles use DeepSeek
            k = (os.getenv("DEEPSEEK_API_KEY") or "").strip() or (os.getenv("OPENAI_API_KEY") or "").strip()
            return k or None, "https://api.deepseek.com/v1", "deepseek-chat", None

        # openai (default)
        k = (os.getenv("OPENAI_API_KEY_OPENAI") or "").strip() or (os.getenv("OPENAI_API_KEY") or "").strip()
        if r == "fast":
            return k or None, "https://api.openai.com/v1", "gpt-4o", None
        if r == "judge":
            return k or None, "https://api.openai.com/v1", "gpt-4o", None
        return k or None, "https://api.openai.com/v1", "gpt-4o", None

    preset_key, preset_base_url, preset_model, preset_temp = _resolve_by_preset()

    key = api_key or role_api_key or preset_key or (os.getenv("OPENAI_API_KEY") or "")
    model_name = model or role_model or preset_model or os.getenv("OPENAI_MODEL", "gpt-4o")
    base_url = role_base_url or preset_base_url or os.getenv("OPENAI_BASE_URL") or os.getenv("OPENAI_API_BASE") or os.getenv("OPENAI_BASE")
    if role_temp:
        try:
            temperature = float(role_temp)
        except Exception:
            pass
    elif preset_temp is not None:
        temperature = float(preset_temp)

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
        wrapped: Any = llm
        if _stats_enabled():
            wrapped = InstrumentedLLM(wrapped, role=r, base_url=base_url or "", model=model_name)
        if _truthy(os.getenv("LTSR_LLM_PERF")):
            wrapped = TimedLLM(wrapped)
        return wrapped
    llm = MockLLM(model=model_name, temperature=temperature)
    wrapped2: Any = llm
    if _stats_enabled():
        wrapped2 = InstrumentedLLM(wrapped2, role=r, base_url=base_url or "", model=model_name)
    if _truthy(os.getenv("LTSR_LLM_PERF")):
        wrapped2 = TimedLLM(wrapped2)
    return wrapped2


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
