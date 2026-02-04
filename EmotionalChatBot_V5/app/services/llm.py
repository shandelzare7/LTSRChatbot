"""LLM 客户端封装：支持普通调用与 Structured Output。"""
import os
from typing import Any, Type, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

T = TypeVar("T")


def get_llm(
    model: str | None = None,
    api_key: str | None = None,
    temperature: float = 0.3,
) -> BaseChatModel:
    """获取配置好的 LLM 实例。未配置 API Key 时返回 MockLLM。"""
    key = api_key or os.getenv("OPENAI_API_KEY")
    model_name = model or os.getenv("OPENAI_MODEL", "gpt-4o")
    if key:
        return ChatOpenAI(
            model=model_name,
            temperature=temperature,
            api_key=key,
        )
    return MockLLM(model=model_name, temperature=temperature)


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
                if "侧写" in s:
                    content = '{"reasoning":"用户语气正常","target_mode_id":"normal_mode"}'
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

    def invoke(self, prompt: str) -> T:
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
