"""LLM 客户端封装：支持普通调用与 Structured Output。"""
import os
from typing import Any, Optional, Type, TypeVar

from langchain_core.language_models import BaseChatModel
from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

T = TypeVar("T")


def get_llm(
    model: Optional[str] = None,
    api_key: Optional[str] = None,
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
                if "STATIC KNOWLEDGE (The Signal Rubric)" in s or "relationship across 6 dimensions" in s:
                    # Relationship Analyzer：返回可解析 JSON（含 thought_process / detected_signals / deltas）
                    content = (
                        '{"thought_process":"用户表达想聊聊，属于轻度自我暴露与求助信号。总体符合当前阶段。",'
                        '"detected_signals":["展示脆弱性 (在无助时求助)"],'
                        '"topic_category":"life_goals",'
                        '"self_disclosure_depth_level":2,'
                        '"is_intellectually_deep":true,'
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
