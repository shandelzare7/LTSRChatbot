"""心理引擎：根据用户输入与上下文，通过 LLM 侧写检测当前应进入的心理模式。"""
from typing import Any, Dict, List

from pydantic import BaseModel, Field

from app.core.mode_base import PsychoMode


class PsychoAssessment(BaseModel):
    """LLM 侧写输出结构"""

    reasoning: str = Field(description="简短分析理由")
    target_mode_id: str = Field(description="目标模式 id，如 normal_mode, stress_mode")


class PsychoEngine:
    """大脑：加载所有模式，用 LLM 判断当前应进入哪种模式。"""

    def __init__(self, modes: List[PsychoMode], llm_invoker: Any):
        """
        :param modes: 所有可用的心理模式列表
        :param llm_invoker: 具备 with_structured_output(PsychoAssessment).invoke(prompt) 的 LLM 封装
        """
        self.modes = modes
        self._mode_by_id = {m.id: m for m in modes}
        self.llm = llm_invoker

    def get_mode_obj(self, mode_id: str) -> PsychoMode:
        """根据 id 返回模式对象，找不到则返回第一个（默认）模式"""
        return self._mode_by_id.get(mode_id, self.modes[0])

    def detect_mode(self, user_msg: str, context_data: Dict[str, Any]) -> PsychoMode:
        """
        根据用户最新消息与上下文，调用 LLM 判断应进入的模式，返回对应的 PsychoMode。
        """
        options = "\n".join(
            [f"- {m.id}: {m.trigger_description}" for m in self.modes]
        )
        current = context_data.get("current_mode_id", "normal_mode")

        prompt = f"""你是心理侧写师。请分析用户当前言论，判断 Bot 应该进入哪种心理状态。
当前状态: {current}

可选模式:
{options}

用户最新消息:
{user_msg}

请只输出上述模式 id 之一（如 normal_mode）。若用户输入正常、无攻击/越界/崩溃迹象，请选择 normal_mode。"""

        try:
            result = self.llm.with_structured_output(PsychoAssessment).invoke(
                prompt
            )
            if isinstance(result, PsychoAssessment):
                return self.get_mode_obj(result.target_mode_id)
            if isinstance(result, dict):
                return self.get_mode_obj(result.get("target_mode_id", "normal_mode"))
        except Exception as e:
            # 降级：保持当前或回退 normal
            print(f"[PsychoEngine] LLM 检测异常，使用默认模式: {e}")
        return self.get_mode_obj("normal_mode")
