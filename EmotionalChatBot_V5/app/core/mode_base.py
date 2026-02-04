"""心理模式的数据定义。所有行为由 YAML 配置驱动，不写死在代码里。"""
from typing import List

from pydantic import BaseModel, Field


class PsychoMode(BaseModel):
    """
    心理模式的数据定义。
    所有的行为逻辑都不写死在代码里，而是由这个对象控制。
    """

    id: str
    name: str

    # --- 1. 给 LLM 侧写师看的 (用于检测) ---
    trigger_description: str = Field(
        description="一段自然语言描述，告诉侧写师在什么情况下应该进入这个模式"
    )

    # --- 2. 给 Generator 看的 (用于生成) ---
    system_prompt_template: str = Field(
        description="覆盖 System Prompt 的核心指令"
    )

    # --- 3. 给 DeepReasoner 看的 (用于思考) ---
    enable_deep_reasoning: bool = True
    monologue_instruction: str = Field(
        description="指导 Bot 在做决策时的思考方向，如'不要讲道理，只宣泄情绪'"
    )

    # --- 4. 给 Critic 看的 (用于质检) ---
    critic_criteria: List[str] = Field(
        description="质检列表，如['必须包含颤抖的语气', '不能超过10个字']"
    )

    # --- 5. 给 Processor 看的 (用于表现) ---
    split_strategy: str = Field(
        default="normal",
        description="fragmented(碎碎念) 或 normal(正常)",
    )
    typing_speed_multiplier: float = Field(
        default=1.0,
        description="1.0为正常速度，0.5为慢速犹豫",
    )
