"""心理模式的数据定义。所有行为由 YAML 配置驱动，不写死在代码里。"""
from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class BehaviorContract(BaseModel):
    """行为层契约：允许/禁止做什么"""
    allowed_actions: List[str] = Field(default_factory=list)
    forbidden_actions: List[str] = Field(default_factory=list)
    notes: List[str] = Field(default_factory=list)


class LatsBudget(BaseModel):
    """LATS 搜索预算与消息结构硬约束"""
    enabled: bool = True
    max_iters: int = 2
    rollouts: int = 2
    expand_k: int = 2
    max_messages: int = 3
    min_first_len: int = 8
    max_message_len: int = 220


class RequirementsPolicy(BaseModel):
    """Hard gate / must-have 覆盖策略"""
    must_have_policy: str = Field(default="soft", description="soft=覆盖率影响分数, none=不要求覆盖")
    must_have_min_coverage: float = Field(default=0.75, description="最小覆盖率要求")
    allow_short_reply: bool = False
    allow_empty_reply: bool = False


class CriticCriteria(BaseModel):
    """评估重点：告诉 evaluator/critic 怎么判好坏"""
    quality_threshold: float = 0.70
    focus: List[str] = Field(default_factory=list)
    penalties: Dict[str, float] = Field(default_factory=dict)


class StyleBias(BaseModel):
    """给 Style 节点的偏置（只给"旋钮方向"，不写台词）"""
    verbal_length: Optional[float] = None
    tone_temperature: Optional[float] = None
    social_distance: Optional[float] = None
    advice_style: Optional[float] = None
    wit_and_humor: Optional[float] = None
    emotional_display: Optional[float] = None


class PsychoMode(BaseModel):
    """
    心理模式的数据定义。
    所有的行为逻辑都不写死在代码里，而是由这个对象控制。
    Mode 只控制"行为策略与预算"，不直接写语气台词；表达层仍由 Style 的 12 维旋钮决定。
    """

    id: str
    name: str
    description: str = Field(description="该模式的触发语义")

    behavior_contract: BehaviorContract = Field(default_factory=BehaviorContract)
    lats_budget: LatsBudget = Field(default_factory=LatsBudget)
    requirements_policy: RequirementsPolicy = Field(default_factory=RequirementsPolicy)
    critic_criteria: CriticCriteria = Field(default_factory=CriticCriteria)
    style_bias: StyleBias = Field(default_factory=StyleBias)
    disallowed: List[str] = Field(default_factory=list, description="该模式禁止的输出类型")

    # 向后兼容字段（可选）
    trigger_description: Optional[str] = None  # 旧字段，映射到 description
    system_prompt_template: Optional[str] = None  # 保留但不使用
    enable_deep_reasoning: bool = True  # 保留但不使用
    monologue_instruction: Optional[str] = None  # 保留但不使用
    split_strategy: str = Field(default="normal")  # 保留但不使用
    typing_speed_multiplier: float = Field(default=1.0)  # 保留但不使用

    def model_post_init(self, __context: Any) -> None:
        """后处理：将旧字段映射到新字段"""
        if self.trigger_description and not self.description:
            self.description = self.trigger_description
