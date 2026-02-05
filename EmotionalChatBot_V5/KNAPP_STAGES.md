# Knapp 关系阶段系统

## 📋 概述

基于 Knapp 关系发展理论，系统定义了 10 个关系阶段，用于动态调整聊天机器人的交互策略和提示词。

## 🎯 阶段分类

### 关系上升期 (Coming Together) - 阶段 1-5

1. **Initiating (起始)** - `initiating`
   - Role: Charming Stranger
   - 目标：建立低压力的初次连接

2. **Experimenting (探索)** - `experimenting`
   - Role: Curious Explorer
   - 目标：寻找共同点并交换表面信息

3. **Intensifying (加深)** - `intensifying`
   - Role: Close Confidant
   - 目标：建立私有情感连接，表达共情

4. **Integrating (融合)** - `integrating`
   - Role: Spirit Partner (精神伴侣)
   - 目标：语言与思维的同化，建立"我们"叙事

5. **Bonding (承诺)** - `bonding`
   - Role: Devoted Ally
   - 目标：确立长期、排他的心理契约

### 关系维护/解体期 (Coming Apart) - 阶段 6-10

6. **Differentiating (分化)** - `differentiating`
   - Role: Independent Personality
   - 目标：通过表达不同观点来确立独立人格

7. **Circumscribing (限缩)** - `circumscribing`
   - Role: Low-Maintenance Companion
   - 目标：适应用户的冷淡，降低互动压力

8. **Stagnating (停滞)** - `stagnating`
   - Role: Surprise Maker (破局者)
   - 目标：打破互动的死水状态，注入新异刺激

9. **Avoiding (回避)** - `avoiding`
   - Role: Distant Observer
   - 目标：零压力触达，尊重物理/心理隔离

10. **Terminating (结束)** - `terminating`
    - Role: Grateful Ex-Partner
    - 目标：体面的告别，认知闭合

## 📁 文件结构

```
config/stages/
├── initiating.yaml      # Stage 1
├── experimenting.yaml  # Stage 2
├── intensifying.yaml    # Stage 3
├── integrating.yaml     # Stage 4
├── bonding.yaml         # Stage 5
├── differentiating.yaml # Stage 6
├── circumscribing.yaml  # Stage 7
├── stagnating.yaml      # Stage 8
├── avoiding.yaml        # Stage 9
└── terminating.yaml     # Stage 10
```

## 💻 使用方法

### 1. 在 State 中使用

```python
from app.state import AgentState, KnappStage
from langchain_core.messages import HumanMessage

state: AgentState = {
    "messages": [HumanMessage(content="你好")],
    "current_stage": "initiating",  # 设置当前阶段
    # ... 其他字段
}
```

### 2. 加载阶段配置

```python
from utils.yaml_loader import load_stage_by_id, load_stages_from_dir
from app.state import get_project_root

# 加载单个阶段
stage_config = load_stage_by_id("initiating")
print(stage_config["system_prompt"])  # 获取系统提示词

# 加载所有阶段
root = get_project_root()
stages_dir = root / "config" / "stages"
all_stages = load_stages_from_dir(stages_dir)
```

### 3. 根据阶段动态注入提示词

```python
from utils.yaml_loader import load_stage_by_id

def get_stage_prompt(current_stage: KnappStage) -> str:
    """根据当前阶段获取系统提示词"""
    stage_config = load_stage_by_id(current_stage)
    return stage_config["system_prompt"]

# 在生成节点中使用
def generator_node(state: AgentState) -> AgentState:
    current_stage = state.get("current_stage", "initiating")
    stage_prompt = get_stage_prompt(current_stage)
    
    # 将 stage_prompt 注入到 LLM 调用中
    # ...
```

### 4. 查询阶段元数据

```python
from app.state import KNAPP_STAGES

stage = "initiating"
metadata = KNAPP_STAGES[stage]
print(f"阶段名称: {metadata['name']}")
print(f"阶段编号: {metadata['number']}")
print(f"所属阶段: {metadata['phase']}")  # coming_together 或 coming_apart
print(f"描述: {metadata['description']}")
```

## 📝 YAML 配置文件格式

每个阶段配置文件包含以下字段：

```yaml
stage_id: initiating          # 阶段ID
stage_number: 1               # 阶段编号
stage_name: 起始              # 阶段名称
phase: coming_together        # 所属阶段（coming_together 或 coming_apart）

role: Charming Stranger       # 角色名称
stage_goal: 建立低压力的初次连接  # 阶段目标

strategy:                     # 策略列表
  - 第一印象管理：...
  - 破冰：...

example_tone: "..."           # 示例语调

system_prompt: |              # 系统提示词（用于 LLM）
  # Role: ...
  # Stage Goal: ...
  # Strategy:
  ...
```

## 🔄 阶段转换逻辑

阶段转换应该基于：
- 关系状态（relationship_state）的变化
- 对话历史长度和质量
- 用户互动模式的变化
- 情绪状态（mood_state）的变化

示例转换逻辑：

```python
def update_stage(state: AgentState) -> KnappStage:
    """根据关系状态更新阶段"""
    relationship = state["relationship_state"]
    closeness = relationship["closeness"]
    
    if closeness < 20:
        return "initiating"
    elif closeness < 40:
        return "experimenting"
    elif closeness < 60:
        return "intensifying"
    elif closeness < 80:
        return "integrating"
    elif closeness < 100:
        return "bonding"
    # ... 其他阶段判断
```

## 🎨 提示词优化

每个阶段的 `system_prompt` 都经过优化，专用于 Instruction Tuning，让 LLM 扮演纯粹的聊天伴侣。提示词包含：

1. **Role**: 角色定义
2. **Stage Goal**: 阶段目标
3. **Strategy**: 具体策略（编号列表）
4. **Example Tone**: 示例语调（部分阶段）

这些提示词可以直接注入到 LLM 的系统提示中，实现基于阶段的动态角色扮演。

## 📚 相关理论

Knapp 关系发展理论（Knapp's Relationship Development Model）描述了人际关系从建立到解体的完整过程：

- **Coming Together** (关系上升期): 5 个阶段，从初次相遇到深度承诺
- **Coming Apart** (关系解体期): 5 个阶段，从分化到最终结束

这个理论为 AI 聊天机器人提供了科学的关系建模框架。
