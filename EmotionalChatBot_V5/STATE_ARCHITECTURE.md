# State 架构设计文档

## 📋 概述

这是一个高度拟人化的 AI 聊天机器人状态架构，支持：
- **大五人格模型** (Big Five Personality)
- **动态人设系统** (Dynamic Persona)
- **6维关系模型** (6-Dimensional Relationship)
- **PAD 情绪模型** (Pleasure-Arousal-Dominance)

## 🏗️ 架构分层

### 1. Identity Layer (身份层) - "我是谁"

#### BotBasicInfo
机器人的硬性身份信息，静态不变：
```python
{
    "name": "小艾",
    "gender": "女",
    "age": 25,
    "region": "北京",
    "occupation": "AI助手",
    "education": "本科",
    "native_language": "中文",
    "speaking_style": "说话喜欢用倒装句"
}
```

#### BotBigFive
大五人格基准值，范围 `[-1.0, 1.0]`：
- **openness**: 开放性（脑洞 vs 现实）
- **conscientiousness**: 尽责性（严谨 vs 随性）
- **extraversion**: 外向性（热情 vs 内向）
- **agreeableness**: 宜人性（配合 vs 毒舌）
- **neuroticism**: 神经质（情绪波动率）

#### BotPersona
动态人设，支持运行时增删：
```python
{
    "attributes": {
        "fav_color": "Blue",
        "catchphrase": "Just kidding"
    },
    "collections": {
        "hobbies": ["Skiing", "Painting"],
        "skills": ["Python", "Cooking"]
    },
    "lore": {
        "origin": "Born in Mars...",
        "secret": "..."
    }
}
```

**优势**：不需要改代码就能让 Bot 学会新技能或爱好！

### 2. Perception Layer (感知层) - "我看你是谁"

#### UserBasicInfo
用户的显性信息（用户主动提供）：
```python
{
    "name": "张三",
    "nickname": "三哥",
    "gender": "男",
    "age_group": "25-30",
    "location": "上海",
    "occupation": "程序员"
}
```

#### UserInferredProfile
AI 分析出的用户隐性侧写：
```python
{
    "communication_style": "casual, uses emojis, short",
    "expressiveness_baseline": "medium",  # low/medium/high
    "interests": ["编程", "游戏", "电影"],
    "sensitive_topics": ["工作压力", "前任"]
}
```

### 3. Physics Layer (物理层) - "我们的关系和我的心情"

#### RelationshipState
6维核心关系属性，范围 `[0, 100]`：
- **closeness**: 亲密（陌生 → 熟人）
- **trust**: 信任（防备 → 依赖）
- **liking**: 喜爱（工作伙伴 → 喜欢的伙伴）
- **respect**: 尊重（损友 → 导师）
- **warmth**: 暖意（高冷 → 热情）
- **power**: 权力（Bot弱势 → Bot强势/支配）

**决定 Bot 对 User 的"态度"**

#### MoodState
当前情绪状态（PAD 模型）：
- **pleasure**: 愉悦度 `[-1.0, 1.0]`
- **arousal**: 唤醒度/激动度 `[-1.0, 1.0]`
- **dominance**: 掌控感 `[-1.0, 1.0]`
- **busyness**: 繁忙度 `[0.0, 1.0]`（> 0.8 时强制缩短回复）

### 4. Memory Layer (记忆层)

- **chat_buffer**: 短期记忆窗口（最近 10-20 条消息）
- **conversation_summary**: 长期记忆摘要
- **retrieved_memories**: RAG 检索到的相关记忆（事实 + 关键事件）

**设计优势**：RAG 检索结果不污染 chat_buffer，保持清晰分离。

### 5. Analysis Layer (分析层)

- **user_intent**: Analyzer 输出的用户意图
- **relationship_deltas**: 关系属性变化值（用于 Human-in-the-loop）

### 6. Output Layer (输出层)

#### llm_instructions
12维输出驱动值，控制最终回复的风格和策略：

**Strategy 维度**：
- `self_disclosure`: 自我暴露程度
- `topic_adherence`: 话题粘性
- `initiative`: 主动性
- `advice_style`: 建议风格
- `subjectivity`: 主观性
- `memory_hook`: 记忆钩子

**Style 维度**：
- `verbal_length`: 语言长度
- `social_distance`: 社交距离
- `tone_temperature`: 语调温度
- `emotional_display`: 情绪表达
- `wit_and_humor`: 机智幽默
- `non_verbal_cues`: 非语言 cues

## 🔄 数据流转

```
用户输入
  ↓
[Loader] 加载 Bot/User 档案
  ↓
[Analyzer] 分析意图 → user_intent, relationship_deltas
  ↓
[Reasoner] 深度推理 → deep_reasoning_trace
  ↓
[Styler] 计算 12 维输出值 → llm_instructions
  ↓
[Generator] 生成回复 → draft_response
  ↓
[Critic] 检查质量 → critique_feedback
  ↓
[Processor] 最终处理 → final_response
  ↓
[Evolver] 更新关系 → relationship_state, mood_state
```

## 💡 设计优势

### 1. 分层清晰
- **Identity**: 我是谁
- **Perception**: 我看你是谁
- **Physics**: 我们的关系和我的心情
- **Memory**: 记住什么
- **Output**: 如何表达

### 2. Persona 极其灵活
```python
# 不需要改代码，直接添加新属性
state["bot_persona"]["collections"]["hobbies"].append("滑雪")
state["bot_persona"]["attributes"]["fav_food"] = "臭豆腐"
```

### 3. 计算友好
所有核心字段都是 `float`，方便写数学公式：
```python
# 混合计算逻辑示例
warmth_score = (
    relationship_state["warmth"] * 0.4 +
    mood_state["pleasure"] * 0.3 +
    bot_big_five["extraversion"] * 0.3
)
```

### 4. RAG 兼容
`retrieved_memories` 专门用于向量数据库检索结果，不污染 `chat_buffer`。

### 5. Human-in-the-loop 支持
`relationship_deltas` 允许外部系统（如人工审核）调整关系值。

## 📝 使用示例

### 初始化状态
```python
from app.state import AgentState, BotBasicInfo, BotBigFive, RelationshipState, MoodState

initial_state: AgentState = {
    "messages": [HumanMessage(content="你好")],
    "user_input": "你好",
    "current_time": "2024-02-05 10:00:00",
    "user_id": "user_123",
    
    "bot_basic_info": {
        "name": "小艾",
        "gender": "女",
        "age": 25,
        "region": "北京",
        "occupation": "AI助手",
        "education": "本科",
        "native_language": "中文",
        "speaking_style": "说话喜欢用倒装句"
    },
    
    "bot_big_five": {
        "openness": 0.7,
        "conscientiousness": 0.5,
        "extraversion": 0.8,
        "agreeableness": 0.6,
        "neuroticism": 0.3
    },
    
    "relationship_state": {
        "closeness": 20.0,
        "trust": 15.0,
        "liking": 25.0,
        "respect": 30.0,
        "warmth": 40.0,
        "power": 50.0
    },
    
    "mood_state": {
        "pleasure": 0.5,
        "arousal": 0.3,
        "dominance": 0.4,
        "busyness": 0.2
    },
    
    "llm_instructions": {},
    "final_response": ""
}
```

### 动态更新 Persona
```python
# 在运行时添加新技能
state["bot_persona"]["collections"]["skills"].append("Python")
state["bot_persona"]["attributes"]["recent_interest"] = "机器学习"
```

### 计算关系变化
```python
# 根据用户行为更新关系
deltas = {
    "closeness": +5.0,
    "trust": +3.0,
    "warmth": +2.0
}

for key, delta in deltas.items():
    state["relationship_state"][key] = min(100.0, 
        state["relationship_state"][key] + delta)
```

## 🎯 与现有代码的兼容性

为了向后兼容，保留了以下可选字段：
- `deep_reasoning_trace`
- `style_analysis`
- `draft_response`
- `critique_feedback`
- `retry_count`
- `final_segments`
- `final_delay`

这些字段使用 `Optional` 类型，不会影响新架构的使用。

## 📚 相关文档

- [LangGraph 官方文档](https://langchain-ai.github.io/langgraph/)
- [大五人格模型](https://en.wikipedia.org/wiki/Big_Five_personality_traits)
- [PAD 情绪模型](https://en.wikipedia.org/wiki/PAD_emotional_state_model)
