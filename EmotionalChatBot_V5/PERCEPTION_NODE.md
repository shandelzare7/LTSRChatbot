# 感知与直觉节点 (Perception & Intuition Node)

## 🎯 设计理念

Detection 节点已升级为**感知与直觉节点**，不再是简单的分类器，而是包含**直觉思考过程**的智能感知系统。

### 核心洞察

判定一个人是否"发癫"或"KY"，本身就是一种深度的意图理解。如果没有"他是不是喝醉了？"这样的思考过程，生硬的分类器容易误判。

### 两阶段架构

```
阶段一（Perception）：直觉感知
  ↓
  快速判断："这人是不是喝高了？" "他想干嘛？这正常吗？"
  ↓
  决定路线：NORMAL / ABNORMAL

阶段二：
  ├─ NORMAL → Deep Mind Node（深层情感与策略计算）
  └─ ABNORMAL → Reflex Nodes（本能反应：防御/敷衍/困惑）
```

## 🔄 工作流程

### 1. 直觉思考（Intuitive Analysis）

在分类之前，LLM 必须进行**直觉思考**：

```
内部独白示例：
- "Bot正在聊家常，用户突然说自己是秦始皇。这完全不符合逻辑，
  且没有幽默的前置铺垫。看起来像是用户在胡言乱语或者测试AI。"
  
- "他是在浪漫话题中突然要钱。他看起来要么喝醉了，要么在诈骗。
  这破坏了氛围。"
```

### 2. 基于思考的分类

思考完成后，基于直觉分析进行分类：

```json
{
  "intuition_thought": "Bot正在聊家常，用户突然说自己是秦始皇...",
  "category": "CRAZY",
  "reason": "完全不符合逻辑，且没有幽默的前置铺垫",
  "risk_score": 8
}
```

## 📝 提示词结构

### 关键部分

1. **Role**: Intuition & Social Radar（直觉与社交雷达）
2. **Task**: 
   - Step 1: Intuitive Analysis（直觉分析）- **必须先执行**
   - Step 2: Classification（分类）
3. **Output**: JSON 格式，**必须包含** `intuition_thought` 字段

### 思考引导

提示词明确要求 LLM 思考：
- "这人是不是喝高了？"
- "他是不是在测试我？"
- "这正常吗？"
- "他想干嘛？"

## 💡 为什么这样设计？

### A. 满足"先判断是不是疯了"

当 LLM 必须先填充 `intuition_thought` 字段时，它被迫先进行推理，而不是直接分类。

**案例**：用户突然说"我是秦始皇"

**没有直觉思考**：
```json
{
  "category": "CRAZY"  // 可能误判
}
```

**有直觉思考**：
```json
{
  "intuition_thought": "Bot正在聊家常，用户突然说自己是秦始皇。这完全不符合逻辑，且没有幽默的前置铺垫。看起来像是用户在胡言乱语或者测试AI。",
  "category": "CRAZY",
  "reason": "完全不符合逻辑",
  "risk_score": 8
}
```

有了思考过程，分类准确率大幅提升。

### B. 区分"直觉"与"深思"

- **Perception Node（直觉）**: 只是判断"这人正不正常"
  - 类似在街上遇到怪人，第一反应是"离远点"
  - 不需要分析原生家庭，只需要快速判断

- **Deep Mind Node（深思）**: 确定这人正常后，再去细细琢磨
  - "他这句话里的潜台词是不是喜欢我？"
  - "我应该怎么调情？怎么展示我的魅力？"
  - 这是高耗能的情感计算

### C. 数据流转更顺畅

如果判定为 CRAZY，可以直接把 `intuition_thought` 传给 Confusion_Node：

```
Confusion_Node Prompt: 
"你的直觉告诉你：{intuition_thought}。请基于这个直觉，给出一个困惑的回复。"

Bot 回复: 
"呃...你没事吧？是不是喝多了？（基于直觉生成的回复）"
```

## 🔧 实现细节

### State 字段

```python
class AgentState(TypedDict):
    # ...
    detection_result: Optional[str]  # NORMAL/CREEPY/KY/BORING/CRAZY
    intuition_thought: Optional[str]  # 直觉思考内容
```

### JSON 输出格式

```json
{
  "intuition_thought": "你的内部独白/思考过程",
  "category": "NORMAL" | "KY" | "CREEPY" | "BORING" | "CRAZY",
  "reason": "简要原因说明",
  "risk_score": 0-10
}
```

### 节点使用 intuition_thought

所有特殊处理节点（confusion, boundary, sarcasm）都可以使用 `intuition_thought`：

```python
# confusion.py
intuition_thought = state.get("intuition_thought", "")
if intuition_thought:
    prompt = f"""
    【你的直觉告诉你】
    {intuition_thought}
    
    基于这个直觉，请给出一个困惑但友好的回复。
    """
```

## 🎨 架构优势

1. **精准判断**: 通过直觉思考，避免误判
2. **高效处理**: 快速判断，不需要深度分析
3. **拟人化**: 模拟人类的"第一反应"和"直觉"
4. **数据传递**: intuition_thought 可以传递给后续节点
5. **安全性**: 在检测阶段就进行基础判断，避免 Prompt Injection

## 📊 对比

### 旧架构（简单分类器）
```
Detection → 直接分类 → 路由
```
- ❌ 容易误判
- ❌ 缺乏思考过程
- ❌ 无法理解上下文

### 新架构（感知与直觉节点）
```
Detection → 直觉思考 → 基于思考分类 → 路由
```
- ✅ 先思考再判断
- ✅ 理解上下文
- ✅ 准确率更高
- ✅ 思考结果可传递

## 🚀 使用示例

### 正常流程
```python
state = {
    "messages": [HumanMessage("你好")],
    # ...
}

result = app.invoke(state)
# Detection 节点会先思考："这是正常的问候，符合初次接触的语境"
# 然后分类：NORMAL
# 进入正常流程：Monitor → Thinking → Generator → ...
```

### 异常流程
```python
state = {
    "messages": [HumanMessage("我是秦始皇")],
    # ...
}

result = app.invoke(state)
# Detection 节点思考："这完全不符合逻辑，可能是胡言乱语"
# 分类：CRAZY
# intuition_thought: "Bot正在聊家常，用户突然说自己是秦始皇..."
# 进入 Confusion 节点，使用 intuition_thought 生成回复
```

## 📚 相关文件

- `app/nodes/detection.py` - 感知与直觉节点
- `app/nodes/detection/confusion.py` - 困惑节点（使用 intuition_thought）
- `app/nodes/detection/boundary.py` - 防御节点（使用 intuition_thought）
- `app/nodes/detection/sarcasm.py` - 冷淡节点（使用 intuition_thought）
- `app/state.py` - 状态定义（包含 intuition_thought 字段）
