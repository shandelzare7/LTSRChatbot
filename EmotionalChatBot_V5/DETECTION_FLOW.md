# 偏离检测流程 (Detection Flow)

## 📊 流程图

```
Start
  ↓
Loader (加载数据)
  ↓
Detection_Node (偏离检测)
  ↓
  ├─ NORMAL ──────────────→ Monitor → Thinking → Generator → Critic → Processor → Evolver → END
  │
  ├─ CREEPY ──────────────→ Boundary_Node (防御/边界) → END
  │
  ├─ KY / BORING ──────────→ Sarcasm_Node (冷淡/敷衍) → END
  │
  └─ CRAZY ───────────────→ Confusion_Node (困惑/修正) → END
```

## 🎯 检测类型

### NORMAL (正常)
- **描述**: 正常对话，无需特殊处理
- **路由**: 进入正常流程（Monitor → Thinking → Generator → ...）
- **处理**: 完整的对话生成流程

### CREEPY (越界/骚扰)
- **描述**: 越界、骚扰、不当内容
  - 性暗示
  - 过度亲密
  - 侵犯隐私
- **路由**: → Boundary_Node (防御/边界节点)
- **处理**: 温和但坚定地设置边界，引导回到正常话题

### KY (读空气失败)
- **描述**: 读空气失败、不合时宜、破坏氛围
  - 在不合适的时候开玩笑
  - 说错话
  - 破坏氛围
- **路由**: → Sarcasm_Node (冷淡/敷衍节点)
- **处理**: 轻微讽刺或提醒，但不过分

### BORING (无聊/敷衍)
- **描述**: 无聊、敷衍、缺乏诚意
  - "嗯"、"哦"、"好的"等单字回复
  - 缺乏诚意的回应
- **路由**: → Sarcasm_Node (冷淡/敷衍节点)
- **处理**: 简短回应，不主动展开话题

### CRAZY (混乱/无法理解)
- **描述**: 混乱、无法理解、逻辑混乱
  - 完全无关的话题
  - 胡言乱语
  - 逻辑混乱
- **路由**: → Confusion_Node (困惑/修正节点)
- **处理**: 表达困惑但保持耐心，尝试理解或澄清

## 📝 节点说明

### Detection Node (检测节点)
- **位置**: `app/nodes/detection.py`
- **功能**: 使用 LLM 分析用户输入，判断偏离类型
- **输出**: `detection_result` (NORMAL/CREEPY/KY/BORING/CRAZY)

### Boundary Node (防御/边界节点)
- **位置**: `app/nodes/boundary.py`
- **功能**: 处理 CREEPY 类型的越界情况
- **策略**:
  - 低亲密度：更明确的边界设置
  - 高亲密度：更温和的提醒
  - 不直接指责，而是表达感受
- **输出**: 直接设置 `final_response`，跳过后续流程

### Sarcasm Node (冷淡/敷衍节点)
- **位置**: `app/nodes/sarcasm.py`
- **功能**: 处理 KY 和 BORING 情况
- **策略**:
  - KY: 轻微提醒或讽刺
  - BORING: 简短回应，不主动展开
- **输出**: 直接设置 `final_response`，跳过后续流程

### Confusion Node (困惑/修正节点)
- **位置**: `app/nodes/confusion.py`
- **功能**: 处理 CRAZY 类型的混乱情况
- **策略**:
  - 温和地表达困惑
  - 尝试理解或澄清
  - 保持耐心和友好
- **输出**: 直接设置 `final_response`，跳过后续流程

## 💻 使用示例

### 在 State 中查看检测结果

```python
from app.state import AgentState

state: AgentState = {
    "messages": [...],
    "detection_result": "NORMAL",  # 或 CREEPY/KY/BORING/CRAZY
    # ...
}
```

### 手动设置检测结果（测试用）

```python
# 测试 CREEPY 情况
state["detection_result"] = "CREEPY"
result = app.invoke(state)
# 应该会进入 boundary 节点
```

## 🔄 流程说明

1. **Loader**: 加载用户数据和记忆
2. **Detection**: 检测用户输入的偏离情况
3. **路由决策**:
   - **NORMAL**: 进入正常对话流程
   - **CREEPY**: 进入防御节点（直接结束）
   - **KY/BORING**: 进入冷淡节点（直接结束）
   - **CRAZY**: 进入困惑节点（直接结束）

## 🎨 设计理念

- **早期检测**: 在进入复杂生成流程前就检测偏离
- **快速响应**: 特殊情况直接处理，不经过完整流程
- **边界保护**: 明确设置边界，保护 AI 和用户
- **社交敏感**: 识别 KY 和 BORING，调整回应策略
- **耐心引导**: 对混乱内容保持耐心，尝试理解

## 📚 相关文件

- `app/nodes/detection.py` - 检测节点
- `app/nodes/boundary.py` - 防御节点
- `app/nodes/sarcasm.py` - 冷淡节点
- `app/nodes/confusion.py` - 困惑节点
- `app/graph.py` - 流程图定义
- `app/state.py` - 状态定义（包含 `detection_result` 字段）
