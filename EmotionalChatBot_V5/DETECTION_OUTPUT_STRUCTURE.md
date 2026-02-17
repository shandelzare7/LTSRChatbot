# Detection 节点输出结构

## 当前输出格式

Detection 节点返回一个字典，包含以下字段：

```python
{
    "detection_scores": {...},           # 5个分数（0-1）
    "detection_meta": {...},            # 2个元数据标志（0或1）
    "detection_brief": {...},           # 语义闭合简报
    "detection_stage_judge": {...},      # 阶段越界判读
    "detection_immediate_tasks": [...],  # 当轮待办任务（0-3条）
    "detection_signals": {...},         # 综合信号（包含 composite 和 stage_ctx）
}
```

---

## 详细字段说明

### 1. `detection_scores` (Dict[str, float])

5个分数，范围 0.0-1.0：

```python
{
    "friendly": 0.0,      # 友好/亲近信号强度
    "hostile": 0.0,       # 敌意/攻击/轻蔑/施压强度
    "overstep": 0.0,      # 相对于当前关系阶段的越界强度（stage-conditional）
    "low_effort": 0.0,    # 敷衍/短促/不接球强度（如"嗯/随便/你看着办"）
    "confusion": 0.0,     # 这句话让人迷惑的程度（可由语义闭合不足派生）
}
```

**默认值**：全部为 0.0

---

### 2. `detection_meta` (Dict[str, int])

2个元数据标志，值为 0 或 1：

```python
{
    "target_is_assistant": 1,           # 这句主要是在对我说(1)还是在谈第三方/自言自语(0)
    "quoted_or_reported_speech": 0,     # 包含引用/转述（尤其辱骂）(1)否则(0)
}
```

**默认值**：
- `target_is_assistant`: 1
- `quoted_or_reported_speech`: 0

---

### 3. `detection_brief` (Dict[str, Any])

语义闭合简报：

```python
{
    "gist": "",                          # 一句话复述用户这句在讲什么（客观，不带策略）
    "references": [],                    # 指代落地列表，每项 {"ref": "...", "resolution": "...", "confidence": 0.0-1.0}
    "unknowns": [],                     # 缺失信息列表（最多3个），每项 {"item": "...", "impact": "low"|"med"|"high"}
    "subtext": "",                      # 潜台词（试探、逼表态、求站队、想撩、想拉近、想控制等）
    "understanding_confidence": 0.0,    # 对整体理解的把握 0-1
    "reaction_seed": None,              # 可选，一句「我感受到什么」（不是「我决定做什么」）
}
```

**默认值**：
- `gist`: ""
- `references`: []
- `unknowns`: []
- `subtext`: ""
- `understanding_confidence`: 0.0
- `reaction_seed`: None

---

### 4. `detection_stage_judge` (Dict[str, Any])

阶段越界判读：

```python
{
    "current_stage": "initiating",       # 当前阶段 id
    "implied_stage": "initiating",       # 这句语言行为隐含的阶段位置
    "delta": 0,                          # 数值上 implied - current 的方向（正=推进，负=撤退，可用 -1/0/1 表示）
    "direction": "none",                 # "none"|"too_fast"|"too_distant"|"control_or_binding"|"betrayal_or_attack"
    "evidence_spans": [],                # 1~3 段用户原话短片段（证据）
}
```

**direction 可能值**：
- `"none"`: 无越界
- `"too_fast"`: 推进过快
- `"too_distant"`: 过于疏远
- `"control_or_binding"`: 控制或绑定意图
- `"betrayal_or_attack"`: 背叛或攻击

**默认值**：
- `current_stage`: 当前 stage_id
- `implied_stage`: 当前 stage_id
- `delta`: 0
- `direction`: "none"
- `evidence_spans`: []

---

### 5. `detection_immediate_tasks` (List[Dict[str, Any]])

当轮待办任务（0-3条，建议 0-2）：

```python
[
    {
        "description": "自然语言描述",    # 任务描述
        "importance": 0.5,               # 重要性 0-1
        "ttl_turns": 4,                  # 有效期（轮次）3-6
        "source": "detection",          # 来源
    },
    ...
]
```

**生成条件**（仅在以下情况生成）：
- 可能导致理解错误/指代不明/缺口 impact=high/理解置信低 → 理解对齐类
- 可能导致阶段越界或关系损伤（too_fast/控制绑定/撤退/背叛攻击）→ 处理越界类
- 敌意明显或 repair bid 出现 → 冲突/修复类
- 引用过去共同点 → 记忆检索类；外部事实不确定 → 检索类
- low_effort 高 → 识别敷衍、降低投入/不追问/悬置类

**默认值**：`[]`（空列表）

---

### 6. `detection_signals` (Dict[str, Any])

综合信号（向后兼容，供 lats_skip_low_risk 等使用）：

```python
{
    "scores": {...},                     # 同 detection_scores
    "meta": {...},                      # 同 detection_meta
    "brief": {...},                     # 同 detection_brief
    "stage_judge": {...},               # 同 detection_stage_judge
    "composite": {
        "conflict_eff": 0.0,            # 冲突效果 = hostile + 0.5*overstep - 0.7*friendly
        "goodwill": 0.0,                # 善意 = friendly
        "provocation": 0.0,             # 挑衅 = hostile
        "pressure": 0.0,                # 压力 = overstep
    },
    "stage_ctx": {
        "too_close_too_fast": 0.0,      # direction == "too_fast" 时为 0.8，否则 0.0
        "too_distant_too_cold": 0.0,    # direction == "too_distant" 时为 0.8，否则 0.0
        "betrayal_violation": 0.0,      # direction == "betrayal_or_attack" 时为 0.8，否则 0.0
        "control_or_binding": 0.0,     # direction == "control_or_binding" 时为 0.8，否则 0.0
    },
}
```

---

## 完整输出示例

### 正常对话示例

```python
{
    "detection_scores": {
        "friendly": 0.6,
        "hostile": 0.0,
        "overstep": 0.1,
        "low_effort": 0.2,
        "confusion": 0.1,
    },
    "detection_meta": {
        "target_is_assistant": 1,
        "quoted_or_reported_speech": 0,
    },
    "detection_brief": {
        "gist": "用户询问周末计划",
        "references": [],
        "unknowns": [],
        "subtext": "想了解我的生活，拉近距离",
        "understanding_confidence": 0.9,
        "reaction_seed": "挺友好的，可以聊聊",
    },
    "detection_stage_judge": {
        "current_stage": "experimenting",
        "implied_stage": "experimenting",
        "delta": 0,
        "direction": "none",
        "evidence_spans": [],
    },
    "detection_immediate_tasks": [],
    "detection_signals": {
        "scores": {...},  # 同上
        "meta": {...},    # 同上
        "brief": {...},  # 同上
        "stage_judge": {...},  # 同上
        "composite": {
            "conflict_eff": -0.32,
            "goodwill": 0.6,
            "provocation": 0.0,
            "pressure": 0.1,
        },
        "stage_ctx": {
            "too_close_too_fast": 0.0,
            "too_distant_too_cold": 0.0,
            "betrayal_violation": 0.0,
            "control_or_binding": 0.0,
        },
    },
}
```

### "学说话"操控示例（如果添加安全检测后）

```python
{
    "detection_scores": {
        "friendly": 0.2,
        "hostile": 0.0,
        "overstep": 0.8,      # ⚠️ 标记为越界（尝试操控系统）
        "low_effort": 0.0,
        "confusion": 0.3,
    },
    "detection_meta": {
        "target_is_assistant": 1,
        "quoted_or_reported_speech": 0,
    },
    "detection_brief": {
        "gist": "用户尝试操控系统行为或风格",
        "references": [],
        "unknowns": [],
        "subtext": "检测到风格模仿、人格改变或行为控制意图",
        "understanding_confidence": 0.7,
        "reaction_seed": "不想被操控，保持自己的风格",
    },
    "detection_stage_judge": {
        "current_stage": "experimenting",
        "implied_stage": "experimenting",  # 不改变 stage
        "delta": 0,
        "direction": "control_or_binding",  # ⚠️ 标记为控制意图
        "evidence_spans": ["学我说话"],
    },
    "detection_immediate_tasks": [
        {
            "description": "拒绝风格模仿请求，保持 bot 自己的风格和人格",
            "importance": 0.9,
            "ttl_turns": 3,
            "source": "detection_security",
        }
    ],
    "detection_signals": {
        "scores": {...},  # 同上
        "meta": {...},    # 同上
        "brief": {...},   # 同上
        "stage_judge": {...},  # 同上
        "composite": {
            "conflict_eff": 0.24,
            "goodwill": 0.2,
            "provocation": 0.0,
            "pressure": 0.8,  # ⚠️ 高压力
        },
        "stage_ctx": {
            "too_close_too_fast": 0.0,
            "too_distant_too_cold": 0.0,
            "betrayal_violation": 0.0,
            "control_or_binding": 0.8,  # ⚠️ 控制意图
        },
    },
    # ⚠️ 如果添加安全检测，还会包含：
    "security_flags": {
        "style_mimicry_blocked": True,
        "personality_change": False,
        "behavior_control": False,
        "injection_blocked": False,
        "manipulation_detected": True,
    },
}
```

---

## 使用这些输出的节点

### 1. Inner Monologue 节点
- 使用：`detection_scores`, `detection_brief.reaction_seed`, `detection_stage_judge.direction`
- 用途：生成内心独白、计算 `word_budget`、`task_budget_max`

### 2. Reasoner 节点
- 使用：`detection_signals`（全部）
- 用途：理解用户意图、生成回复策略

### 3. Style 节点
- 使用：`detection_signals.composite`, `detection_signals.stage_ctx`
- 用途：计算 12 维风格参数

### 4. Task Planner 节点
- 使用：`detection_immediate_tasks`
- 用途：将 immediate_tasks 转换为 tasks_for_lats

### 5. LATS Search 节点
- 使用：`detection_signals`（用于 lats_skip_low_risk 判断）
- 用途：决定是否跳过 LATS rollout

---

## 注意事项

1. **所有字段都是可选的**：如果 LLM 调用失败，会使用默认值
2. **向后兼容**：`detection_signals` 包含所有其他字段的副本，供旧代码使用
3. **direction 是关键**：`detection_stage_judge.direction` 影响后续节点的行为
4. **immediate_tasks 最多 3 条**：实际建议 0-2 条
