# 回复分割逻辑与影响因素

## 一、整体流程（谁决定「几条」）

最终 `final_segments` 的条数由**两条互斥路径**之一决定：

| 路径 | 条件 | 条数由谁决定 |
|------|------|--------------|
| **A. LATS 路径** | LATS 开启且产出了 `processor_plan`（含 `messages`） | **ReplyPlanner LLM** 输出的 `messages` 条数，再经 final_validator 可能合并 |
| **B. Processor 规则路径** | 无可用 `processor_plan`（LATS 关闭/空）且未用 Processor LLM | **HumanizationProcessor._segment_text()**：按句号/感叹/问号/换行 + 长度阈值切 `final_response` |

默认走 **A**；**B** 仅在 mute_mode、或 LATS 未产出有效 plan 时生效。

---

## 二、路径 A：LATS 路径（分割「少/多」由谁定）

### 2.1 条数来源

- **ReplyPlanner**（`reply_planner.plan_reply_via_llm`）根据 prompt 生成 `ReplyPlan.messages`，**条数 = 1～max_messages**。
- LATS 把最优 `ReplyPlan` 编译成 `ProcessorPlan`（`messages` 列表），Processor 节点直接取 `processor_plan["messages"]` 作为 `final_segments`，**不再做二次切句**。

因此：**路径 A 下「分割少」= Planner 只输出 1 条；「分割多」= 输出 2～max_messages 条**。

### 2.2 影响条数的因子（路径 A）

| 因子 | 作用 | 少段倾向 | 多段倾向 |
|------|------|----------|----------|
| **max_messages**（requirements） | 硬上限，且 final_validator 会按此合并超出的条 | 设小（如 1） | 设大（如 5） |
| **mode_id** | 决定默认 max_messages | cold_mode→1；mute_mode→0 | normal_mode→3 |
| **lats_budget.max_messages** | 覆盖 mode 的 max_messages | 配置小 | 配置大 |
| **word_budget** | 总字数上限，影响 Planner 倾向写长/写短 | 很小（如 20）→ 易一条说完 | 大（如 80）→ 易多条 |
| **ReplyPlanner 的 prompt** | 「多条消息」「先回应再补充」等 | 任务简单/用户只问一句 | 任务多、要分步回应/解释/反问 |
| **style_targets（如 verbal_length）** | 风格目标进 requirements，影响 Planner 话多话少 | 偏简短 | 偏长/丰富 |
| **final_validator** | 首条过短会与第二条合并；条数 > max_messages 从尾部合并 | 首条 < min_first_len 或 条数 > max_messages | 首条 ≥ min_first_len 且 条数 ≤ max_messages |
| **min_first_len**（requirements） | 首条低于此次数会触发合并 | 设大 → 更容易触发合并 → 少段 | 设小 → 少合并 → 多段 |
| **evaluator hard_gate** | 候选若 messages 条数 > max_messages 直接淘汰 | 同上限 | 同上限 |

小结：**路径 A 下要「分割多」**：max_messages 足够大、word_budget 足够、任务/风格偏「分步说」；**要「分割少」**：max_messages=1（如 cold_mode）或 word_budget 很小，或 final_validator 合并掉多余条。

---

## 三、路径 B：Processor 规则分割（_segment_text）

仅当**没有**可用 `processor_plan` 且未用 Processor LLM 时，对 `final_response` 做规则切分。

### 3.1 算法要点（`processor._segment_text`）

- **切分点**：仅在有 **句号/感叹号/问号/换行**（`。！？\n`）处考虑断句；**逗号、顿号不断**。
- **长度阈值**：`split_threshold = 45 - fragmentation_tendency * 40`，再 clamp 到 [5, 60]（字符数）。
  - 遇到 `。！？` 时：若当前累计 `len(current_buf) >= split_threshold` 则输出为一段并清空。
  - 遇到 `\n` 时：**无条件**输出为一段（不比较阈值）。
- **过滤**：长度 < `MIN_BUBBLE_LENGTH`（2）的段会先被滤掉（若全部被滤则退回未滤结果）。

因此：
- **少段**：`final_response` 很少出现 `。！？\n`，或 tendency 低导致 threshold 高（如 45），容易多句并成一段。
- **多段**：`final_response` 里 `。！？\n` 多，或 tendency 高导致 threshold 低（如 5），几乎每句都断。

### 3.2 影响路径 B 段数的因子

| 因子 | 作用 | 少段倾向 | 多段倾向 |
|------|------|----------|----------|
| **fragmentation_tendency**（0～1） | 决定 split_threshold（5～60 字符） | 低（≈0）→ threshold≈45，长段才断 | 高（≈1）→ threshold≈5，易多段 |
| **extraversion**（大五） | tendency = 0.4×e + 0.4×closeness + 0.2×arousal | 低 | 高 |
| **closeness**（关系） | 同上 | 低 | 高 |
| **arousal**（情绪） | 同上 | 低 | 高 |
| **final_response 内容** | 只在 `。！？\n` 处断句 | 少标点、长句、无换行 | 多句号/问号/感叹/换行 |
| **MIN_BUBBLE_LENGTH** | 过滤极短段 | 几乎不改变条数 | - |

小结：**路径 B 下要「分割多」**：提高 extraversion/closeness/arousal（提高 tendency）、或让 `final_response` 多写 `。！？` 和换行；**要「分割少」**：tendency 低、回复少标点少换行。

---

## 四、路径 C：Processor LLM 拆句（可选）

- 当 `processor_use_llm` 为真且**没有**可用 `processor_plan` 时，会调 `_humanize_via_llm`，由 **LLM 直接返回 segments 列表**。
- 条数完全由模型和 prompt 决定，无固定公式；prompt 要求「至少 1 条，短回复可只 1 条」。

---

## 五、因子汇总表（按类型）

| 类型 | 因子 | 少段 | 多段 |
|------|------|------|------|
| **配置/模式** | max_messages | 小（如 1） | 大（如 5） |
| | mode_id | cold_mode / mute_mode | normal_mode |
| | word_budget | 小 | 大 |
| | min_first_len | 大（易触发合并） | 小 |
| **LATS/Planner** | ReplyPlanner 行为 | 简单问句、单句可答 | 多任务、需分步/解释/反问 |
| | style_targets（verbal_length 等） | 偏短 | 偏长/丰富 |
| **Validator** | 首条长度 | < min_first_len → 与第二条合并 | ≥ min_first_len |
| | 条数 | > max_messages → 尾部合并 | ≤ max_messages |
| **规则分割（路径 B）** | fragmentation_tendency | 低 | 高 |
| | extraversion / closeness / arousal | 低 | 高 |
| | final_response 中标点/换行 | 少 `。！？\n` | 多 `。！？\n` |

---

## 六、如何快速调「整句 vs 多段」

- **想要整句（少段）**  
  - 调配置：`max_messages=1`（或 cold_mode）、或把 `word_budget` 调小。  
  - 若走规则分割：降低 extraversion/closeness/arousal，或让生成回复少用 `。！？`（偏长句）。

- **想要多段**  
  - 调配置：`max_messages=3～5`、`word_budget` 适当放大。  
  - 若走规则分割：提高 extraversion/closeness/arousal；或让 `final_response` 多句、多 `。！？` 和换行。

- **实际多数对话**：走 LATS 路径，条数主要由 **max_messages + ReplyPlanner 的 LLM 行为** 决定，final_validator 的合并只做兜底。
