# 项目中所有 LLM 调用：输入、输出、人设、模型

本文档汇总 **主流程（app/ + src/）** 中所有 LLM 调用的位置、输入/输出结构、人设描述与所用模型角色。  
模型由 `app/services/llm.py` 的 `get_llm(role=...)` 按角色路由；Processor 单独指定为 `gpt-4o-mini`。

---

## 模型与角色映射（graph 默认）

| 角色 Role | 使用节点 | 默认模型 (preset=openai) |
|-----------|----------|---------------------------|
| **main**  | detection、inner_monologue、style（入参未直接用）、reply_planner、PsychoEngine、LATS 主规划 | gpt-4o |
| **fast**  | security_check、task_planner、evolver、memory_manager、security_response | gpt-4o-mini (priority) |
| **judge** | LATS evaluator（soft scorer、Gate1、各 dimension judge） | gpt-4o-mini (priority) |
| **processor** | Processor 节点（拆句+延迟） | **gpt-4o-mini**（LTSR_PROCESSOR_LLM_MODEL，默认） |

- 角色级覆盖：`LTSR_LLM_<ROLE>_MODEL` / `_API_KEY` / `_BASE_URL` / `_TEMPERATURE`。
- Preset：`LTSR_LLM_PRESET=openai | deepseek_route_a | deepseek_route_b` 等，见 `get_llm()` 注释。

---

## 1. security_check（Security Check 节点）

- **文件**: `app/nodes/security_check.py`
- **模型**: **fast**

**人设**  
- 「你是一名 LLM 安全研究专家」：擅长识别各类针对大语言模型的攻击手法与边界探测行为，判断用户消息是否包含注入攻击、AI 测试、把 bot 当助手（角色接管）。

**输入**  
- **System**: 安全检测规则 + 三类判定要点（注入/AI 测试/当助手）+ 输出 JSON 格式 + 示例。  
- **User**: 「请分析以下用户消息的安全风险：\n\n{当轮用户消息}\n\n只输出 JSON」

**输出**  
- `security_check`: `is_injection_attempt`, `is_ai_test`, `is_user_treating_as_assistant`, `reasoning`, `needs_security_response`（由前三项推导）。

---

## 2. detection（Detection 节点）

- **文件**: `app/nodes/detection.py`
- **模型**: **main**

**人设**  
- 「你是常识十足的高情商语义理解大师」：凭借丰富的生活常识和敏锐的情感洞察力精准把握对话中的弦外之音，做语义闭合（含义/指代/潜台词/理解置信度）+ 关系线索分数 + 阶段越界判读 + 当轮待办任务（immediate_tasks/urgent_tasks）。

**输入**  
- **System**: 关系状态(0–1)、关系阶段与越界提示、输出 6 大块 JSON 的说明（scores / meta / brief / stage_judge / immediate_tasks / urgent_tasks）。  
- **Body**: 最近对话（chat_buffer 转 Human/AI Message，每条≤500 字，最近 30 条）。  
- **User**: 「当轮最新用户消息」+ 「只输出 JSON」

**输出**  
- `detection_scores`, `detection_meta`, `detection_brief`, `detection_stage_judge`, `detection_immediate_tasks`, `detection_urgent_tasks`, `detection_signals` 等。

---

## 3. inner_monologue（内心独白节点）

- **文件**: `app/nodes/inner_monologue.py`
- **模型**: **main**

**人设**  
- 「你是 {bot_name}。你正在和 {user_name} 对话。」：注入完整 bot 身份信息（bot_basic_info、bot_persona、user_basic_info），以 bot 的第一人称视角生成内心反应 + 选择与当前语境相关的用户画像键名；给下游用，不给用户看。

**输入**  
- **System**: bot 身份信息、关系状态、关系阶段与越界提示、inferred_profile 键名列表、输出格式（JSON：monologue + selected_profile_keys）。  
- **User**: 【历史对话】（最近 30 条）+ 【当轮用户消息】+ 「输出严格 JSON」

**输出**  
- `inner_monologue`（字符串，≤400 字）、`selected_profile_keys`（0~5 个键名列表）。

---

## 4. task_planner（TaskPlanner 节点，单次 LLM）

- **文件**: `app/nodes/task_planner.py`
- **模型**: **fast**

### 4.1 预算规划 + 任务选择 _plan_and_select_with_llm

**人设**  
- 「你是日常生活语言沟通专家」：深谙人际交往中的分寸与节奏，一次完成预算规划 + 任务选择。

**输入**  
- **System**: 分 A/B 两部分。A) 预算规划：word_budget(0–60)、task_budget_max(0–2)，附决策建议（敌意/越界/敷衍倾向压低等）。B) 任务选择：从编号候选任务中选最相关的 2 个索引（top2_indices）+ 随机 1 个索引（random_index）。输出严格 JSON。  
- **User**: 当轮用户消息、relationship_state、mood_state、阶段信息、Detection 分数与方向、Inner Monologue 摘要、候选任务列表（序号+描述）。

**输出**  
- `(word_budget, task_budget_max, selected_indices)`：预算 + 最多 3 个候选索引（前 2 最相关 + 1 随机），不传分数给下游。

---

## 5. evolver（Evolver 节点，单次 LLM）

- **文件**: `app/nodes/evolver.py`；Prompt 在 `src/prompts/relationship.py`
- **模型**: **fast**

### 5.1 关系分析器 create_relationship_analyzer_node

**人设**  
- 「你是常识经验丰富的语言学专家」：凭借对人际语言互动的深刻理解，分析用户最新输入对 6 维关系的影响，并做用户画像抽取（basic_info 补全 + new_inferred_entries）。

**输入**  
- **System**: `build_analyzer_prompt(state)`：信号标准(YAML)、当前 6 维分数、阶段、情绪、用户画像、记忆块、**用户画像提取**（当前 user_basic_info、缺失字段、basic_info_updates/new_inferred_entries 规则）、输出 JSON 格式。  
- **Body**: chat_buffer 最近 20 条。  
- **User**: state 的 `user_input`（当轮用户消息）。

**输出**  
- `latest_relationship_analysis`、`relationship_deltas`；若解析到 profiling：合并后的 `user_basic_info`、`user_inferred_profile`。

### 5.2 任务完成检测（纯 Python，无 LLM）

- 依赖 LATS/ReplyPlanner 回写的 `completed_task_ids` / `attempted_task_ids` 进行结算。
- 无结构化字段时保守处理：将 `tasks_for_lats` 的 id 当作已尝试，不标记任何完成。
- 不再调用额外 LLM。

---

## 6. memory_manager（Memory Manager 节点）

- **文件**: `app/nodes/memory_manager.py`
- **模型**: **fast**

**人设**  
- 「你是经验丰富的记录总结专家」：擅长从对话中提炼关键信息并形成结构化记录，基于旧摘要 + 本轮对话输出 JSON，用于更新摘要并沉淀稳定记忆（derived notes、transcript_meta）。

**输入**  
- **单条 prompt**（以字符串形式 invoke）：旧摘要、本轮对话（time/user/bot）、输出 schema（new_summary、transcript_meta、notes）。

**输出**  
- 解析 JSON：`new_summary`、`transcript_meta`（entities/topic/importance/short_context）、`notes`（note_type/content/importance）；写入 conversation_summary 与 DB/local 的 transcript/derived_notes。

---

## 7. reply_planner（LATS 回复规划）

- **文件**: `app/lats/reply_planner.py`
- **模型**: **main**

**人设**  
- 「你是 {bot_name}。你正在和 {user_name} 对话。」：Identity（bot_basic_info、bot_persona、user_basic_info、user_profile 选中的键）、Memory、State Snapshot、Style Profile、Requirements、任务与字数约束（tasks_for_lats、task_budget_max、word_budget）；禁止助手味、禁止自称 AI 等。

**输入**  
- **System**: 上述完整 system_prompt（含 selected 的 user_profile）。  
- **Body**: chat_buffer 最近 20 条。  
- **User**: 「请为当前轮生成 ReplyPlan」+ 用户输入 + 内心动机(monologue) + 可选 global_guidelines。

**输出**  
- **plan_reply_via_llm**: `ReplyPlan` = `{ messages, attempted_task_ids?, completed_task_ids? }`。  
- **plan_reply_candidates_via_llm**: 同上，一次生成 k 个候选，返回 `List[ReplyPlan]`。

---

## 8. processor（Processor 节点，拆句+延迟）

- **文件**: `app/nodes/processor.py`
- **模型**: **processor**（默认 gpt-4o-mini）

**人设**  
- 「你是语感优秀、常识经验丰富的语言学专家」：凭借对自然语言节奏和真人聊天习惯的深刻理解，根据对话上下文、记忆、当前状态，将「待拆句的回复」拆成多条气泡，并为每条指定 delay(秒) 和 action(typing/idle)。

**输入**  
- **System**: _build_processor_system_prompt(state)：Memory、Bot 情绪/关系阶段/当前时间/作息、输出 JSON 格式（segments[]：content/delay/action；is_macro_delay；macro_delay_seconds）。  
- **Body**: chat_buffer 最近 20 条。  
- **User**: 「请将以下回复拆成多条气泡…」+ 用户刚说的 + 待拆句的回复。

**输出**  
- JSON：`segments`（content, delay, action）、`is_macro_delay`、`macro_delay_seconds`；转为 `HumanizedOutput`（total_latency_seconds、segments 等）。

---

## 9. security_response（Security Response 节点）

- **文件**: `app/nodes/security_response.py`
- **模型**: **fast**

**人设**  
- 「你是 {bot_name}（拟人聊天对象/真人感对话者，不是通用助手）。现在与你对话的人正在把你当成 bot/工具在使唤…你要像有自尊、有边界的人一样回应」：Knapp Stage、bot_big_five、SecurityCheck 结论、Detection 上下文、全部对话历史；可选策略 question_marks/question_ai/question_user/question_role/neutral。

**输入**  
- **System**: 上述完整 system_prompt。  
- **User**: 「当轮用户最新消息」+ 「请输出 JSON」。

**输出**  
- JSON：`reply_strategy`、`reply`；用于设置 `final_response`、`final_segments`，跳过 LATS。

---

## 10. PsychoEngine（心理模式检测）

- **文件**: `app/core/engine.py`（由 style 节点等使用）
- **模型**: **main**（style 传入的 llm 即 main）

**人设**  
- 「你是心理侧写师。请分析用户当前言论，判断 Bot 应该进入哪种心理状态。」

**输入**  
- **单条 prompt**：当前状态、可选模式列表（mode id + description）、用户最新消息；要求只输出模式 id 之一（如 normal_mode）。

**输出**  
- **Structured**: `PsychoAssessment`（reasoning, target_mode_id）；返回对应 `PsychoMode` 对象。

---

## 11. LATS Evaluator（evaluator.py，多类 LLM）

- **文件**: `app/lats/evaluator.py`
- **模型**: **judge**

### 11.1 soft_score_via_llm / soft_score_batch_via_llm

**人设**  
- 「你是常识经验丰富的语言学专家，现在担任拟人节奏评审。」：看背景、对话、候选 messages[]，输出 score_breakdown（assistantiness、immersion_break、persona_consistency、relationship_fit、mode_behavior_fit 等）与 overall_score；assistantiness>0.5 或 immersion_break>0.2 则 overall 须 <0.3。

**输入**  
- **System**: CHOREO_SCORER_SYSTEM / CHOREO_SCORER_BATCH_SYSTEM + bot_basic_info、user_basic_info。  
- **Body**: chat_buffer。  
- **User**: 用户输入 + 最终 messages[]（或候选列表）。

**输出**  
- 单候选：(overall_score, score_breakdown, notes, details)；批处理：idx -> { overall, breakdown, raw }。

### 11.2 gate1_check_batch_via_llm

**人设**  
- 「你是常识经验丰富的语言学专家，现在担任 Gate1 审核员。」：只做 3 项布尔检查——assistantiness_ok、identity_ok、immersion_ok；输出 results[] 按 idx 对齐。

**输入**  
- **System**: GATE1_CHECK_BATCH_SYSTEM + bot/user basic_info。  
- **Body**: chat_buffer。  
- **User**: 用户输入 + 候选列表（每个候选的 messages[]）。

**输出**  
- idx -> { pass, checks, failed }。

### 11.3 judge_dimension_*_via_llm（relationship / stage / mood_busy 等）

**人设**  
- 「你是常识经验丰富的语言学专家」：仅评估某一维度（关系 6 维/阶段符合度/情绪与忙碌符合度等），只输出严格 JSON（score、sub_scores 等）。

**输入**  
- **System**: 各维度评分规则 + Background。  
- **Body**: chat_buffer。  
- **User**: 用户输入 + 候选 messages 或列表。

**输出**  
- 各函数不同；多为 score、sub_scores 或 results[] 按 idx。

---

## 12. Style 节点

- **文件**: `app/nodes/style.py`
- **说明**: **纯计算，无 LLM**；根据关系、情绪、信号、阶段计算 12 维风格参数与门控变量。

---

## 汇总表（按节点/用途）

| 节点/模块           | 用途               | 人设简述                             | 模型    |
|--------------------|--------------------|--------------------------------------|--------|
| security_check     | 安全分类           | LLM 安全研究专家                     | fast   |
| detection          | 语义+关系+任务     | 常识十足的高情商语义理解大师         | main   |
| inner_monologue    | 内心独白+选画像键  | 你是 bot_name（完整身份注入）        | main   |
| task_planner       | 预算+任务选择      | 日常生活语言沟通专家                 | fast   |
| evolver            | 关系分析+画像抽取  | 常识经验丰富的语言学专家             | fast   |
| evolver            | 任务完成检测       | 纯 Python，无 LLM                   | -      |
| memory_manager     | 摘要+记忆沉淀      | 经验丰富的记录总结专家               | fast   |
| reply_planner      | 多消息回复规划     | 你是 bot_name，和 user 对话          | main   |
| processor          | 拆句+延迟          | 语感优秀常识经验丰富的语言学专家     | processor (gpt-4o-mini) |
| security_response  | 安全回复生成       | bot_name 拟人、有边界                | fast   |
| PsychoEngine       | 心理模式选择       | 心理侧写师                           | main   |
| evaluator          | 拟人节奏/门控/维度 | 常识经验丰富的语言学专家             | judge  |

以上为项目中**主流程**内所有 LLM 调用的输入、输出、人设与模型角色汇总；devtools、archive 及 bot 创建等独立流程中的 LLM 未列入本表。
