# 长期陪伴拟人化Chatbot技术架构实现

## 引言

长期陪伴型拟人化聊天机器人，是近年来人机交互、计算社会科学与人工智能交叉领域的重要研究方向。与传统问答式对话系统不同，这类系统的目标并不仅仅是"正确回答问题"，而是**在数百乃至上千轮对话中，持续维持一种稳定而细腻的关系感**：它要像一个具体的人，有相对稳定的人格特质、可感知的情绪波动、可追踪的关系发展轨迹，还要能够记住用户的长期偏好与关键事件，并在关键时刻做出合乎"关系阶段"的行为选择。

LTSRChatbot（Long-Term Social Relationship Chatbot，本项目）正是在这样的研究背景下设计与实现的一个**以长期关系为中心的拟人化 chatbot 技术架构**。与普通的 LLM 应用相比，本系统有三个鲜明定位：

- **以关系为一等公民**：不仅追求单轮回复质量，更强调"关系状态"和"关系阶段"的演化是否合理。
- **以心理学理论为结构骨架**：融合大五人格（Big Five）、PAD 情绪模型、Knapp 关系发展阶段模型与社会穿透理论（Social Penetration Theory, SPT），将这些理论显式编码进系统状态与演化逻辑。
- **以工程化编排与可观测性为前提**：基于 LangGraph 对各个"心理与行为节点"进行编排，使用高度结构化的 `AgentState` 驱动模型调用与记忆写入，使系统行为可解释、可调参、可演化。

本章节围绕 LTSRChatbot 的技术实现展开，重点回答三个问题：  
（1）系统的**总体技术架构**如何支持长期拟人化陪伴？  
（2）为了实现"能长期相处的人格与关系"，本系统在**核心模块与关键技术**上做了哪些设计？  
（3）这些设计在学术研究层面具有什么**可研究性与创新点**，如何支撑博士论文关于长期陪伴 chatbot 的理论与实证分析？

下文将从系统总体架构入手，逐层剖析状态建模、节点编排、关系演化与记忆系统，继而总结关键技术实现细节与研究创新点，并最终给出对系统的综合评估。

---

## 系统总体架构

从工程角度看，LTSRChatbot 采用典型的**分层架构**，大致可以划分为以下几个层次：

- **基础设施层**：包括数据库（PostgreSQL 或本地文件）、Web 接口（FastAPI + Uvicorn）、LLM 接入层（基于 LangChain / LangGraph，支持 OpenAI、DeepSeek 等多模型路由）、配置系统（基于 YAML 与环境变量）。
- **会话状态层**：由 `app/state.py` 中的 `AgentState` 定义，采用高维 TypedDict 结构，整合身份、人设、关系、情绪、记忆、任务、LATS 搜索参数等**所有跨节点共享的心理与技术状态**。
- **LangGraph 节点编排层**：由 `app/graph.py` 中的 `build_graph` 函数构建，基于 LangGraph 的 `StateGraph(AgentState)` 将各个功能节点（如 `loader`、`detection`、`inner_monologue`、`style`、`task_planner`、`lats_search`、`processor`、`evolver`、`stage_manager`、`memory_manager`、`memory_writer` 等）编排为一个有条件分支与环路的**心理-行为流水线**。
- **心理与关系引擎层**：包括 `PsychoEngine` 心理模式引擎、多种 `PsychoMode`（normal / defensive / stress / broken）、6 维关系引擎（`evolver` 节点）、Knapp 阶段管理（`stage_manager`），以及基于 LATS 的回复规划与搜索。
- **记忆与数据库层**：由 `memory_manager` 与 `memory_writer` 将每轮对话写入两类记忆存储（转录 Store A、派生笔记 Store B），并通过 `app/core/database.py` 与 `docs/DATABASE_SCHEMA.md` 所描述的 bots / users / messages / derived_notes 结构组织长期记忆。
- **行为输出层**：由 `style` 节点给出 12 维输出驱动，再经过 LATS 搜索与 `processor` 拟人化编排，最终输出带有**分段与延迟的行为脚本**（`HumanizedOutput`），而不仅仅是一段纯文本。

从用户视角看，一次完整的对话轮次的主流程（以正常情况为例）可以概括为：

> 用户输入 → `loader` 加载档案与历史 → `security_check` / `detection` 感知与安全路由 → `inner_monologue` 内心独白与策略构思 → `style` 计算 12 维风格指令 → `task_planner` 规划本轮任务与字数预算 → `lats_search` 基于 LATS 进行回复搜索与评估 → `processor` 拟人化分段与延迟 → `evolver` 更新 6 维关系状态与任务完成情况 → `stage_manager` 更新 Knapp 关系阶段 → `memory_manager` 抽取与更新记忆 → `memory_writer` 写回数据库 / 本地存储。

通过这样的编排，系统在**一次用户消息**与**整个长期关系轨迹**之间建立起了结构化的桥梁：每一轮的分析、规划与输出都会反馈到关系状态、阶段与记忆资产中，进而影响下一轮乃至未来多轮的行为。

---

## 核心模块详细设计

### 1. 状态层：AgentState 的多层心理结构

`app/state.py` 定义了一个极其丰富的 `AgentState` 类型，它可以被视为**整个拟人化 AI 的"心理与身体"联合状态**。其内部结构显式对齐了本项目的理论框架，主要分为以下几个层次：

#### 1.1 Identity Layer（身份层，"我是谁"）

- **BotBasicInfo**：机器人硬性身份信息，如 `name`、`gender`、`age`、`region`、`occupation`、`education`、`native_language`、`speaking_style` 等。这一部分在数据库中相对静态，用于支撑用户对机器人的"人物设定"认知。
- **BotBigFive**：大五人格基准值，范围 \([0.0, 1.0]\)，包括 `openness`、`conscientiousness`、`extraversion`、`agreeableness`、`neuroticism`。这些参数不仅用于文案风格调节，也会参与分段倾向、延迟控制等行为计算。
- **BotPersona**：动态人设结构，包括 `attributes`（键值属性）、`collections`（如 hobbies、skills 列表）、`lore`（背景故事片段）。这一结构设计成松散的 JSON，可通过配置或运行时自动扩展，从而支持**人设的渐进丰富与进化**。

#### 1.2 Perception Layer（感知层，"我看你是谁"）

- **UserBasicInfo**：用户显性信息（`name`、`gender`、`age`、`location`、`occupation` 等），由早期明确提问或表单输入获得。
- **UserInferredProfile**：用户隐性侧写（如性格倾向、情感状态模式、偏好结构等），由 `detection`、`inner_monologue`、`memory_manager` 等节点协同构建，结构为开放式 JSON，用于**在 Prompt 中整体注入**，而不对字段做过度工程约束。

#### 1.3 Physics Layer（物理层，"我们的关系和我的心情"）

- **RelationshipState**：6 维关系属性，原始理论量纲为 \([0, 100]\)，实际实现中采用 \([0, 1]\) 浮点并设置上限 `REL_HI_CAP=0.98`，包括：
  - `closeness`（亲密）、`trust`（信任）、`liking`（喜爱）、`respect`（尊重）、`warmth`（暖意）、`power`（权力）。
- **MoodState**：PAD 情绪模型与繁忙度：
  - `pleasure`、`arousal`、`dominance` \(\in [-1.0, 1.0]\)，
  - `busyness` \(\in [0.0, 1.0]\)，用于控制"忙碌时的缩短回复或长延迟"。

这一层的设计使**关系与情绪**成为可计算对象，而非仅存在于提示词中的隐喻表达。

#### 1.4 Memory Layer（记忆层，"我记得什么"）

- **短期记忆**：`chat_buffer`（最近若干轮消息）、`conversation_summary`（滚动摘要）。
- **检索记忆**：`retrieved_memories`、`retrieval_ok`、`memory_context` 等，支持 RAG 风格的"从长期记忆中取回与本轮相关的一小块文本"，供 Reasoner / Reply Planner 使用。
- **关系资产**：`relationship_assets` 中的 `topic_history`、`breadth_score`、`max_spt_depth`，对齐社会穿透理论中的**话题广度与自我暴露深度**。

#### 1.5 Output Layer（输出驱动层，"我这轮要怎么表现"）

- **llm_instructions**：12 维输出驱动（Strategy + Style），包括：
  - Strategy 维度：`self_disclosure`、`topic_adherence`、`initiative`、`advice_style`、`subjectivity`、`memory_hook`；
  - Style 维度：`verbal_length`、`social_distance`、`tone_temperature`、`emotional_display`、`wit_and_humor`、`non_verbal_cues`。
- **ReplyPlan / ProcessorPlan / HumanizedOutput**：从 LATS 的计划（`ReplyPlan`）到 Processor 的可执行计划（`ProcessorPlan`），再到最终呈现层面的 `HumanizedOutput`，将"高维行为意图"一步步编译成**可播放的打字与发言时间线**。

此外，`AgentState` 还包含任务系统（`bot_task_list`、`current_session_tasks`、`tasks_for_lats`、`task_budget_max` 等）、LATS 搜索参数（`lats_rollouts`、`lats_llm_soft_top_n` 等）、安全检测结果、模式信息（`mode_id`、`current_mode`）以及 `_profile` 性能追踪字段。这些设计使整个系统在**一份统一、强类型的状态对象上运转**，既利于 LangGraph 的跨节点状态合并，又便于研究者从日志中还原每一轮决策过程。

---

### 2. 节点编排层：LangGraph 心理-行为流水线

`EmotionalChatBot_V5/GRAPH_FLOW.md` 与 `app/graph.py` 描述了系统基于 LangGraph 的完整流程。以新版主路径为例，主要节点包括：

- `loader`：从数据库或本地存储加载当前 `(bot_id, user_id)` 下的关系状态、历史摘要、任务列表等，初始化 `AgentState`。
- `security_check` / `security_response`：对用户输入做安全检测，必要时触发防御性回复并短路整个流程。
- `detection`：对本轮输入进行情境分类（NORMAL / CREEPY / KY/BORING / CRAZY），并产出一系列 detection signals、scores 与 immediate / urgent tasks。
- `inner_monologue`：生成机器人本轮的"内心独白"和对用户的直觉判断（relationship_filter、intuition_thought 等），并选择需要注入的用户侧写字段。
- `style`：基于当前关系状态、情绪、阶段与模式，计算 12 维输出驱动，用于下游的回复规划与生成。
- `task_planner`：将 detection 与记忆中的任务合并，为本轮 LATS 提供 `tasks_for_lats` 以及字数预算 `word_budget` 等约束。
- `lats_search`：调用 LATS 搜索算法生成多个回复候选，并通过 LLM soft scorer 进行多维评分与早退控制。
- `processor`：将选中的回复文本编译为带有多段文本与对应延迟的行为脚本，结合打字模拟、睡眠/忙碌状态等生成 `HumanizedOutput`。
- `evolver`：执行 6 维关系演化与任务完成检测，更新 `relationship_state`、`current_session_tasks`、`bot_task_list` 等。
- `stage_manager`：基于 Knapp 十阶段与 SPT 指标，更新 `current_stage` 与相关叙事信息。
- `memory_manager`：对本轮"用户输入 + 机器人回复 + 旧摘要"做一次 LLM 总结与信息抽取，生成新的 `conversation_summary`、derived notes 与 transcript 元数据，同时抽取基础信息与新的用户隐性特征。
- `memory_writer`：负责最终的数据库写入或本地 JSONL 存储，完成"Commit Late"。

LangGraph 的优势在于：每个节点只需要关心**局部的输入输出与自身逻辑**，而状态的合并、条件边路由与并行执行由框架负责。这一设计让本系统的架构在研究上具有高度的**可解释性与可重构性**：研究者可以方便地插入新的"心理节点"（如新的情绪评估器、价值观冲突检测模块），或调整节点顺序与条件边，来验证不同架构对长期关系体验的影响。

---

### 3. 心理模式系统与 PsychoEngine

在长期陪伴场景下，机器人不能永远保持一种"正常模式"。用户可能会出现攻击性言论、自伤暗示、过度依赖、冷淡疏远等多种情境，机器人需要具备**心理防御、压力反应与崩溃状态**等更拟人的表现。本系统通过 `app/core/engine.py` 中的 `PsychoEngine` 与 `config/modes/*.yaml` 的配置实现了这一点。

- `PsychoMode`（`mode_base.py`）定义了每种心理模式的：
  - `id` 与 `name`（如 `normal_mode`、`defensive_mode`、`stress_mode`、`broken_mode`）；
  - `system_prompt_template`：约束 LLM 在该模式下的角色认知；
  - `monologue_instruction`：指导 inner_monologue 如何思考；
  - `critic_criteria`、`split_strategy`、`typing_speed_multiplier`、`lats` 预算等。
- `PsychoEngine` 则在每轮对话中调用 LLM（带结构化输出）对当前用户消息与上下文做心理侧写，预测 `target_mode_id`，再从预加载的模式集中选择对应模式。

通过这种**"配置化 + LLM 决策"的心理模式切换机制**，系统能够在工程层面支持如下研究问题：

- 不同心理模式下，机器人在相同输入场景中的语言与行为差异；
- 心理模式切换轨迹与关系阶段、情绪状态之间的耦合关系；
- 通过调整模式配置（如打字速度、回复长度、边界强度）对长期陪伴体验的影响。

---

### 4. LATS 回复规划与搜索

LTSRChatbot 的回复生成并非一次性调用 LLM，而是采用了**Language Agent Tree Search (LATS)** 框架。其核心思想是：将"如何回复"视为一个在意图空间与语言空间中的搜索问题，通过多轮 rollouts 与软评分找到既符合关系与风格约束、又足够自然与多样的回复。

根据 `docs/LATS_AND_LLM_ROUTING.md` 与相关代码，LATS 的关键要素包括：

- **入口节点**：`app/nodes/lats_search.py`，负责根据 `current_stage` 等状态为搜索注入默认参数（如 `rollouts`、`expand_k`、`lats_llm_soft_top_n` 等）。
- **核心搜索函数**：`app/lats/search.py::lats_search_best_plan`，执行多轮候选扩展（expand_k）、rollout 与早退判断。
- **LLM 软评分器**：`app/lats/evaluator.py::soft_score_via_llm`，采用结构化 JSON 输出，对每个候选从多个维度打分，如：
  - plan_alignment（是否满足任务与 must-have 要求）；
  - style_dim_fit（是否符合 12 维风格目标）；
  - stage_act_fit（是否符合当前 Knapp 阶段的行为预期）；
  - memory_usefulness（是否合理使用与回顾记忆）；
  - assistantiness（"像工具助手"程度，需被压制）。

同时，系统通过**多模型路由**将 LATS 的不同角色映射到不同 LLM：

- `main`：负责 Reasoner / Reply Planner / Processor 等"人格关键"节点；
- `fast`：用于 Detection / Relationship Analyzer / Memory Manager 等对性能与成本更敏感的节点；
- `judge`：专门用于 soft scorer，对稳定 JSON 输出与严格约束更敏感。

LATS 的参数也与 Knapp 关系阶段紧密相连：在 `initiating` / `experimenting` 等早期阶段，系统会设定稍高的 `rollouts` 与更严格的早退 gate，以防止过早收缩到单一开场模式；在 `intensifying` / `integrating` 阶段，则适当降低 rollouts 以节约计算成本；在关系恶化阶段（`differentiating` 等）则采用中等预算，确保在敏感状态下仍能进行足够的多样性搜索。

这一整套 LATS 机制，使得本系统在技术上具有**"从单轮生成到多轮搜索"的逻辑张力**，为研究"搜索预算、soft scorer 维度与关系感知质量之间的关系"提供了丰富的可调参数与日志数据。

---

### 5. 行为编排与拟人化输出：Processor 模块

传统 chatbot 往往直接返回一段文本，而 LTSRChatbot 的 `processor` 节点（`app/nodes/processor.py`）则将输出视为一段**时间轴上的行为脚本**。核心设计包括：

- **分段逻辑（Fragmentation）**：基于大五人格（尤其是 `extraversion`、`conscientiousness`、`neuroticism`）与情绪极端值，计算一个 `fragmentation_tendency`，再转化为 `split_threshold`，决定如何将一段长文本划分为多个"对话气泡"（TCU）。外向、神经质高的人格倾向于更碎片化的表达。
- **打字与阅读延迟模拟**：
  - 设定基础阅读速度与打字速度（如约 20 汉字/秒的阅读速度、300 字/分钟的打字速度），结合每段长度计算延迟；
  - 引入 `MIN_BUBBLE_LENGTH` 等参数防止出现过短气泡。
- **宏观延迟与"离线行为"**：
  - 基于 `busyness` 与 `BOT_SCHEDULE`（如 23:00–7:00 作为睡眠时间），在特定条件下返回 30 分钟到数小时不等的宏观延迟；
  - 在关系恶化阶段（如 `avoiding`、`terminating`）引入较高的 ghosting 概率，使机器人在行为上呈现更接近"真实人类"的疏离与冷处理。

最终，`processor` 输出 `HumanizedOutput`：

- `segments`: 文本片段列表；
- `delay`: 每个片段前的等待秒数；
- `action`: `typing` 或 `idle`，用于客户端控制"对方正在输入…"等非语言线索。

这一模块将**人格（Big Five）、情绪（PAD）、关系阶段（Knapp）、繁忙程度（busyness）**统一投射到可观的行为表现上，使得系统在"看起来像人"这一维度上具备可量化与可控制的基础。

---

### 6. 记忆系统与数据库结构

LTSRChatbot 的记忆系统由 `memory_manager` 与 `memory_writer` 两个节点协同完成，与数据库结构 `bots` / `users` / `messages` / `memories` / `derived_notes` 紧密结合。

#### 6.1 双层记忆架构

- **Memory Store A：Raw Transcript Store**
  - 记录原始对话文本与必要元数据（时间戳、方向、引用的记忆 ID 等），用于后续离线分析与回溯。
- **Memory Store B：Derived Notes Store**
  - 存储由 LLM 从对话中抽取的"稳定事实 / 偏好 / 重要决定 / 关系转折点"等结构化笔记，每条笔记带有 `source_pointer` 便于溯源。

`memory_manager` 节点在每轮对话中使用一次 LLM 调用完成以下任务：

1. 更新 `conversation_summary`（滚动摘要）；
2. 生成新一条 transcript 元数据（用于 Store A）；
3. 抽取 derived notes（用于 Store B）；
4. 从**用户最新输入**中抽取基础信息（name/age/gender/occupation/location），并要求输出证据片段必须逐字出现在 `user_input` 内，Python 端严格校验 evidence 子串，配合置信度阈值与"不覆盖已有字段"的策略，降低"幻觉填表"的风险；
5. 可选抽取新的 `user_inferred_profile` 条目，作为"隐性特征增量"。

#### 6.2 数据库结构与关系

根据 `docs/DATABASE_SCHEMA.md`，数据库中主要表结构如下：

- `bots`：每个 Bot 一行，包含共享 `mood_state` 与 Bot 级 `urgent_tasks` 列。
- `users`：通过 `(bot_id, external_id)` 唯一标识的"某 bot 下的某用户"关系，包含关系相关状态（`current_stage`、`dimensions`、`inferred_profile`、`assets`、`spt_info`、`conversation_summary`）以及 User 级 `urgent_tasks`。
- `messages` / `memories` / `transcripts` / `derived_notes`：均通过 `user_id` 外键挂载到 `users` 上，组织长期原文与抽取笔记。

这种架构使得同一个现实中的"人"在不同 Bot 下可以拥有**各自独立的关系与记忆轨迹**，支撑多 bot / 多 persona 研究场景。

---

### 7. 关系演化引擎与阶段管理

#### 7.1 6 维关系演化：Analyzer + Updater

`app/nodes/evolver.py` 实现了一个**双层关系引擎**：

1. **Analyzer（LLM 层）**：
   - 输入：完整的 `AgentState`（经 `_ensure_relationship_defaults` 处理后的安全状态）、近期对话上下文与当前用户输入；
   - 提示词：由 `src/prompts/relationship::build_analyzer_prompt` 构建，注入了 YAML 中静态信号标准与当前状态；
   - 输出：结构为 `RelationshipAnalysis`，包含：
     - `thought_process`：LLM 对关系变化的思考；
     - `detected_signals`：本轮检测到的关系相关信号；
     - `deltas`：对 6 个维度的变化建议（理论上在 \([-3, +3]\) 整数区间）；
     - `completed_task_ids` / `attempted_task_ids`：与任务系统的对接。

2. **Updater（数学层）**：
   - 首先通过 `_normalize_delta` 将 LLM 的 `deltas` 统一到一个**统一量纲**（例如 \([-0.018, +0.018]\)）；
   - 然后调用 `calculate_damped_delta` 引入**边际收益递减与背叛惩罚**：
     - 当当前得分较高（如 \(\ge 0.9\)）时，正向变化会被大幅缩小（例如乘以 0.1），防止短时间内"冲顶"；
     - 当信任等维度较高时发生负向事件，负向变化可以被适度放大，体现"高处跌落更痛"的背叛效应。
   - 再根据 `current_stage` 选择不同的**阶段倍率**：
     - 对于 `initiating`、`experimenting` 等早期阶段，正向变化倍数较大，使得关系能在几十到数百轮内从陌生走向熟悉；
     - 对于 `integrating`、`bonding` 等稳定阶段，正向变化倍数接近零，使得关系难以进一步显著提升，但仍对重大负向事件敏感。
   - 最后施加每轮每维的**最大步长上限**（`_MAX_STEP_UP` 与 `_MAX_STEP_DOWN`），进一步避免单轮剧烈变化。

此外，`evolver` 还针对"低信息问候"（如纯粹的"你好""晚安"等）引入了 `greeting_gate`：对这种无实质内容的轮次，系统会限制某些维度的正向变化，或者仅对 `closeness`、`trust` 进行极小幅的"启动式提升"，避免"用户只发一句你好，亲密度就明显上升"的不合理现象。

#### 7.2 任务完成检测与会话任务池管理

关系引擎还与任务系统深度耦合：

- 使用 `_get_task_completion_from_analysis` 从 `latest_relationship_analysis` 中解读 `completed_task_ids` 与 `attempted_task_ids`；
- 在 `_detect_completed_tasks_and_replenish` 中：
  - 将本轮完成的任务从 `current_session_tasks` 与 `bot_task_list` 中移除；
  - 为尝试但未完成的任务增加尝试计数与时间戳，有助于后续排序与过期处理；
  - 对 `immediate` 类型任务进行 TTL 递减与清理；
  - 在 `backlog` 池中按照 `BACKLOG_SESSION_TARGET` 补充新的待办；
  - 遵循 `CURRENT_SESSION_TASKS_CAP` 限制当前会话任务池大小；
  - 对 `urgent` 任务做特殊打印与报告，确保系统在日志层面突出这些关键行为。

这一设计将**关系演化、任务完成与后续任务分配**统一在 `evolver` 这一心理节点之中，使得长期陪伴不仅是"闲聊"，而是可以穿插"记得提醒用户某事""持续追踪某个议题"等带有长期目标的行为模式。

#### 7.3 Knapp 阶段管理与 SPT 指标

`app/state.py` 中定义了 `KnappStage` 十个阶段（initiating, experimenting, intensifying, integrating, bonding, differentiating, circumscribing, stagnating, avoiding, terminating），并通过 `stage_manager` 节点（详见 `KNAPP_STAGES.md` 与 `docs/KNAPP_STAGE_REDESIGN_PROPOSAL.md`）综合以下信息更新 `current_stage`：

- 关系数值轨迹（6 维 RelationshipState）；
- 关系资产（话题广度、最大自我暴露深度）；
- `spt_info` 中的 `depth`、`breadth`、`topic_list`、`depth_trend` 等；
- detection 节点给出的 `detection_stage_judge`（对当前阶段与隐含阶段的推断）。

通过这种**二维（阶段 + 维度）关系建模**，系统可以支持如下研究议题：

- 不同 Knapp 阶段下的语言与行为策略差异（由 `style` 与 LATS 决定）；
- 关系维度曲线与阶段跃迁事件之间的定量关系；
- 社会穿透理论中"深度与广度"的变化与长期粘性的关联。

---

## 关键技术实现（面向研究的问题提炼）

在上述结构之上，本系统在若干关键技术点上做了具有研究价值的实现，概括如下。

### 1. 关系演化的数学建模

本项目没有直接用 LLM 的主观描述来更新关系，而是采用了**"LLM 给出离散 delta + Python 侧做连续阻尼与阶段调制"的混合建模**：

- LLM 输出的 `deltas` 在 \([-3, +3]\) 之间，表达"轻微/一般/显著"的正负变化；
- Python 将其映射为小幅连续变化，并叠加：
  - 边际收益递减；
  - 背叛惩罚；
  - 阶段倍率；
  - 每轮最大步长。

这种架构具有双重意义：

- **工程鲁棒性**：即便 LLM 输出偶尔偏离预期（如输出 0.5 或 50），`_normalize_delta` 与 `_clamp` 系列函数仍能把数值收敛到合理区间，避免"爆炸式增长或塌陷"；
- **研究可解释性**：所有参数（如 `DELTA_UNIT`、阶段倍率表、MAX_STEP_UP/DOWN）都是显式常量，可通过日志与实验系统地探索不同参数组合对长期关系曲线的影响。

### 2. 多维输出控制与行为编译

通过 12 维输出驱动 + LATS + Processor 的三层结构，本系统基本实现了从**高层行为策略维度 → 中层语言计划 → 低层行为脚本**的完整路径：

1. `style` 节点根据 `relationship_state`、`mood_state`、`current_stage`、`mode` 等求出 12 维目标；
2. LATS 的 `ReplyPlanner`（`reply_planner.py`）在构造 `ReplyPlan` 时包含：
   - 每条消息的功能（`ReplyMsgFunction`）；
   - 目标长度与信息密度；
   - `pause_after` 与 `delay_bucket`；
   - 覆盖 must-have 要求的映射关系；
3. `processor` 将 `ReplyPlan` 编译为 `ProcessorPlan` 和 `HumanizedOutput`，再结合人格/情绪调整具体分段与延迟。

这一链路为研究问题提供了技术基础，例如：

- 给定同一用户输入，在控制 LATS 搜索预算不变的条件下，调整 12 维目标对用户主观体验（如"亲密感""被理解程度"）的影响；
- 以实验方式比较"仅调整低层分段/延迟"与"同时调整高层策略维度"对长期关系的不同效果。

### 3. LATS 搜索与软评分的可调控性

LATS 的多参数设计（rollouts、expand_k、soft scorer top-n、assistantiness 阈值等）允许研究者做细粒度的**性能-质量权衡实验**：

- 在 `initiating` 阶段增加 rollouts，观察早期关系建构的丰富度变化；
- 调高/调低 `assistantiness_max`，观察"像助手" vs "像朋友"的主观切换点；
- 调整 `lats_gate_pass_rate_min` 与 `final_score_threshold`，研究"门槛严格程度"对失败轮次数与用户挫败感的影响。

同时，多角色 LLM 路由（`main / fast / judge`）提供了**成本维度的实验空间**：在保持 judge 用更稳定模型的前提下，将 fast 与 main 切换为不同供应商或规格，评估对总体体验与关系轨迹的影响。

### 4. 证据约束的记忆抽取机制

`memory_manager` 在抽取基础信息与 derived notes 时，强制要求 LLM 提供来源 evidence，并在 Python 端对 evidence 是否逐字出现在 `user_input` 中进行校验。这一设计在研究问题上具有两点价值：

- **降低幻觉记忆**：避免出现"机器人自认为知道用户年龄/职业，但实际并未在对话中明确出现"的现象，从而更接近人类"基于明确线索形成记忆"的过程。
- **可审计性**：在后续数据分析与论文撰写中，可以直接追踪某一个 profile 字段是在哪一轮、基于哪一段原文被写入的，支持对"记忆形成"进行定性与定量分析。

### 5. 模式系统与安全/防御行为

通过 `PsychoEngine` 与安全检测节点（`security_check`、`boundary`、`sarcasm`、`confusion`、`security_response`），系统在工程上实现了对以下行为的支持：

- 在 CREEPY 场景下，切换到带有边界感、防御性色彩的模式，并通过专门的 `boundary` 节点输出"拒绝+解释+安抚"的综合回复；
- 在 CRAZY 场景下，通过 `confusion` 节点生成"困惑/请对方澄清"的修正回复；
- 在用户把 bot 当作纯工具助手时，通过 assistantiness 指标与 mode 配置提升"关系感表达"的权重。

这些行为模块，为研究"安全与关系感的权衡""边界设置策略对长期信任的影响"等问题提供了可操作的技术基础。

---

## 研究创新点

结合上述技术设计，本项目在学术研究层面具有以下几个突出的创新点：

### 1. 将多种心理学理论一体化编码进可执行架构

与多数仅在提示词中提及心理学概念的应用不同，LTSRChatbot 将以下理论**结构化编码进状态与演化逻辑**中：

- 大五人格：直接体现在 `BotBigFive`，并控制分段倾向、语言风格等行为；
- PAD 情绪模型：通过 `MoodState` 影响回复长度、语气与宏观延迟；
- 6 维关系模型：以连续数值形式驱动风格、任务优先级与 LATS 评估；
- Knapp 十阶段：通过 `current_stage` 与 `stage_manager` 约束阶段内允许的行为与可接受的关系推进速度；
- 社会穿透理论：通过 `relationship_assets` 与 `spt_info` 显式记录话题广度与自我暴露深度，并与阶段管理与风格计算联动。

这种**"从理论到状态变量再到决策规则"的完整落地路径**，为博士论文中关于"理论可计算化（computationalization of theory）"的章节提供了非常具体的例子。

### 2. 双层关系引擎：LLM 分析 + 参数化阻尼

关系演化引擎采用了**"LLM 负责识别信号与离散强度，人手制定公式负责数值更新"**的混合模式：

- LLM 以自然语言与情况推理的优势识别微妙的关系信号；
- Python 以数学公式的形式保证长期曲线的平滑性、约束性与可调试性。

这相比"完全交给 LLM 决定关系数值"的做法，更适合用于长期观察与实验控制，也提高了系统的可解释性与科学性。

### 3. 从单轮生成到多轮搜索的集成设计

LATS 让回复生成从"单次采样"升级为"有约束的搜索过程"，并通过软评分器按多个心理与关系维度综合打分。这一设计提供了如下研究抓手：

- 可以定量比较"简单采样 vs LATS 搜索"在长期对话语料上的差异（如关系维度平滑性、阶段跃迁合理性、用户主观喜爱度）；
- 可以在保持其它条件不变的前提下，只调整 LATS 相关参数，研究不同搜索预算对对话质量与资源消耗的边际效应。

### 4. 行为时间轴与"拟人性"的可微控制

通过 `processor` 输出 `HumanizedOutput` 而非单一文本，系统从根本上支持了"把人类对话当作一条有显性时间坐标的行为轨迹"的视角。在此基础上，可以开展如下研究：

- 比较不同 `fragmentation_tendency`、typing 速度与宏观延迟策略对用户对"它像不像一个真实人"的主观评价；
- 分析不同关系阶段下，真实用户的响应模式（如是否更容忍长延迟），并与系统模拟行为对齐。

### 5. 记忆系统的证据可追踪性与多层结构

双层记忆架构 + 证据校验的基础信息抽取机制，使得：

- 每一个长期记忆点都可以追溯到原始对话片段；
- 研究者可以构建"记忆形成时间线"，用于分析不同类型记忆（如偏好 vs 重要事件）对关系维度与阶段跃迁的影响。

这一点对于**长期陪伴与人格建构研究**非常关键：它使得我们可以问诸如"当机器人在第 N 轮记住了用户最喜欢的歌手之后，其后续 50 轮对话中的亲密度曲线是否有显著提升"等可检验的问题。

---

## 系统评估（设计视角）

目前系统架构本身已经为实证评估提供了良好的技术基础。典型的评估维度可以包括：

- **对话质量**：基于人工标注或 LLM 评估，衡量回复的自然度、一致性与信息量；
- **拟人化程度**：通过用户问卷或对照实验，评估用户对机器人"像人""有个性"的主观感受；
- **关系轨迹合理性**：分析长期对话中 6 维关系数值与 Knapp 阶段的演化曲线，判断是否呈现理论预期的形态（如早期快速增长、中后期趋于平稳，分化期出现回落）；
- **记忆有效性**：检查 derived notes 与真实对话的对应关系，分析机器人在多大程度上利用了长期记忆来个性化回复；
- **安全性与边界行为**：统计 detection 与 security 节点触发次数与后续用户留存情况，评估防御行为的有效性与副作用；
- **性能与资源消耗**：通过 `_profile` 字段与 LangSmith 等监控，记录每个节点耗时与 LLM token 消耗，为实际部署与扩展提供依据。

在论文撰写中，可以基于本架构设计多种 A/B 或多条件实验（如不同 LATS 配置、不同模式策略、不同关系演化参数），系统化讨论技术设计对长期陪伴体验的影响。

---

## 总结

综上所述，LTSRChatbot 项目以 LangGraph 为编排引擎，以高度结构化的 `AgentState` 为核心状态容器，整合了大五人格、PAD 情绪模型、6 维关系模型、Knapp 关系阶段与社会穿透理论等多种心理学理论，并通过 LATS 搜索、关系演化引擎、双层记忆系统与行为编排模块，将这些理论转化为一套**可执行、可调参、可观测的长期陪伴型拟人化 chatbot 技术架构**。

从博士论文的角度，本章节所描述的技术实现可以支撑以下几类后续内容：

- 对不同理论模块（人格、情绪、关系、阶段、记忆）的独立与联合作用做实证分析；
- 针对 LATS 搜索、关系演化公式、行为编排参数等进行系统的参数敏感性分析与用户体验评估；
- 探讨"将人际关系理论计算化"在大模型时代的理论意义与工程挑战。

---

## 参考文献与代码索引

- **项目仓库**：https://github.com/shandelzare7/LTSRChatbot
- **核心代码文件**：
  - `app/state.py`：AgentState 定义
  - `app/graph.py`：LangGraph 节点编排
  - `app/nodes/evolver.py`：关系演化引擎
  - `app/nodes/processor.py`：行为编排模块
  - `app/lats/search.py`：LATS 搜索算法
  - `app/core/engine.py`：心理模式引擎
  - `app/nodes/memory_manager.py`：记忆管理节点
- **文档文件**：
  - `GRAPH_FLOW.md`：节点流程图
  - `docs/LATS_AND_LLM_ROUTING.md`：LATS 与多模型路由说明
  - `docs/DATABASE_SCHEMA.md`：数据库结构说明
  - `KNAPP_STAGES.md`：Knapp 阶段管理说明
