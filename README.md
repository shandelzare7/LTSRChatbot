# EmotionalChatBot V5.0 (LTSRChatbot)

**GitHub Repository**: https://github.com/shandelzare7/LTSRChatbot

基于 **LangGraph** 的拟人化 AI 聊天系统。以**内心独白**为决策中枢，结合 **8 种内容策略（Content Moves）**、**6 维写作风格**、**Knapp 关系阶段模型**与 **6 维关系 / PAD 情绪模型**，产出带分段与打字延迟的拟人化回复。

---

## 核心架构

### 流水线概览

```
loader
  └→ safety ──(triggered)──→ fast_safety_reply ────────────┐
         │(normal)                                          │
         ↓                                                  │
  after_safety (并行 fan-out)                               │
    ├→ detection                                           │
    └→ state_prep                                          │
         ↓ (fan-in: monologue_join)                         │
  [可选] knowledge_fetcher   ← knowledge_gap=True 时触发    │
         ↓                                                  │
  inner_monologue            ← 核心：生成角色内心独白文本    │
         ↓                                                  │
  extract                    ← 从独白提取结构化信号         │
    ┌────┴────┐                                             │
    ↓         ↓                                            │
state_update  style          ← 并行：冲量/情绪 + 6维风格    │
    └────┬────┘                                             │
         ↓                                                  │
  generate_join (fan-in)                                    │
         ↓                                                  │
  generate   ← 多路并行（move 路 × n=4 + FREE 路 × n=4）   │
         ↓                                                  │
  judge ◄────────────────────────────────────────────────┘
         ↓
  processor  ← 分段 + 打字延迟 + 宏观延迟
         ↓
  evolver → stage_manager → memory_manager → memory_writer → END
```

---

## 节点详解

### loader
加载 Bot/User 档案（`BotBasicInfo`、`BotBigFive`、`BotPersona`、`UserBasicInfo`）、历史对话缓冲（`chat_buffer`）、关系状态、情绪状态、Knapp 阶段、任务列表（`bot_task_list`、`current_session_tasks`）。从数据库或本地存储中读取，初始化 AgentState。

### safety
检测当轮用户消息是否为注入攻击、AI 测试、或试图把 Bot 当工具使唤。输出 `safety_triggered` 标志与 `safety_strategy_id`。触发时走 `fast_safety_reply` 直连 `processor`，跳过主链路。

### fast_safety_reply
以 Bot 人设生成有边界感的安全回复，而非冷冰冰的拒绝语。走安全路径时直接进入 processor。

### detection
对当轮用户消息做客观量化分析（不依赖 Bot 视角）：

| 输出字段 | 含义 |
|---|---|
| `hostility_level` | 0–10 敌意/攻击性 |
| `engagement_level` | 0–10 投入度/信息量 |
| `stage_pacing` | 正常 / 过分亲密 / 过分生疏 |
| `urgency` | 0–10 紧迫程度 |
| `knowledge_gap` | 是否涉及需要外部检索的近期事实/专有名词 |
| `search_keywords` | knowledge_gap=True 时的搜索关键词 |

### state_prep
将当前状态（关系、情绪、阶段等）格式化为文本（`state_text`），供 inner_monologue 构建 prompt 使用。

### knowledge_fetcher（可选）
当 `detection.knowledge_gap=True` 时触发，执行外部检索，将结果写入 `retrieved_external_knowledge`，供独白和生成节点使用。

### inner_monologue ⭐ 核心节点
以 Bot 完整人设（身份信息 + 大五人格 + 动态人设 + 关系状态 + PAD 情绪 + Knapp 阶段 + 对话历史）生成**第一人称内心活动文本**（≤400 字）。这是整个系统的情感与决策中枢——后续所有节点的行为基调都由此文本驱动。

同步输出 `selected_profile_keys`：从用户画像（`user_inferred_profile`）中选出本轮与独白最相关的键名，供 generate 注入用户侧写信息。

### extract
单次 LLM 调用，从内心独白中提取结构化信号：

| 字段 | 含义 |
|---|---|
| `emotion_tag` | 当前情绪标签（如 心疼 / 烦躁 / 期待 / 无聊） |
| `bot_stance` | 本轮沟通立场：supportive / exploratory / self_sharing / redirecting / challenging |
| `topic_appeal` | 话题吸引力（0–10），影响冲量与风格计算 |
| `selected_profile_keys` | 2–5 个用户画像键（驱动 generate 的画像注入块） |
| `selected_content_move_ids` | 2–4 个 Content Move ID（驱动并行生成路由） |
| `inferred_gender` | 仅在用户性别未知时推断填写 |

Schema 定义见 `src/schemas.py`（`MonologueExtractOutput`）。

### state_update（与 style 并行）
应用 `config/momentum_formula.yaml` 中的公式，根据 detection 信号与 topic_appeal 更新 `conversation_momentum`。同时按 PAD 模型规则更新 `mood_state`（pleasure/arousal/dominance）。

### style（与 state_update 并行）
**纯 Python 计算，无 LLM 调用**。根据 6 维关系向量、PAD 情绪状态、大五人格与 conversation_momentum，计算 6 维写作风格参数，写入 `state.style`：

| 维度 | 含义 | 类型 |
|---|---|---|
| `FORMALITY` | 正式度（书面↔口语网聊）| 0–1 连续值，五档 |
| `POLITENESS` | 礼貌度（措辞软化、面子保护）| 0–1 连续值，五档 |
| `WARMTH` | 温暖度（情感流露阈值）| 0–1 连续值，五档 |
| `CERTAINTY` | 确定度（笃定↔不确定/保留意见）| 0–1 连续值，五档 |
| `EMOTIONAL_INTENSITY` | 情感激活强度（Arousal 的风格出口）| 0–1 连续值，五档 |
| `EXPRESSION_MODE` | 意义编码形式 | 离散四值（0/1/2/3）|

**EXPRESSION_MODE 四路径**：

| 值 | 名称 | 触发条件 |
|---|---|---|
| 0 | 字面直白 | 默认；低 figurative_bias 且无其他路径触发 |
| 1 | 欲言又止 | 关系处于模糊中间带（closeness/trust 在 0.35–0.65）+ 有张力（tension 0.25–0.55）+ 情绪偏负（P≤0.45）；基于 Pinker 2008 的"社会风险管理/plausible deniability"理论 |
| 2 | 比喻/意象 | figurative_bias≥0.60（高开放性 + 话题吸引力 + 情绪激活）|
| 3 | 讽刺/调侃 | 三条路径：A 温暖调侃（关系亲密 + 正面情绪 + irony_propensity 高）；B 烦躁讽刺（P≤0.35 + Ar≥0.65 + EI≥0.55）；C 冷静蔑视（P≤0.30 + Ar≤0.40 + liking≤0.35 + EI≤0.40，对应心理学中的 contempt，同时强制 CERTAINTY≥0.65）|

Guardrails：高尊重（respect≥0.80）强制 EM=3→0；忙碌（busy≥0.80）强制 EM=0；极低信任/亲密时只限制正面路径（P>0.35），不限制负面路径。

### generate ⭐ 生成节点
多路并行生成，每路独立发起 Qwen API 调用（n=4，一次返回 4 个候选）：

- **Move 路**：extract 选出的每个 Content Move 对应一路，prompt 中注入该 move 的内容操作约束
- **FREE 路**：无 move 约束，完全自由生成

共享输入：Bot 人设 + 用户画像注入块（profile_block）+ 当前时间 + 外部知识（如有）+ 内心独白 + 信息密度指令 + 6 维风格参数 + 日常话题素材 + 回复规则。

总候选数 ≤ 20，延迟 ≈ max(单路延迟)。

### judge
从所有候选中选出**与内心独白情绪/态度/意愿最匹配**的一条（而非最流畅或最长）。输出 `final_response` 与 `judge_result`。

### processor
将 `final_response` 拆分为多条气泡，为每条指定：
- `content`：气泡文本
- `delay`：距上一条的等待秒数（模拟真实打字速度）
- `action`：`typing`（显示"正在输入…"）或 `absence`（长离线/策略性沉默）

支持**宏观延迟**门控：睡眠时段、高忙碌度等场景下大幅拉长响应时间（`is_macro_delay=True`）。输出 `HumanizedOutput`。

### evolver
**双层关系演化**：
1. **LLM Analyzer**：读取 `config/relationship_signals.yaml` 信号标准，分析本轮对话对 6 维关系的影响，输出 JSON deltas
2. **Python Updater**：应用阻尼公式（边际收益递减 + 背叛惩罚）将 deltas 应用到 `relationship_state`

同步更新 `mood_state`（情绪演化）与任务完成状态（`completed_task_ids`）。

### stage_manager
基于 `config/stages.yaml` 与 `config/knapp_rules.yaml` 做 Knapp 关系阶段门控，同时处理社会穿透理论（SPT）深度/广度门控，控制 Bot 在不同阶段的行为边界。

### memory_manager
单次 LLM 调用（fast 模型）：
- 更新 `conversation_summary`（滚动摘要）
- 沉淀稳定记忆（`derived_notes`）
- 提取用户信息更新（`user_basic_info`、`user_inferred_profile`）
- 输出 `transcript_meta`（话题/实体/重要性标注）

### memory_writer
将 memory_manager 的输出写入持久化存储（PostgreSQL 或本地 store），包括对话原文、摘要、派生记忆条目。

---

## Content Moves（8 种内容操作）

每轮由 extract 从以下 8 种中选 2–4 种，generate 为每种生成对应路候选：

| ID | 中文名 | 操作约束 |
|----|--------|---------|
| 1 | 向下细化 | 追问或补充对方提到的某个词的具体细节（时间、数量、场景、感受） |
| 2 | 向上概括 | 将对方所说归纳到更大的规律或普遍现象，做一层向上抽象 |
| 3 | 横向联想 | 从对方内容出发，跳到不同领域但结构相似的事物，完成跨域连接 |
| 4 | 自我暴露 | 说出一条关于自己的具体事实（经历/习惯/状态/偏好），明确的自我披露 |
| 5 | 机制溯源 | 给出"为什么会这样"的解释，使用明确的因果逻辑追溯原因 |
| 6 | 假设推演 | 在对方说的基础上改变条件/角度/时间线，推出不同结果或可能性 |
| 7 | 状态评价 | 对对方描述或问题给出简短且具体的评价或判断（可正面或负面） |
| 8 | 物理锚定 | 引入当前空间内的一个感官细节（事物/光线/声音/温度），避免重复 |

---

## 心理与关系模型

### 大五人格（Bot）
`openness`（开放性）、`conscientiousness`（尽责性）、`extraversion`（外向性）、`agreeableness`（宜人性）、`neuroticism`（神经质），范围 [0,1]。用于 style 节点计算风格基线。

### 6 维关系状态
范围 [0,1]，由 evolver 每轮更新：

| 维度 | 含义 |
|---|---|
| `closeness` | 亲密度（陌生→熟人） |
| `trust` | 信任度（防备→依赖） |
| `liking` | 喜爱度（工作伙伴→朋友） |
| `respect` | 尊重度（损友→导师） |
| `attractiveness` | 吸引力（无感→被吸引） |
| `power` | 用户相对主导程度（越高用户越强势） |

### PAD 情绪模型
`pleasure`（愉悦度）、`arousal`（唤醒度）、`dominance`（掌控感），支持 [0,1] 或 [-1,1] 两种尺度（由 `pad_scale` 字段标记）。每轮由 state_update 与 evolver 更新。

### Knapp 关系阶段（10 阶段）
定义在 `config/stages.yaml`，由 stage_manager 门控：

| 阶段 | 中文 |
|---|---|
| initiating → experimenting → intensifying → integrating → bonding | 升级期：陌生→探索→加深→融合→承诺 |
| differentiating → circumscribing → stagnating → avoiding → terminating | 降级期：分化→限缩→停滞→回避→终止 |

---

## 模型配置

配置文件：`config/llm_models.yaml`，API Key 在 `.env`。

| 角色 | 使用节点 | 默认模型 |
|---|---|---|
| `main` | detection、inner_monologue | GPT-4o |
| `fast` | evolver、memory_manager、safety 相关 | GPT-4o-mini |
| `judge` | judge | GPT-4o-mini |
| `generate`（单独配置）| generate | Qwen（`qwen3-next-80b`，阿里云 DashScope）|

generate 节点使用 Qwen，支持 `n=4` 参数一次返回多候选，配置 `temperature=1.0`、`top_p=0.95`、`presence_penalty=0.3`。

---

## 项目结构

```
EmotionalChatBot_V5/
├── app/
│   ├── graph.py                  # LangGraph 编排（节点、边、条件路由）
│   ├── state.py                  # AgentState 及所有 TypedDict 定义
│   ├── core/
│   │   ├── bot/
│   │   │   ├── profile_factory.py      # Bot 档案工厂
│   │   │   ├── relationship_templates.py
│   │   │   └── bot_creation_llm.py
│   │   ├── db/
│   │   │   ├── database.py             # SQLAlchemy 数据库连接
│   │   │   └── local_store.py          # 本地文件存储（开发用）
│   │   └── graph_llm_config.py
│   ├── nodes/
│   │   ├── pipeline/             # 主流水线节点
│   │   │   ├── loader.py
│   │   │   ├── detection.py
│   │   │   ├── state_prep.py
│   │   │   ├── inner_monologue.py
│   │   │   ├── extract.py
│   │   │   ├── state_update.py
│   │   │   ├── style.py
│   │   │   ├── generate.py
│   │   │   ├── judge.py
│   │   │   ├── processor.py
│   │   │   └── absence_gate.py   # 宏观离线/忙碌门控
│   │   ├── relation/
│   │   │   ├── evolver.py        # 6维关系演化（LLM Analyzer + Python Updater）
│   │   │   └── stage_manager.py  # Knapp 阶段与 SPT 门控
│   │   ├── memory/
│   │   │   ├── knowledge_fetcher.py
│   │   │   ├── memory_manager.py
│   │   │   └── memory_writer.py
│   │   └── safety/
│   │       ├── safety.py
│   │       └── fast_safety_reply.py
│   ├── prompts/
│   │   └── prompt_utils.py       # 风格参数格式化注入、safe_text 等工具函数
│   ├── services/
│   │   ├── llm.py                # 多角色 LLM 路由（get_llm(role=...)）
│   │   ├── db_service.py
│   │   └── memory/               # 记忆服务抽象层
│   └── web/
│       └── session.py
├── src/
│   └── schemas.py                # Pydantic 结构化输出 Schema
│                                 # MonologueExtractOutput / DetectionOutput / JudgeOutput 等
├── config/
│   ├── llm_models.yaml           # 模型配置（只改此处，不改代码）
│   ├── content_moves.yaml        # 8 种 Content Move 定义
│   ├── stages.yaml               # Knapp 10 阶段元数据
│   ├── knapp_rules.yaml          # 阶段晋升/降级触发规则
│   ├── relationship_dimensions.yaml  # 6 维关系维度定义
│   ├── relationship_signals.yaml     # evolver 信号判断标准
│   ├── momentum_formula.yaml         # conversation_momentum 更新公式
│   ├── daily_topics.yaml             # 日常话题素材库（注入 generate）
│   ├── daily_tasks.yaml              # Bot 日常任务配置
│   ├── strategies.yaml               # 对话策略配置
│   ├── PERSONA_PRESETS.yaml          # Bot 人设预设模板
│   └── settings.yaml                 # 全局开关与阈值
├── utils/                        # 工具模块（time_context / yaml_loader / tracing 等）
├── docs/
│   ├── LLM_CALLS_REFERENCE.md    # 所有 LLM 调用的输入/输出/人设汇总
│   └── DATABASE_SCHEMA.md
├── web_app.py                    # FastAPI 入口（/api/chat 等路由）
├── main.py                       # 命令行交互入口
├── main_gui.py                   # GUI 入口
└── requirements.txt
```

---

## 状态关键字段（AgentState）

| 字段 | 写入节点 | 含义 |
|---|---|---|
| `user_input` | loader | 当轮用户消息 |
| `chat_buffer` | loader | 历史对话消息列表 |
| `bot_basic_info` / `bot_persona` / `bot_big_five` | loader | Bot 静态档案 |
| `relationship_state` | loader / evolver | 6 维关系向量 |
| `mood_state` | loader / state_update / evolver | PAD 情绪 + busyness |
| `current_stage` | loader / stage_manager | 当前 Knapp 阶段 |
| `inner_monologue` | inner_monologue | Bot 内心独白文本 |
| `monologue_extract` | extract | 结构化提取信号（情绪/立场/move 等）|
| `conversation_momentum` | state_update | 对话动能（0–1）|
| `style` | style | 6 维风格参数 dict |
| `generation_candidates` | generate | 所有路候选文本列表 |
| `final_response` | judge / processor | 最终回复文本 |
| `humanized_output` | processor | 分段气泡列表（含 delay/action）|

---

## 快速开始

```bash
cd EmotionalChatBot_V5
pip install -r requirements.txt
```

配置 `.env`：

```bash
# 主链路（detection / inner_monologue / judge 等）
OPENAI_API_KEY=...
OPENAI_BASE_URL=https://api.openai.com/v1
OPENAI_MODEL=gpt-4o

# 生成节点（generate）使用阿里云 Qwen
QWEN_API_KEY=...

# 数据库（可选，不配置则使用本地文件存储）
DATABASE_URL=postgresql+asyncpg://user:pass@host/dbname
```

- **命令行**：`python main.py`
- **Web 服务**：`uvicorn web_app:app --host 0.0.0.0 --port 8000`
  - `POST /api/chat`：发送消息，返回 `segments`（分段气泡列表，含 content/delay/action）

---

## 技术栈

| 组件 | 用途 |
|---|---|
| **LangGraph** | 图编排、条件边、并行 fan-out/fan-in |
| **LangChain / LangChain-OpenAI** | LLM 调用封装 |
| **Pydantic** | 结构化输出 Schema 与校验 |
| **SQLAlchemy + asyncpg** | 数据库持久化（PostgreSQL，可选）|
| **PyYAML** | 配置文件加载 |
| **FastAPI / Uvicorn** | Web API 服务 |

---

**许可证**: MIT
**贡献**: 欢迎 Issue 与 Pull Request。
