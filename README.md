# EmotionalChatBot V5.0 (LTSRChatbot)

**GitHub Repository**: https://github.com/shandelzare7/LTSRChatbot

基于 **LangGraph** 的拟人化 AI 聊天系统：以**内心独白**为中枢，结合 **8 种内容策略（Content Moves）**、**6 维写作风格**、**Knapp 关系阶段**与 **6 维关系 / PAD 情绪**模型，产出带分段与打字延迟的拟人回复。

---

## 核心架构

### 流水线概览

```
loader
  └→ safety ──(触发)──→ fast_safety_reply ──────────────────────┐
         │(正常)                                                 │
         ↓                                                       │
  after_safety (并行)                                            │
    ├→ detection                                                │
    └→ state_prep                                               │
         ↓ (汇合 monologue_join)                                 │
  [可选] knowledge_fetcher (当 detection.knowledge_gap 时)       │
         ↓                                                       │
  inner_monologue   ← 核心：产出角色内心活动文本                  │
         ↓                                                       │
  extract           ← 从独白中抽取 move_ids / momentum_delta 等   │
         ↓                                                       │
  state_update ∥ style  (并行：冲量/情绪更新 + 6 维风格计算)      │
         ↓                                                       │
  generate          ← 5 路并行（4 路 move + 1 路 FREE）× n=4 候选 │
         ↓                                                       │
  judge             ← 从约 20 个候选中选出一条「人味」回复        │
         ↓                                                       │
  processor         ← 分段、打字延迟、宏观延迟（睡眠/忙碌）       │
         ↓                                                       ┘
  evolver → stage_manager → memory_manager → memory_writer → END
```

- **入口**：`loader` 加载 Bot/User 档案与对话缓冲。
- **安全**：`safety` 检测敏感/违规；若触发则走 `fast_safety_reply` 直连 `processor`，否则进入主链路。
- **主链路**：`detection`（敌意/投入度/知识缺口等）+ `state_prep` 后，可选 `knowledge_fetcher` 补实时知识，再经 **inner_monologue → extract** 得到本轮的「情绪/态度」与 **Content Move 候选**。
- **生成**：`state_update` 与 `style` 并行更新冲量/情绪和 6 维风格；`generate` 按选中的 move 多路并行生成候选，`judge` 选出一条最符合「人味」的回复。
- **拟人输出**：`processor` 必跑，产出 `humanized_output.segments`（每段含 `content`、`delay`、`action: typing|idle`），供前端按延迟播放气泡。
- **收尾**：`evolver` 更新关系与情绪，`stage_manager` 做阶段门控，`memory_manager` 写摘要与记忆，`memory_writer` 落库。

### 核心特性

| 模块 | 说明 |
|------|------|
| **内心独白 (inner_monologue)** | 根据当前状态、关系、PAD、历史对话生成「角色内心活动」长文本，供 extract 解析与 generate 调节基调。 |
| **Content Moves（8 种）** | 向下细化、向上概括、横向联想、自我暴露、机制溯源、假设推演、状态评价、物理锚定；extract 每轮选 4 个 move，generate 为每个 move 一路 + 1 路 FREE 并行生成。 |
| **6 维写作风格** | FORMALITY / POLITENESS / WARMTH / CERTAINTY / CHAT_MARKERS（五档：极低→极高）+ EXPRESSION_MODE（字面直白 / 字面委婉 / 比喻意象 / 轻调侃）；style 节点按关系与人格算出数值并注入 generate 的 prompt。 |
| **Judge 选回复** | 从多路多候选中按「情景契合、内容新鲜、人味优先、避免写作感」选一条，输出 `final_response`。 |
| **Processor 拟人化** | 对 `final_response` 做分段、每段 `delay` 与 `action`（typing/idle），支持宏观延迟（如睡眠时段）；输出 `humanized_output.segments`，Web 端按序播放。 |
| **心理与关系模型** | **大五人格**（Bot）、**6 维关系**（closeness/trust/liking/respect/attractiveness/power）、**PAD 情绪**（pleasure/arousal/dominance）、**Knapp 关系阶段**（10 阶段，见 `config/stages.yaml`）；evolver 按对话更新关系与情绪，stage_manager 做阶段与 SPT 门控。 |
| **记忆与检索** | conversation_summary、retrieved_memories（RAG）、session_summary；memory_manager 写摘要与任务完成状态，memory_writer 写入持久化存储。 |
| **安全与知识** | safety 检测违规并可选走 fast_safety_reply；detection 可置 `knowledge_gap`，触发 knowledge_fetcher 做外部检索并写入 `retrieved_external_knowledge`。 |

---

## 项目结构

```
EmotionalChatBot_V5/
├── app/
│   ├── graph.py              # LangGraph 编排（节点与边）
│   ├── state.py              # AgentState 与各 TypedDict（Bot/User/关系/情绪/输出）
│   ├── core/                 # 数据库、图用 LLM 配置、人设与关系模板
│   │   ├── database.py
│   │   ├── graph_llm_config.py
│   │   ├── local_store.py
│   │   ├── profile_factory.py
│   │   └── relationship_templates.py
│   ├── nodes/                # 图节点实现
│   │   ├── loader.py         # 加载 Bot/User、缓冲、任务
│   │   ├── safety.py        # 安全检测
│   │   ├── fast_safety_reply.py
│   │   ├── detection.py     # 敌意/投入度/知识缺口等
│   │   ├── state_prep.py    # 状态文本预处理
│   │   ├── inner_monologue.py  # 内心独白生成
│   │   ├── extract.py       # 从独白抽取 move_ids、momentum_delta 等
│   │   ├── state_update.py  # 冲量/情绪更新
│   │   ├── style.py         # 6 维风格计算
│   │   ├── generate.py      # 多路并行生成候选
│   │   ├── judge.py         # 候选筛选
│   │   ├── processor.py     # 分段与延迟（humanized_output）
│   │   ├── evolver.py       # 关系与情绪演化
│   │   ├── stage_manager.py
│   │   ├── memory_manager.py
│   │   ├── memory_writer.py
│   │   ├── knowledge_fetcher.py
│   │   └── ...
│   ├── lats/                 # 提示与风格工具（prompt_utils 等）
│   ├── services/
│   │   ├── llm.py           # 多角色 LLM 与 config/llm_models.yaml
│   │   ├── db_service.py
│   │   └── memory/
│   └── web/
├── config/
│   ├── llm_models.yaml      # 主/快/judge/generate 模型（改模型只改此处）
│   ├── content_moves.yaml   # 8 种 Content Move 定义
│   ├── stages.yaml          # Knapp 10 阶段
│   ├── knapp_rules.yaml
│   ├── relationship_dimensions.yaml
│   ├── momentum_formula.yaml
│   └── ...
├── utils/                    # time_context, yaml_loader, state_to_text 等
├── web_app.py                # FastAPI Web 与 /api/chat（使用 graph + processor 输出）
├── main.py                   # 命令行入口
├── main_gui.py
└── requirements.txt
```

---

## 状态与配置要点

- **AgentState**：包含 `user_input`、`chat_buffer`、`bot_basic_info`/`bot_persona`、`relationship_state`、`mood_state`（PAD + busyness）、`current_stage`、`inner_monologue`、`monologue_extract`、`style`、`generation_candidates`、`final_response`、`humanized_output`、`current_time` 等；各节点按需读写，详见 `app/state.py`。
- **模型配置**：`config/llm_models.yaml` 中 `roles.main/fast/judge` 与 **generate** 单独配置（含 `model`、`base_url`、`temperature`、`top_p`、`n`）；API Key 仍在 `.env`（如 `OPENAI_API_KEY`、`QWEN_API_KEY`）。
- **Content Moves**：`config/content_moves.yaml` 定义 8 个 move 的 `id`、`name`、`content_operation`，由 extract 选 4 个，generate 按 move 描述生成并交 judge 挑选。

---

## 快速开始

```bash
cd EmotionalChatBot_V5
pip install -r requirements.txt
```

配置 `.env`（示例）：

```bash
OPENAI_API_KEY=...
# 使用 Qwen 生成时
QWEN_API_KEY=...
```

- **命令行**：`python main.py`
- **Web**：启动 `web_app.py`（如 `uvicorn web_app:app`），前端通过 `/api/chat` 发送消息，响应中的 `segments` 即 processor 产出的带延迟气泡列表。

---

## 技术栈

- **LangGraph**：图编排与条件边
- **LangChain / LangChain-OpenAI**：LLM 调用
- **Pydantic**：结构化输出与校验
- **SQLAlchemy**：数据库与持久化（可选）
- **PyYAML**：配置加载

---

## 文档与开发

- 状态与类型定义见 `app/state.py` 内注释。
- 图结构与节点职责见 `app/graph.py` 顶部 docstring。
- 同步到 GitHub 见 [GITHUB_SETUP.md](GITHUB_SETUP.md)。

**许可证**: MIT  
**贡献**: 欢迎 Issue 与 Pull Request。
