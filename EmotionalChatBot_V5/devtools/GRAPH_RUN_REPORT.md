# Graph 各环节运行报告

按顺序跑一遍 graph（loader → detection → … → memory_writer），验证 state 是否从数据库正确加载，并记录问题。

---

## 1) State 是否能从数据库正确获得 bot 和 user 等信息？

**结论：能。**

- **load_state**（DBManager）返回的字段完整，包含：
  - `bot_basic_info`、`bot_big_five`、`bot_persona`
  - `user_basic_info`、`user_inferred_profile`
  - `relationship_state`（6 维）、`mood_state`、`current_stage`
  - `relationship_assets`、`spt_info`、`conversation_summary`
  - `chat_buffer`（历史消息）
- **loader 节点**在配置了 `DATABASE_URL` 时会调用上述接口，并把上述内容合并进 state，因此 **state 能从数据库正确获得 bot、user、关系、历史等信息**。
- 抽样验证：bot 名称、persona.lore（origin/secret）、persona.collections（hobbies/quirks）均与库中一致。

---

## 2) 按顺序跑 Graph 时发现的问题

### 已发现并处理的问题

| 环节 | 问题 | 处理 |
|------|------|------|
| **reasoner** | LLM 返回内容非纯 JSON（如带 markdown、前后文字），`json.loads` 报错 `Expecting value: line 1 column 2 (char 1)`，导致未写入 `inner_monologue` / `response_strategy`。 | 使用 `utils/llm_json.parse_json_from_llm()` 做稳健解析（支持直接 JSON、markdown 代码块、首尾 `{}` 截取）；解析失败时使用默认 fallback，保证写入 `inner_monologue` 与 `response_strategy`。 |
| **style** | 同上，`json.loads(response.content)` 在 LLM 返回非纯 JSON 时报错。 | 同样改为 `parse_json_from_llm()`，解析失败时使用 12 维默认指令。 |
| **state 合并** | reasoner 写入了 `inner_monologue`、`response_strategy`，但最终 state 中看不到，导致下游 generator 可能拿不到策略。 | 在 `AgentState`（state.py）中显式声明 `inner_monologue`、`response_strategy`；LangGraph 只保留 TypedDict 中声明的键，未声明的会被丢弃。 |

### 运行结果摘要（修复后复跑）

- **loader**：从 DB 正确注入 bot/user/relationship/chat_buffer 等。
- **detection**：输出 `detection_category`（如 NORMAL）及 `intuition_thought`。
- **reasoner**：输出 `inner_monologue`、`response_strategy`、`user_intent`、`mood_state` 更新（解析失败时走 fallback）。
- **style**：输出 `llm_instructions`、`style_analysis`（解析失败时走默认 12 维）。
- **generator**：输出 `draft_response` / `final_response`。
- **critic**：决定 pass/retry，pass 后进入 processor。
- **processor**：输出 `humanized_output`、`final_segments`。
- **evolver**：更新 6 维关系与 deltas。
- **stage_manager**：可能输出 `stage_transition`、`stage_narrative`。
- **memory_writer**：写回 DB/本地。

### 建议后续关注

1. **LLM 输出格式**：若需稳定 JSON，可在 prompt 中明确要求「只输出一行 JSON，不要 markdown 包裹」，或使用模型原生 JSON mode（若支持）。
2. **configurable.llm_model**：reasoner/style/generator 等从 `config["configurable"]["llm_model"]` 取 LLM；当前通过 `create_*_node(llm)` 注入，若以后改为从 config 读取，需保证 run 时传入该键。
3. **bot_id 形式**：loader 用 `state.get("bot_id")` 或 `bot_basic_info.name` 作为 bot 标识；数据库侧建议使用 **UUID** 作为 bot_id，调用图时传入 `bot_id=<uuid>`，避免用 name 与库中 UUID 不一致导致查不到。

---

## 3) 如何复现验证

```bash
cd EmotionalChatBot_V5
# 确保 .env 中 DATABASE_URL 已配置，且库中已有 bot/user（如执行过 reset_and_seed）
.venv/bin/python devtools/graph_run_report.py
```

脚本会：

1. 直接调用 `DBManager.load_state`，检查返回键与抽样（bot name、persona、relationship_state）。
2. 用同一 `user_id` / `bot_id` 执行一次完整 `app.invoke()`，从最终 state 反推各环节是否写入预期字段，并打印问题汇总。

---

*报告由 `devtools/graph_run_report.py` 运行结果整理，reasoner/style 已接入 `utils/llm_json.parse_json_from_llm` 做稳健解析。*
