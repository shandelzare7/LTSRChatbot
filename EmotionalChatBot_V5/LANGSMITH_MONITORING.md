# 用 LangSmith 监控「思维过程」里的数据（Detection / Reasoner / Style / Generator）

你现在的架构里，节点之间不“传话”，而是通过共享的 `AgentState` 传递信息。  
LangSmith 的最佳用法就是：**把每个节点当成一个 run（span）**，让它在 UI 里呈现“输入 state → 输出 state(增量)”。

本项目已在关键节点上加了 `@traceable(...)`：
- `Perception/Detection`
- `Thinking/Reasoner`
- `Thinking/Styler`
- `Response/Generator`

它们会把以下关键信息写回 state，并出现在 LangSmith trace 里：
- `detection_category` / `detection_result`
- `intuition_thought`
- `inner_monologue` / `response_strategy`
- `llm_instructions`（12维风格）
- `draft_response` / `final_response`

---

## 1) 配置环境变量（推荐用 `.env`）

在 `EmotionalChatBot_V5/.env` 里写（你也可以写到 shell 环境变量）。如果你不确定格式，可以先复制 `EmotionalChatBot_V5/env.example`：

```bash
# LangSmith / LangChain tracing（两套变量都兼容）
LANGCHAIN_TRACING_V2=true
LANGCHAIN_API_KEY=你的_langsmith_key
LANGCHAIN_PROJECT=LTSRChatbot-dev

# 可选：如果你是私有 LangSmith 或自建
# LANGCHAIN_ENDPOINT=https://api.smith.langchain.com

# OpenAI（如果你要用真实模型）
# OPENAI_API_KEY=你的_openai_key
# OPENAI_MODEL=gpt-4o
```

说明：
- **`LANGCHAIN_TRACING_V2=true`** 打开 tracing。
- **`LANGCHAIN_API_KEY`** 是 LangSmith Key（在 LangSmith 里创建）。
- **`LANGCHAIN_PROJECT`** 用来把 runs 分项目归档，方便筛选。

---

## 2) 运行一次流程（产生 trace）

### A. 跑完整 LangGraph（推荐）

```bash
cd EmotionalChatBot_V5
python3 main.py
```

### B. 跑随机 state 的冒烟测试（会跑多次）

```bash
cd EmotionalChatBot_V5
python3 devtools/random_state_smoke_test.py
```

---

## 3) 在 LangSmith UI 里看什么？

打开 LangSmith → 选择 `LANGCHAIN_PROJECT` 对应的 Project，你会看到：
- 一个总 run（代表一次 `invoke`）
- 下面有多个子 run（代表每个节点）

重点关注：
- **Detection 节点**：输出 `intuition_thought` + `detection_category`
- **Reasoner 节点**：输出 `inner_monologue` / `response_strategy`（并写入兼容字段 `deep_reasoning_trace`）
- **Styler 节点**：输出 `llm_instructions`（12维）
- **Generator 节点**：输出 `final_response` / `draft_response`

---

## 4) 安全提醒（非常重要）

LangSmith 会记录 inputs/outputs（包括你传入 state 的内容）。如果你在生产环境里跑：
- **不要把敏感信息（手机号、身份证、地址等）放进 state**  
- 或者在写入 state 前先做脱敏/裁剪（例如只保留最近 N 条消息、隐藏具体数字等）

