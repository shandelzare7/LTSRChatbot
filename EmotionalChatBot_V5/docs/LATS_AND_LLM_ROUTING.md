# LATS 性能/拟人质量方案（现状 vs 最终）与多模型路由（OpenAI / DeepSeek）

本文记录两件事：
- **LATS（rollout 搜索 + soft scorer）**：当前实现的默认行为是什么、为什么会慢、最终采用的“平衡版”默认参数是什么。
- **多模型路由（role routing）**：支持 `main / fast / judge` 三角色，并提供快速切换预设（OpenAI / DeepSeek 路线 A / DeepSeek 路线 B）。

> 目标约束：**禁止关闭 soft scorer**。本方案只做“减少 soft scorer 调用次数/范围”，不做“禁用 soft scorer”。

---

## 1. 现状方案（改动前的默认行为）

### 1.1 LATS 主要调用路径

入口节点：`app/nodes/lats_search.py`  
核心搜索：`app/lats/search.py::lats_search_best_plan`  
软评分：`app/lats/evaluator.py::soft_score_via_llm`（LLM 结构化 JSON 评分，prompt 很重）

### 1.2 现状默认预算（高层）

在 `app/nodes/lats_search.py` 内，如果 state 未显式覆盖 rollouts/expand_k，会按阶段推断：
- `initiating / experimenting`: **rollouts=8**, expand_k=1
- `intensifying / integrating`: **rollouts=6**, expand_k=2
- 其它：rollouts=2, expand_k=2

此外，LLM soft scorer 会在搜索中对 TopN 候选做精评（默认至少 Top1），且并发默认 2。

### 1.3 现状问题（你遇到的慢）

慢的根因不是单点，而是叠加：
- rollout 数偏高（早期阶段 8 次）
- expand 变体生成 + soft scorer 评分 prompt 都很重
- soft scorer 输出要求包含 plan_alignment_details / style_dim_report / stage_act_report / memory_report，天然慢且耗 token

---

## 2. 最终方案（平衡版默认参数：拟人质量优先且可控成本）

### 2.1 核心原则

- **soft scorer 永远启用**：不允许“完全跳过”。但可以：
  - 只对 Top1 做 soft scorer
  - 降低 soft scorer 并发，换稳定性
- **避免过早早退**：initiating/experimenting 至少跑 1 次 rollout，避免根计划（通用开场白）长期获胜。
- **按阶段分配预算**：不同阶段用不同 rollouts/expand_k，避免“一刀切”。

### 2.2 最终默认预算（按阶段）

由 `app/nodes/lats_search.py` 作为 **默认策略** 注入（可被 state/env 显式覆盖）：

- `initiating / experimenting`
  - rollouts = **4**
  - expand_k = **2**
  - min_rollouts_before_early_exit = **1**
  - early-exit gates（更严格，避免“像助手”也早退）：
    - lats_early_exit_root_score = 0.82
    - lats_early_exit_plan_alignment_min = 0.80
    - lats_early_exit_assistantiness_max = 0.18
    - lats_early_exit_mode_fit_min = 0.65

- `intensifying / integrating`
  - rollouts = **2**
  - expand_k = **1**
  - min_rollouts_before_early_exit = **0**

- `differentiating / circumscribing / stagnating / avoiding / terminating`
  - rollouts = **3**
  - expand_k = **1**
  - min_rollouts_before_early_exit = **0**（若需要更保守可改为 1）

### 2.3 soft scorer 调用范围（不禁用，但降成本）

默认注入：
- `lats_llm_soft_top_n = 1`（只精评 Top1）
- `lats_llm_soft_max_concurrency = 1`（并发=1 更稳，减少限流/抖动）
- `lats_assistant_check_top_n = 0`（额外的助手味检测属于额外 LLM 调用；soft scorer 已有 assistantiness 维度）

### 2.4 关键“质量保险丝”：早退的 gate 必须走 LLM breakdown

`app/lats/search.py` 中已实现：
- 当 soft scorer 可用时，early-exit 以 `llm_plan_alignment / assistantiness / llm_mode_behavior_fit` 为准；
- breakdown 缺字段时保守失败，从而阻止误早退。

---

## 3. 多模型路由：main / fast / judge

### 3.1 为什么要拆角色

- `judge`（soft scorer）需要最稳的结构化 JSON 输出（最怕格式漂移）。
- `fast`（Detection / Relationship Analyzer / Memory Manager）多为分类/抽取/摘要，更适合小模型降成本。
- `main`（Reasoner / ReplyPlanner / Processor）决定“对话像不像这个人”，通常需要更强的模型。

### 3.2 预设与切换

通过环境变量 `LTSR_LLM_PRESET` 选择：
- `openai`
  - main: gpt-4o
  - fast: gpt-4o-mini
  - judge: gpt-4o
- `deepseek_route_a`（推荐：性能×质量最平衡）
  - main: DeepSeek（`https://api.deepseek.com/v1`, `deepseek-chat`）
  - fast: OpenAI（默认 gpt-4o-mini）
  - judge: OpenAI（默认 gpt-4o；可按成本换 mini）
- `deepseek_route_b`
  - main/fast/judge 全部 DeepSeek（单供应商，简单但性能上限更低）

### 3.3 角色级覆盖（最高优先级）

可对单一角色做覆盖：
- `LTSR_LLM_MAIN_API_KEY / BASE_URL / MODEL / TEMPERATURE`
- `LTSR_LLM_FAST_API_KEY / BASE_URL / MODEL / TEMPERATURE`
- `LTSR_LLM_JUDGE_API_KEY / BASE_URL / MODEL / TEMPERATURE`

---

## 4. 代码落点（方便检索）

- LATS 节点默认策略：`app/nodes/lats_search.py`
- LATS 搜索与 early-exit：`app/lats/search.py`
- soft scorer：`app/lats/evaluator.py::soft_score_via_llm`
- 多模型路由：`app/services/llm.py::get_llm(role=...)`
- Graph wiring：`app/graph.py`（detection/evolver/memory_manager 用 fast，lats_search 用 judge）

