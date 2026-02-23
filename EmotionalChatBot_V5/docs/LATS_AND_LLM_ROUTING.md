# LATS 性能/拟人质量方案（现状 vs 最终）与多模型路由（OpenAI / DeepSeek）

本文记录两件事：
- **LATS（27 候选 + 单模型评估）**：当前实现的默认行为、以及“平衡版”默认参数。
- **多模型路由（role routing）**：支持 `main / fast / judge` 三角色，并提供快速切换预设（OpenAI / DeepSeek 路线 A / DeepSeek 路线 B）。

> 当前 LATS V3：27 条候选由 **evaluate_27_candidates_single_llm**（judge）一次评估选 best + accept；skip 路径不再做规则检查。Gate1 / evaluate_candidate / hard_gate 已移除。

---

## 1. 现状方案（改动前的默认行为）

### 1.1 LATS 主要调用路径

入口节点：`app/nodes/lats_search.py`  
核心搜索：`app/lats/search.py::lats_search_best_plan`（LATS V3）  
评估：`app/lats/evaluator.py::evaluate_27_candidates_single_llm`（27 条候选一次选 best + accept）

### 1.2 现状默认预算（高层）

在 `app/nodes/lats_search.py` 内，如果 state 未显式覆盖 rollouts/expand_k，会按阶段推断：
- `initiating / experimenting`: **rollouts=8**, expand_k=1
- `intensifying / integrating`: **rollouts=6**, expand_k=2
- 其它：rollouts=2, expand_k=2

此外，LLM soft scorer 会在搜索中对 TopN 候选做精评（默认至少 Top1），且并发默认 2。

### 1.3 现状问题（你遇到的慢）

慢的根因不是单点，而是叠加：
- （历史）rollout 数偏高、expand + 多轮 judge 调用导致慢；当前 V3 已改为 27 候选 + 单次 judge 评估。

---

## 2. 最终方案（平衡版默认参数：拟人质量优先且可控成本）

### 2.1 核心原则

- **judge 评估**：LATS V3 主路径用 **evaluate_27_candidates_single_llm** 一次选 best + accept；skip 路径不再做规则检查。
- **按阶段分配预算**：不同阶段用不同 rollouts/expand_k（若仍使用 rollout 扩展），避免“一刀切”。

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

### 2.3 LATS V3 评估流程

- 主路径：`plan_reply_27_via_content_moves` 生成 27 条候选 → **evaluate_27_candidates_single_llm** 一次选出 best_id 并给出 accept/fail_type/repair/fallback。
- Skip 路径：不再做规则检查；Gate1 / evaluate_candidate / hard_gate 已移除。

---

## 3. 多模型路由：main / fast / judge

### 3.1 为什么要拆角色

- `judge`（LATS 单模型评估）需要最稳的结构化 JSON 输出（最怕格式漂移）。
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
- LATS 评估：`app/lats/evaluator.py::evaluate_27_candidates_single_llm`
- 多模型路由：`app/services/llm.py::get_llm(role=...)`
- Graph wiring：`app/graph.py`（detection/evolver/memory_manager 用 fast，lats_search 用 judge）

