# 单轮回复的 LLM 调用次数估算

## 结论（默认配置、不早退）

**一次用户消息 → 一次 bot 回复，大约会触发 15～20 次 LLM 请求**（具体取决于是否早退、是否走 processor 的 LLM 拟人化）。

是否「并发限制」导致慢：**主要是「调用次数多 + 绝大部分串行」**，不是我们故意限并发；DeepSeek 侧若有 QPS/RPM 限制，会进一步拉长总时间。

---

## 1. 按节点（主流程顺序）

| 节点 | 角色 | 调用次数 | 说明 |
|------|------|----------|------|
| loader | - | 0 | 只读 DB/状态 |
| detection | fast | **1** | 语境/信号检测 |
| mode_manager | - | 0 | 规则选 mode |
| inner_monologue | main | **1** | 内心独白 |
| emotion_update | - | 0 | 规则 |
| reasoner | main | **1** | 生成 response_plan |
| memory_retriever | - | 0 | 读记忆 |
| style | - | 0 | 当前未用 LLM |
| **lats_search** | main + judge | **见下表** | 根计划 + 多轮 rollout + 评审 |
| processor | main | **0 或 1** | 有 processor_plan 时多为 0，否则可能 1 次拟人化 |
| final_validator | - | 0 | 规则 |
| evolver | fast | **1** | 关系分析 |
| stage_manager | - | 0 | 规则 |
| memory_manager | fast | **1** | 记忆摘要/写入前处理 |
| memory_writer | - | 0 | 写 DB |

固定部分：**detection 1 + inner_monologue 1 + reasoner 1 + evolver 1 + memory_manager 1 = 5**，processor 视情况 +0 或 +1。

---

## 2. LATS 内部（单轮回复内）

默认（initiating/experimenting）：`rollouts=4`，`expand_k=2`，`lats_llm_soft_top_n=1`，`lats_llm_soft_max_concurrency=1`。

| 步骤 | 角色 | 调用次数 | 说明 |
|------|------|----------|------|
| 根计划 | main | **1** | `plan_reply_via_llm` |
| 根评估 | judge | **1** | `evaluate_candidate` → `soft_score_via_llm` |
| 若根早退 | - | 0 | 下面都不跑 |
| **每个 rollout**（共 4 次） | | | |
| └ 扩展变体 | main | **1** | `generate_variants_via_llm`（一次请求出 k 个候选） |
| └ 精评 Top1 | judge | **1** | `evaluate_candidate`（soft scorer） |
| 最终再评一次 best | judge | **1** | 对最终选中的 plan 再跑一次 `evaluate_candidate` |
| （可选）strategy_tag 不足时补一次变体 | main | 0 或 1 | 二次 `generate_variants_via_llm` |

- **不早退**（例如 bot-to-bot 默认关早退）：  
  LATS = 1(根计划) + 1(根评) + 4×(1 变体 + 1 精评) + 1(最终评) = **11**  
  若偶尔触发 strategy_tag 补齐，再 +1。
- **早退**：只做根计划 + 根评 = **2**。

---

## 3. 按角色汇总（不早退、processor 不调 LLM）

| 角色 | 来源 | 次数 |
|------|------|------|
| **main** | inner_monologue, reasoner, LATS 根计划, LATS 每轮 1 次变体 | 1 + 1 + 1 + 4 = **7** |
| **judge** | LATS 根评, 每轮 Top1 精评, 最终再评 | 1 + 4 + 1 = **6** |
| **fast** | detection, evolver, memory_manager | **3** |
| **合计** | | **16** |

若 processor 再打 1 次 main：**17**。  
若早退：LATS 只 2 次，总约 **10**。

---

## 4. 和「并发限制」的关系

- **我们这边**：  
  - 主流程是**串行**的（loader → detection → … → memory_writer）。  
  - 只有 LATS 里对「Top N 候选」做 LLM 精评时用了 `ThreadPoolExecutor`，且默认 `lats_llm_soft_max_concurrency=1`，即**精评也是串行**。  
  - 所以**没有故意把可并发的请求做成串行**，而是流程本身就以串行为主。
- **DeepSeek（或任意厂商）**：  
  - 若对同一 API Key 有 **QPS/RPM 限制**（例如 1 次/秒或 60 次/分钟），16 次请求串行发出时，总时间 ≥ 16 × 单次延迟；若还有排队，会更久。  
  - 所以「并发限制」更可能指：  
    1. **我们单轮调用次数多**（15～20）；  
    2. **这些调用几乎全串行**；  
    3. **厂商对同一 key 的并发/速率限制**（若有）会进一步拉长总时间。

---

## 5. 想加速可以怎么调

- 减少 LATS 调用：`rollouts=2`、`expand_k=1`，或开启早退、提高早退阈值，使更多轮只做根计划+根评。  
- 少用 judge：`lats_llm_soft_top_n=0`（或关闭 soft scorer），精评次数会减少。  
- 确认是否被厂商限流：看 DeepSeek 控制台该 key 的 QPS/RPM 限制，或单测「只打 1 次 main」的延迟，再乘以 16 做下界估算。

---

## 6. LATS 为什么大多串行？哪些已经/可以并行？

**为什么整体是串行的？**

- **主图**：loader → detection → … → lats_search → … 必须按顺序，后一节点依赖前一节点的 state。
- **LATS 内 rollouts**：每一轮 rollout 会「选择叶子 → 扩展 → 回传」并更新树；下一轮的选择依赖当前树状态（visits/value_sum），所以 **rollouts 之间不能简单并行**（除非做并行 MCTS 的 virtual loss，实现复杂）。
- **单次 rollout 内**：先要 `generate_variants_via_llm` 得到候选，才能做 _eval_fast 和 LLM 精评，所以「变体生成」和「评估」是串行的；但「对多个候选做 LLM 精评」可以并行。

**已经实现的并行：**

1. **根评估 + 首轮变体预取**：在根计划生成后，**根评估**和**第一轮 rollout 的变体生成**在 2 个线程里同时跑；首轮扩展 root 时直接复用预取变体，少一次 LLM 往返。
2. **LLM 精评 Top N**：对每个 rollout 里排序后的 Top N 候选做 `evaluate_candidate(..., llm_soft_scorer=...)` 时，用 `ThreadPoolExecutor` 并行（默认 `lats_llm_soft_top_n=1`、`lats_llm_soft_max_concurrency=1`，所以多数情况下仍是 1 个；增大 top_n 和 max_concurrency 即可多路并行）。
3. **助手味检测**：对通过硬门槛的若干候选做 `check_assistant_like_via_llm` 时，多候选**并行**调用（线程池，最多 4 路）。

**配置上如何更并行、更快：**

- 提高 **`lats_llm_soft_top_n`**（如 2～3）：同一 rollout 内对 2～3 个候选做 LLM 精评时会并行。
- 提高 **`lats_llm_soft_max_concurrency`**（如 2～3）：与上面配合，允许同时进行 2～3 次 judge 调用。
- 若 API 有 QPS/RPM 限制，并发太高可能被限流，需在「加速」和「稳定性」之间权衡。
