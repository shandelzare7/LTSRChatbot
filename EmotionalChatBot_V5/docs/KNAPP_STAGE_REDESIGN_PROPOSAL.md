# Knapp 阶段切换条件重设计 — 完整可执行方案

本文档基于项目既有实现（`config/knapp_rules.yaml`、`app/nodes/stage_manager.py`、`app/state.py`）与人际沟通理论，给出以 **6 维关系值 + SPT 深度 + 用户画像充实度** 为核心的 Knapp 阶段切换规则，供直接落地实现。

---

## 一、理论依据与参考文献

### 1.1 Knapp 关系发展模型（Knapp's Relational Development Model）

- **来源**：Mark L. Knapp, *Social Intercourse: From Greeting to Goodbye* (1978)；Knapp, Vangelisti & Caughlin, *Interpersonal Communication and Human Relationships* (7th ed., Pearson).
- **要点**：
  - **Coming Together**（上行）：Initiation → Experimentation → Intensifying → Integration → Bonding。阶段由**沟通行为类型与比例**识别，且“每阶段包含下一阶段的重要前提”，顺序推进更稳。
  - **Coming Apart**（下行）：Differentiating → Circumscribing → Stagnation → Avoidance → Termination。下行可跳过阶段；**背叛/违规会加速恶化**。
  - 判定依据：Welch & Rubin (2002) 通过行为列表让伴侣自评“该行为在关系中的典型程度”，用于识别所处阶段。
- **融入设计**：用 6 维分数近似“行为/态度”的聚合；上行严格顺序、满足“准入+否决”；下行用 max_scores / 条件衰退 / 关键信号触发。

参考链接：
- [Wikipedia: Knapp's Relational Development Model](https://en.wikipedia.org/wiki/Knapp's_Relational_Development_Model)
- [Communication Theory: Knapp's Relationship Model](https://communicationtheory.org/knapps-relationship-model/)

### 1.2 社会穿透理论（Social Penetration Theory, SPT）

- **来源**：Altman & Taylor (1973)。关系由**广度（breadth）**与**深度（depth）**的自我暴露推进。
- **要点**：
  - **Breadth**：话题/生活领域数量（兴趣、家庭、工作等）。
  - **Depth**：从表层事实到情感、秘密的暴露程度。常用 4 层：1=表面/寒暄，2=偏好与轻度情感，3=私密情感与经历，4=核心秘密与承诺。
  - 四阶段：Orientation（表面）→ Exploratory Affective Exchange → Affective Exchange → Stable Exchange（深度、稳定暴露）。
- **融入设计**：上行晋升要求 `min_spt_depth` 与 `min_topic_breadth`；下行用 `depth_trend=decreasing`、`breadth` 收缩触发衰退。

参考链接：
- [Wikipedia: Social penetration theory](https://en.wikipedia.org/wiki/Social_penetration_theory)
- [Communication Theory: Social Penetration](https://www.communicationtheory.org/social-penetration-theory-bringing-people-closer-together/)

### 1.3 不确定性减少理论（Uncertainty Reduction Theory, URT）

- **来源**：Berger & Calabrese (1975)。人们通过沟通减少对对方的认知/行为不确定性；**信息增加 → 不确定性降低 → 关系可预测性增加**。
- **融入设计**：“了解对方程度”用 **user_profile 充实度** 代理：`user_basic_info` + `user_inferred_profile` 的非空字段数。上行时要求一定“了解”再晋升，避免在几乎不了解用户时就进入 intensifying/integrating。

参考链接：
- [Wikipedia: Uncertainty reduction theory](https://en.wikipedia.org/wiki/Uncertainty_reduction_theory)

---

## 二、切换依据总览

| 依据类型 | 数据来源 | 用途 |
|----------|----------|------|
| **6 维关系值** | `relationship_state`（closeness, trust, liking, respect, attractiveness, power），系统内 0–1 | 所有阶段的准入/否决/衰退阈值 |
| **SPT depth** | `spt_info.depth` 或 `relationship_assets.max_spt_depth`，1–4 | 上行：自我暴露深度门槛；下行：depth_trend |
| **SPT breadth** | `spt_info.breadth` 或 `relationship_assets.breadth_score` / topic_history 去重 | 上行：话题广度门槛；下行：限缩触发 |
| **用户画像充实度** | `user_basic_info` + `user_inferred_profile` 非空键数量 | 上行：最低“了解对方”门槛（可选） |
| **recent_signals** | `spt_info.recent_signals` 或 `latest_relationship_analysis.detected_signals` | 上行：required_signals（如 self_disclosure, we_talk）；下行：critical_signals |
| **user_turns** | chat_buffer 中 user/human 条数 | 防 initiating→experimenting 过早（如 ≥3） |

---

## 三、各阶段切换条件表

### 3.1 上行（Coming Together）

所有分数、depth、breadth、profile 均以**当前 step 结束后的状态**为准（evolver 已更新 relationship_state；spt_info 来自 assets + 本轮的 detected_signals）。  
**顺序**：先判 JUMP → 再判 DECAY → 再判 GROWTH → 否则 STAY。

| 当前阶段 | 下一阶段 | 准入条件（up_entry） | 否决条件（up_veto） | 额外说明 |
|----------|----------|----------------------|----------------------|----------|
| **initiating** | experimenting | ① min_scores: closeness≥0.12, trust≥0.08, liking≥0.18<br>② min_spt_depth: 1<br>③ min_topic_breadth: 2<br>④ min_profile_fields: 1（可选） | respect≥0.08 | 且 user_turns ≥ 3 |
| **experimenting** | intensifying | ① min_scores: closeness≥0.40, trust≥0.32, attractiveness≥0.32, liking≥0.28<br>② min_spt_depth: 2<br>③ min_topic_breadth: 3<br>④ required_signals: ["self_disclosure"]<br>⑤ min_profile_fields: 3 | attractiveness≥0.28, respect≥0.12 | — |
| **intensifying** | integrating | ① min_scores: closeness≥0.72, trust≥0.68, liking≥0.68, respect≥0.45<br>② min_spt_depth: 3<br>③ required_signals: ["we_talk"]<br>④ min_profile_fields: 5 | check_power_balance: true, respect≥0.48 | — |
| **integrating** | bonding | ① min_scores: closeness≥0.82, trust≥0.78, liking≥0.75, attractiveness≥0.70, respect≥0.55<br>② min_spt_depth: 3<br>③ required_signals: ["we_talk"] 或 等效承诺信号<br>④ min_profile_fields: 6 | check_power_balance: true, respect≥0.52 | — |
| **bonding** | — | 无 next_up | — | 维持或下行 |

### 3.2 下行（Coming Apart）

**DECAY** 判定顺序：先看 `decay_triggers.max_scores`（任一分 ≤ 阈值即跌入 next_down），再 `conditional_drop`，再 `spt_behavior` / `critical_signals`。

| 当前阶段 | next_down | 衰退条件（decay_triggers） | 说明 |
|----------|-----------|----------------------------|------|
| **initiating** | terminating | max_scores: closeness≤0.0（拉黑/彻底拒绝） | 仅极端情况 |
| **experimenting** | initiating | max_scores: liking≤0.05 或 trust≤0.05 | 不愿继续接触 |
| **intensifying** | experimenting | max_scores: closeness≤0.30 或 trust≤0.28；或 depth_trend=decreasing | 撤回暴露 |
| **integrating** | differentiating | max_scores: trust≤0.50；或 conditional_drop（见下）；或 critical_signals | 信任崩或高亲密低尊重 |
| **bonding** | differentiating | conditional_drop: closeness>0.70 时，若 respect<0.50 或 attractiveness<0.40 或 liking<0.50；或 critical_signals: need_space, contempt | 高亲密但尊重/温情/喜爱不足 |
| **differentiating** | circumscribing | max_scores: trust≤0.45；spt_behavior: depth_reduction | 信任再降或深度回撤 |
| **circumscribing** | stagnating | max_scores: attractiveness≤0.18；spt_behavior: breadth_reduction（breadth≤1） | 冷淡且话题收窄 |
| **stagnating** | avoiding | max_scores: liking≤0.08, respect≤0.08 | 几乎无好感与尊重 |
| **avoiding** | terminating | max_scores: closeness≤0.08；critical_signals: ghosting, block | 物理/心理断联 |
| **terminating** | — | 无 | 终点 |

### 3.3 公式与阈值汇总（0–1 量纲）

- **power_balance**：power 为 Bot 眼中用户强势程度（0–1），`imbalance = |power - 0.5| * 2`；若 `imbalance > power_balance_threshold`（建议 0.3）则 veto 上行。
- **profile_field_count**：  
  `count = len([k for k, v in (user_basic_info or {}).items() if v not in (None, "", [])]) + len([k for k, v in (user_inferred_profile or {}).items() if v not in (None, "", [])])`  
  即：两个字典中“有值”的键数之和。
- **min_profile_fields**：仅在 YAML 的 `up_entry` 中配置；stage_manager 从 state 读 `user_basic_info`、`user_inferred_profile` 并计算 count，与配置比较。

---

## 四、对 `knapp_rules.yaml` 的修改建议

以下为**完整替换** `config/knapp_rules.yaml` 的推荐内容（保留既有结构，仅调整数值与新增 `min_profile_fields`、部分 decay 与 conditional_drop）。

```yaml
# ==============================================================================
# Knapp's Relationship Stage Rules (6-Dim + SPT + Profile)
# Redesign: 切换依据 = 6维 + SPT depth/breadth + user profile 充实度
# ==============================================================================

settings:
  power_balance_threshold: 0.3
  jump_delta_threshold: 0.25

stages:
  # ---------- Coming Together ----------
  initiating:
    order: 1
    next_up: experimenting
    next_down: terminating
    decay_triggers:
      max_scores: { closeness: 0.0 }

  experimenting:
    order: 2
    next_up: intensifying
    next_down: initiating
    up_entry:
      min_scores: { closeness: 0.12, trust: 0.08, liking: 0.18 }
      min_spt_depth: 1
      min_topic_breadth: 2
      min_profile_fields: 1
    up_veto:
      min_scores: { respect: 0.08 }
    decay_triggers:
      max_scores: { liking: 0.05, trust: 0.05 }

  intensifying:
    order: 3
    next_up: integrating
    next_down: experimenting
    up_entry:
      min_scores: { closeness: 0.40, trust: 0.32, attractiveness: 0.32, liking: 0.28 }
      min_spt_depth: 2
      min_topic_breadth: 3
      required_signals: ["self_disclosure"]
      min_profile_fields: 3
    up_veto:
      min_scores: { attractiveness: 0.28, respect: 0.12 }
    decay_triggers:
      max_scores: { closeness: 0.30, trust: 0.28 }
      spt_behavior: "depth_reduction"

  integrating:
    order: 4
    next_up: bonding
    next_down: differentiating
    up_entry:
      min_scores: { closeness: 0.72, trust: 0.68, liking: 0.68, respect: 0.45 }
      min_spt_depth: 3
      required_signals: ["we_talk"]
      min_profile_fields: 5
    up_veto:
      check_power_balance: true
      min_scores: { respect: 0.48 }
    decay_triggers:
      max_scores: { trust: 0.50 }
      spt_behavior: "depth_reduction"
      critical_signals: ["need_space", "contempt"]

  bonding:
    order: 5
    next_up: null
    next_down: differentiating
    decay_triggers:
      conditional_drop:
        condition: "closeness > 0.70"
        triggers: { respect: 0.50, attractiveness: 0.40, liking: 0.50 }
        critical_signals: ["need_space", "contempt"]

  # ---------- Coming Apart ----------
  differentiating:
    order: 6
    next_up: integrating
    next_down: circumscribing
    decay_triggers:
      max_scores: { trust: 0.45 }
      spt_behavior: "depth_reduction"

  circumscribing:
    order: 7
    next_down: stagnating
    decay_triggers:
      max_scores: { attractiveness: 0.18 }
      spt_behavior: "breadth_reduction"

  stagnating:
    order: 8
    next_down: avoiding
    decay_triggers:
      max_scores: { liking: 0.08, respect: 0.08 }

  avoiding:
    order: 9
    next_down: terminating
    decay_triggers:
      max_scores: { closeness: 0.08 }
      critical_signals: ["ghosting", "block"]

  terminating:
    order: 10
```

说明：
- `min_profile_fields` 为可选键；若不存在则不做“了解对方程度”门槛检查。
- `decay_triggers` 中 **max_scores**：任一维度分数 ≤ 该值即触发向 next_down 迁移（与现有逻辑一致）。
- **differentiating** 的 next_up 仍为 integrating，便于关系修复时回到融合阶段。

---

## 五、对 `stage_manager.py` 的修改建议

### 5.1 入参与 State 扩展

- **现有**：`evaluate_transition(current_stage, state)` 通过 `StageManagerInput` 解析 `relationship_state`、`relationship_deltas`/`relationship_deltas_applied`、`spt_info`。
- **新增**：
  - 在传入的 `state` 中提供 `user_basic_info`、`user_inferred_profile`（loader 已写入，无需改 schema 即可用）。
  - 在 **StageManagerInput**（`src/state_schema.py`）中**可选**增加字段，便于校验与默认值（非必须）：  
    `profile_field_count: Optional[int] = None`  
    若 None，则在 manager 内用 `user_basic_info` + `user_inferred_profile` 计算。

### 5.2 计算 profile 字段数

在 `KnappStageManager` 内增加方法（或内联到 `_check_growth`）：

```python
def _count_profile_fields(self, state: Dict[str, Any]) -> int:
    basic = state.get("user_basic_info") or {}
    inferred = state.get("user_inferred_profile") or {}
    def non_empty(d: dict) -> int:
        return sum(1 for k, v in d.items() if v not in (None, "", []))
    return non_empty(basic) + non_empty(inferred)
```

- **数据来源**：`state["user_basic_info"]`、`state["user_inferred_profile"]`（与 loader / DB 一致）。
- **调用时机**：仅在执行 `_check_growth` 且当前阶段的 `up_entry` 中存在 `min_profile_fields` 时调用，与 `min_spt_depth`、`min_topic_breadth` 并列判断。

### 5.3 _check_growth 逻辑（伪代码）

在现有“min_scores → min_spt_depth → min_topic_breadth → required_signals → veto”顺序中，在 **min_topic_breadth** 之后、**required_signals** 之前插入：

```python
# 新增：min_profile_fields（up_entry）
min_profile = int(entry_req.get("min_profile_fields", 0) or 0)
if min_profile > 0:
    profile_count = self._count_profile_fields(state)
    if profile_count < min_profile:
        return None  # 不满足“了解对方程度”
```

其余不变：仍先检查所有 `min_scores`，再检查 `min_spt_depth`、`min_topic_breadth`、`required_signals`，最后做 veto（min_scores + check_power_balance）。

### 5.4 _check_decay 逻辑补充

- **critical_signals**：若 `decay_triggers` 中存在 `critical_signals: [str, ...]`，则检查 `spt.recent_signals` 与列表交集；若有交集则触发 decay（与 conditional_drop 中的 critical_signals 一致，可复用同一段逻辑）。
- **max_scores**：保持现有“任一分 ≤ limit 即返回 next_down”。
- **conditional_drop**：保持现有 `_safe_check_condition(condition, closeness=...)` 后对 sub triggers 逐维比较。
- **spt_behavior**：保持 `depth_reduction` / `breadth_reduction` 的现有实现。

### 5.5 JUMP 逻辑（保留与微调）

- 信任/尊重骤降跳转（trust_delta ≤ -threshold → terminating；respect_delta ≤ -threshold → differentiating）**保留**。
- initiating 下 depth≥3 且 liking>0.4 的“快速亲密”跳 intensifying **可保留**，作为加速路径；若希望更保守，可提高 liking 阈值或去掉该条。

---

## 六、与 user_profile / SPT 的对接方式

### 6.1 user_profile 字段数

| 项目 | 说明 |
|------|------|
| **从哪里读** | `state["user_basic_info"]`、`state["user_inferred_profile"]`。loader 从 DB 的 `user_basic_info` / `user_inferred_profile`（或 local_store 等价字段）加载并写入 state。 |
| **如何计算** | 两字典中“有值”键数之和（值不为 None、""、[]）。见 5.2 节。 |
| **何时使用** | 仅在 **GROWTH** 判定中，且当前阶段 YAML 配置了 `up_entry.min_profile_fields` 时与其它 entry 条件一起判断。 |

### 6.2 SPT depth / breadth

| 项目 | 说明 |
|------|------|
| **depth 从哪里读** | 优先 `state["spt_info"]["depth"]`；若无则 `state["relationship_assets"]["max_spt_depth"]`，缺省 1。stage_manager 的 `_derive_spt_from_assets` 已从 `relationship_assets` 取 `max_spt_depth`。 |
| **breadth 从哪里读** | 优先 `spt_info["breadth"]`；若无则 `relationship_assets["breadth_score"]`，若仍无则 `len(set(relationship_assets["topic_history"]))`。 |
| **如何更新** | 当前代码中 `relationship_assets`（含 topic_history、breadth_score、max_spt_depth）在 save_turn 时由 state 合并进 DB/local_store，**写入方**若尚未实现，可由 Relationship Analyzer 的输出或独立节点根据“本轮话题/暴露深度”更新 `relationship_assets`，再在 save_turn 时持久化；stage_manager 只读不写。 |
| **depth_trend** | 若未在别处计算，可暂时固定为 "stable"；实现时可根据前后轮 depth 比较得到 "increasing" / "decreasing"，供 decay 的 `spt_behavior: depth_reduction` 使用。 |
| **recent_signals** | 从 `latest_relationship_analysis["detected_signals"]` 注入到当轮 `spt_info.recent_signals`（create_stage_manager_node 内已有类似逻辑），供 required_signals / critical_signals 使用。 |

### 6.3 信号名与 YAML 对齐

- **relationship_signals.yaml** 采用 dimensions 结构：每维含 name、definition、anchors（+3 到 -3 的锚点描述）；Analyzer 按 anchors 匹配用户输入并输出整数 deltas，`detected_signals` 可为锚点描述或维度+强度标签。
- **knapp_rules** 中的 `required_signals` / `critical_signals` 建议用**英文简写**（如 `self_disclosure`、`we_talk`、`contempt`、`need_space`、`ghosting`、`block`），便于稳定匹配。
- 实现方式二选一：  
  - 在 relationship 分析阶段让 LLM 同时输出“标准信号标签”列表（self_disclosure, we_talk, …），或  
  - 在 stage_manager 内维护一个**中文/描述 → 标准标签**的映射，将 detected_signals 的文案映射到标准标签再与 required_signals/critical_signals 比较。

---

## 七、实施检查清单

- [ ] 备份并替换 `config/knapp_rules.yaml` 为第四节内容；按需微调阈值。
- [ ] 在 `stage_manager.py` 中实现 `_count_profile_fields(state)`，并在 `_check_growth` 中增加 `min_profile_fields` 判断。
- [ ] 确认 `StageManagerInput` 或调用方传入的 state 包含 `user_basic_info`、`user_inferred_profile`（loader 已提供）。
- [ ] 如需 `profile_field_count` 在 schema 中显式存在，在 `src/state_schema.py` 的 StageManagerInput 中增加可选字段。
- [ ] 在 `_check_decay` 中若尚未支持顶层 `critical_signals`，增加对 `decay_triggers.critical_signals` 与 `spt.recent_signals` 的交集判断。
- [ ] 确认 relationship_assets（topic_history、breadth_score、max_spt_depth）的更新与持久化流程；必要时在 Analyzer 下游或 memory 节点中写入/更新 assets。
- [ ] 统一 required_signals / critical_signals 与 detected_signals 的标签体系（英文简写或映射表）。

完成以上步骤后，Knapp 阶段切换将主要依据 6 维关系值、SPT 深度/广度与用户画像充实度，并与 Knapp、SPT、URT 理论对齐，可直接运行与迭代调参。
