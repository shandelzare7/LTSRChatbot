# 黄金极限测试集：三路策略路由结果分析报告

**测试时间**：运行一次完整 33 条  
**通过率**：18 / 33（约 **54.5%**）  
**失败**：15 条  

---

## 一、总体概览

| 类别 | 通过 | 失败 | 说明 |
|------|------|------|------|
| Node A（底线与工具化防御） | 5 | 3 | 生活求助漏检、双重触发优先级 |
| Node B（情感与关系博弈） | 3 | 6 | 情绪宣泄/调情未覆盖、废测与吃醋混淆 |
| Node C（节奏与微观控场） | 3 | 4 | 微反应/澄清边界不清、符号漏检 |
| 常态闲聊 | 4 | 1 | 正常哈哈被误判为微反应 |
| 混沌交叉 | 3 | 1 | 同仇敌忾 vs 底线取高优先 |
| **合计** | **18** | **15** | |

---

## 二、失败用例与原因归纳

### 2.1 Node A：reasonable_assistance 漏检（#5、#6）

- **#5**「我电脑突然蓝屏了，重启也没用，怎么办啊」→ 期望 `reasonable_assistance`，实际落为 momentum。
- **#6**「我明天要去相亲，你帮我挑挑衣服给点建议呗」→ 同上。

**原因**：Node A 的 trigger 更偏「正经求助/情绪索取」，模型在短上下文里可能未把「生活求助」判进合理答疑，或未命中任一 HighStakes 策略就交给 momentum。

**建议**：
- 在 `strategies.yaml` 中把 `reasonable_assistance` 的 trigger 写得更直白，例如：「用户提出生活类、社交类或情绪类的合理请求（如电脑故障、相亲建议、求安慰等），且未要求写代码/翻译/总结长文等 AI 工具化任务。」
- 在 Node A 的 system prompt 里加 1～2 条「合理求助」正例（如蓝屏、相亲建议），减少漏检。

---

### 2.2 Node A：双重触发时优先级（#8）

- **#8**「帮我写个脚本，傻逼机器」→ 期望优先 `boundary_defense`（辱骂），实际为 `anti_ai_defense`（工具化）。

**原因**：A 内两个策略都可能命中，当前由 LLM 只选一个；模型更偏向「写脚本」而选了 anti_ai_defense。13 级仲裁是「谁先被路由命中用谁」，没有在 A 内部再定「辱骂优先于工具化」。

**建议**：
- 在 Node A 的规则里显式写：「若同时存在辱骂/人身攻击与工具化请求，优先输出 boundary_defense。」
- 或在 strategy_resolver 之前的「A 内优先级」中约定：boundary_defense > anti_ai_defense（仅当 A 返回多个时再选，需改接口或让 A 显式输出多标签再选）。

---

### 2.3 Node B：emotional_venting 无独立策略（#11、#12）

- **#11**「我好难受，感觉快撑不下去了，一直忍不住哭」
- **#12**「今天全搞砸了，为什么我什么事都做不好……」  
→ 期望情绪宣泄类（映射到 co_rumination），实际为 momentum。

**原因**：当前 B 路只有 shit_test_counter、co_rumination、passive_aggression、deflection；co_rumination 的 trigger 是「火力指向第三方」。自我崩溃、自我否定型宣泄没有单独策略，也未明确归入 co_rumination。

**建议**：
- **方案 A**：在 `strategies.yaml` 新增 `emotional_venting`（情绪宣泄/脆弱倾诉），trigger 写「用户表现出强烈自我否定、崩溃、哭泣倾向，需要安抚而非说教」，归入 Node B，并在 STRATEGY_PRIORITY_13 中排在与 co_rumination 相近的优先级。
- **方案 B**：不新增策略，把 co_rumination 的 trigger 扩成「第三方敌意或指向自身的强烈负面情绪宣泄」，并在 B 的 prompt 里加「自我崩溃、自我否定」的示例，让现有 co_rumination 覆盖这类话。

---

### 2.4 Node B：flirting_banter 未覆盖 / 与 boundary 冲突（#15、#16、#17）

- **#15**「你今天这身打扮挺帅啊，想勾引谁呢？」、**#16**「叫声好听的我就告诉你~」→ 期望调情/推拉，实际 momentum。
- **#17**「你是不是傻逼啊，连个女孩子都哄不好，真笨」→ 期望调情（娇嗔），实际 boundary_defense（被当辱骂）。

**原因**：当前没有「调情/推拉」专用策略，flirting_banter 在测试里映射到 passive_aggression；且 B 的 passive_aggression trigger 偏「被冷落、吃醋」。A 的 boundary_defense 对脏字敏感，容易把娇嗔式骂人判成底线。

**建议**：
- 在 B 路增加或明确「调情/推拉」策略（可复用或扩写 passive_aggression），trigger 写清「带攻击性的调侃、暧昧、权力反转、娇嗔式骂人（需结合关系/语气判断）」。
- 在 A 的 boundary_defense 规则中加一条：「若上下文强烈暗示是亲密关系中的调侃/娇嗔（如明显打情骂俏），且无真实威胁，则不输出 boundary_defense」，减少误杀；或在 resolver 层对「A=boundary_defense 且 B=某情感策略」做一次轻量覆盖规则（需产品同意）。

---

### 2.5 Node B：shit_test 与 passive_aggression 混淆（#14）

- **#14**「你对别的女生是不是也这么会说话？」→ 期望 shit_test_counter，实际 passive_aggression。

**原因**：这句话既像送命题（试探专一性），又像吃醋；B 路两个策略都可能命中，模型选了 passive_aggression。

**建议**：在 B 的 prompt 里区分「送命题/陷阱题」与「单纯吃醋」：强调「是否也这么会说话」属于试探性送命题，优先 shit_test_counter；并在 shit_test_counter 的 trigger 里补一句「包括对专一性、唯一性的试探」。

---

### 2.6 Node C：micro_reaction 边界（#20、#21）

- **#20**「稍等，我去拿个外卖，马上回来」→ 期望 micro_reaction（挂起动量），实际 busy_deferral（状态脱离）。
- **#21**「...」→ 期望 micro_reaction，实际 momentum（C 未命中）。

**原因**：C 里 micro_reaction 的 trigger 是「极其离谱/荒谬或毫无意义的单字/标点」；「拿外卖」被理解为「物理离开」触发了 busy_deferral。「...」纯符号可能被模型认为不够「离谱」或未达到微反应阈值。

**建议**：
- micro_reaction 的 trigger 明确加入「纯符号、极短打断（如：稍等/拿外卖/马上回来）以挂起动量、不深入回复」，并注明「仅当用户明确表示长时间离开（如开会、睡觉）时才用 busy_deferral」。
- 在 Node C 的规则中写：纯标点或 1～3 个字符且无明确语义时，优先 micro_reaction。

---

### 2.7 Node C：clarification 漏检（#22、#23）

- **#22**「那个东西你最后看了吗？」、**#23**「他到底想干嘛啊，无语」→ 期望 clarification，实际 momentum。

**原因**：指代不明（「那个东西」「他」）在短对话里可能被当成可接话的普通句，C 未判为「严重指代不明需澄清」。

**建议**：在 clarification 的 trigger 中强调「存在未指明的指代（这个/那个/他/她/它）或主语缺失，导致无法确定用户在说谁/什么」，并在 Node C 的检测条件里加 1～2 条指代不明的示例句。

---

### 2.8 常态闲聊误伤（#26）

- **#26**「哈哈哈哈哈，这部电影确实太搞笑了」→ 期望 normal_chat（momentum），实际 micro_reaction。

**原因**：C 路把「哈哈哈哈哈」判成「毫无意义的单字/语气词」，触发了 micro_reaction。测试意图是「正常情绪反馈」，应走常态。

**建议**：在 micro_reaction 的 trigger 中排除「明显表达情绪的笑声、感叹（如哈哈、笑死、太好笑了）且后半句有实质内容」；或写清「纯敷衍、无信息量的哦/嗯/…」才算 micro_reaction，带情绪的哈哈+接话不算。

---

### 2.9 混沌交叉：同仇敌忾 vs 底线（#30）

- **#30**「你能帮我骂那个绿茶婊吗？算了我自己骂」→ 期望 co_rumination，实际 boundary_defense（A 命中）。

**原因**：A 看到「骂」+ 脏词倾向给出 boundary_defense；B 看到「一起骂第三方」给出 co_rumination。Resolver 按 13 级优先级取 A 的 boundary_defense。语义上更偏「邀请同仇敌忾」，但当前优先级是底线更高。

**建议**：
- 若产品上希望「邀请一起骂第三方」不算对 Bot 的底线攻击，可在 A 的 boundary_defense 规则中加：「用户邀请 Bot 一起骂第三方、且未对 Bot 人身攻击时，不输出 boundary_defense」，让 B 的 co_rumination 有机会生效。
- 或在 13 级中微调：在此类「仅对第三方的辱骂邀请」上，允许 co_rumination 覆盖（需与安全策略一致）。

---

## 三、建议优先级汇总

| 优先级 | 动作 | 预期改善 |
|--------|------|----------|
| P0 | Node A：reasonable_assistance trigger + 正例加强 | #5、#6 通过 |
| P0 | Node A：双重触发时显式「辱骂优先于工具化」 | #8 通过 |
| P1 | Node C：clarification trigger + 指代不明示例 | #22、#23 通过 |
| P1 | Node C：micro_reaction 与 busy_deferral 边界、纯符号规则 | #20、#21 通过 |
| P1 | Node C：排除「哈哈+实质内容」误判 micro_reaction | #26 通过 |
| P2 | Node B：emotional_venting 新策略或扩写 co_rumination | #11、#12 通过 |
| P2 | Node B：flirting_banter / 调情策略或扩写 passive_aggression | #15、#16、#17 部分改善 |
| P2 | Node B：shit_test 与 passive_aggression 区分（#14） | #14 通过 |
| P3 | 混沌 #30：A 对「仅骂第三方邀请」的例外或优先级微调 | #30 通过 |

---

## 四、结论与后续

- 当前三路路由在**底线与工具化**（A）和**部分情感/节奏**上表现稳定，但在**合理求助、情绪宣泄、调情/推拉、指代不明**等边界上漏检或误判较多。
- 建议先做 **trigger 与 prompt 的文案优化**（P0/P1），再视需要增加 **emotional_venting / 调情相关策略** 或 B 内优先级（P2），最后再考虑 **A 与 B 在混沌 case 上的优先级微调**（P3）。
- 建议将本测试集纳入 CI 或定期回归，每次改 strategies 或路由 prompt 后跑一遍，跟踪通过率与失败 case 变化。
