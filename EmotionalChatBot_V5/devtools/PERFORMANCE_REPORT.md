# 性能分析报告

## 总耗时：21.45 秒

### 🔴 耗时分布（Top 10）

| 节点 | 耗时 | 占比 | 性质 |
|------|------|------|------|
| **generate** | 11.5s | 53.8% | **LLM - 5路×4候选并行生成** |
| **inner_monologue** | 9.0s | 41.9% | **LLM - gpt-4o 长文本生成** |
| judge | 2.7s | 12.6% | LLM - 20个候选评分 |
| memory_manager | 1.8s | 8.4% | LLM - 记忆更新 |
| evolver | 1.5s | 7.2% | LLM - 关系演化 |
| safety | 1.1s | 5.3% | LLM - 安全检测 |
| extract | 1.0s | 4.6% | LLM - 信号提取 |
| monologue_extraction | 0.9s | 4.4% | LLM - 独白提取 |
| processor | 0.9s | 4.3% | LLM - 回复处理 |
| detection | 0.8s | 3.8% | LLM - 客观检测 |

---

## 🎯 关键发现

### 两个超大瓶颈（占 95% 时间）

1. **inner_monologue (9.0s) - 41.9%**
   - 模型：gpt-4o
   - 参数：temp=0.85（为了生成自然独白设的较高温度）
   - 问题：生成 ~500 字左右的长独白
   - 单次 LLM 调用耗时长

2. **generate (11.5s) - 53.8%**
   - 架构：5 路内容动作并行 × 4 候选 = 20 个并发生成
   - 模型：qwen-plus-latest
   - 参数：temp=1.1, top_p=0.75
   - 问题：虽然并行，但 5 路顺序（同步）发起，单路耗时 ~2s
   - 实际是 5 个 2s ≈ 10s 加上网络 overhead

---

## 💡 优化方向

### 立即可做的（低风险）

1. **inner_monologue 长度优化**
   ```
   当前：~500 字（为了内心独白的完整性）
   建议降至：~300 字（仍保留核心内心戏，但降速）
   预期收益：可能省 3-4 秒
   ```

2. **generate 的并行度**
   ```
   当前：5 路顺序发起（实际顺序等待）
   建议：改为真正的 asyncio.gather()
   预期收益：理论上 5 路→1 路耗时（但受网络限制）
   ```

3. **缓存 inner_monologue**
   ```
   如果同一 session 多次生成，可缓存独白（只在关键时刻更新）
   预期收益：20-50%（取决于缓存策略）
   ```

### 长期优化（高风险，需测试）

4. **replace inner_monologue 模型**
   ```
   当前：gpt-4o (最强但最慢)
   替代：gpt-4o-mini (快 3 倍，质量 80%)
   权衡：质量 vs 速度
   ```

5. **批量生成（异步真并行）**
   ```
   当前：5 次 API call
   建议：改为 1 次 batch call（如支持）
   ```

---

## 📊 其他节点耗时

| 节点 | 耗时 | 是否可优化 |
|------|------|-----------|
| judge | 2.7s | ⚠️ 评分就是要费时，很难降 |
| memory_manager | 1.8s | ⚠️ RAG 查询 + 记忆更新，难优化 |
| evolver | 1.5s | ➖ 6D 关系演化，必要 |
| safety | 1.1s | ➖ 安全检测，不能跳 |
| extract | 1.0s | ⚠️ 信号分类，可考虑规则化 |
| monologue_extraction | 0.9s | ⚠️ 可合并到 inner_monologue |
| processor | 0.9s | ➖ 回复处理，轻量级 |
| detection | 0.8s | ➖ 客观检测，数据量小 |
| loader | 0.04s | ✅ 已优化 |

---

## 建议行动方案

### Phase 1（立即，<30min）
- [ ] 降低 inner_monologue 字数从 500 → 300
- [ ] 启用 generate 真正的异步 gather()

### Phase 2（近期，<2h）
- [ ] 对比 gpt-4o 和 gpt-4o-mini 的 inner_monologue 质量
- [ ] 考虑缓存策略（session 级别的独白复用）

### Phase 3（后期，<1d）
- [ ] merge monologue_extraction into inner_monologue 节点
- [ ] 批量 API 调用（如果模型支持）

---

## 数据来源

运行命令：`python devtools/profile_nodes.py`
采样时间：2024-02-26
消息：简单测试 "你好"
总耗时：21.45 秒
