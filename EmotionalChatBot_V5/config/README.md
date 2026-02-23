# Config 与代码强绑定说明

本目录下 YAML 由代码直接加载，与代码强绑定；修改 yaml 时需同步检查引用处，否则会导致行为不一致。

| 文件 | 主要消费者 | 说明 |
|------|------------|------|
| **momentum_formula.yaml** | strategy_resolver, utils.yaml_loader.load_momentum_formula_config | 冲量公式（fatigue、EMA 等）；与 STRATEGY_PRIORITY_13 / momentum 五态 配合使用。 |
| **stages.yaml** | stage_manager, utils.yaml_loader.load_stage_by_id, strategy_resolver (STAGE_TO_IDI), requirements | Knapp 阶段元数据；阶段 id 与代码中 stage_index / stage_id 约定一致。 |
| **strategies.yaml** | strategy_resolver (STRATEGY_PRIORITY_13 与 get_strategy_by_id), strategy_routers (HIGH_STAKES_IDS 等) | 策略 id、knapp_stages、min_momentum 等；增删 id 需同步改 strategy_routers 的 id 集合与 strategy_resolver 的 13 级优先级。 |
| **content_moves.yaml** | app.lats.reply_planner.plan_reply_27_via_content_moves, utils.yaml_loader.load_content_moves | LATS 27 候选：需至少 8 条 move，每条含 `tag`、`action`；前 8 路按 tag+action 各生成 3 条，第 9 路为 FREE。应纳入版本控制并与 reply_planner 约定保持一致。 |
| **knapp_rules.yaml** | stage_manager 等 | Knapp 阶段规则与行为约束。 |
| **daily_tasks.yaml** | task_planner / 任务池 | 每日任务池。 |
| **settings.yaml** | 全局配置入口 | 项目级开关与默认值。 |

建议：改上述任一 yaml 后跑相关单测或冒烟；若增删 strategies 的 id，运行时会由 strategy_resolver 的 `_assert_router_ids_in_priority()` 做一次同步校验。
