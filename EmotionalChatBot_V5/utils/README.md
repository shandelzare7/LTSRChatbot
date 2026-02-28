# 工具模块（utils）

项目内共享的工具函数与辅助逻辑，被 `app/`、`src/`、节点及脚本引用。

| 模块 | 用途 |
|------|------|
| `yaml_loader.py` | 加载 config 下 YAML（stages、strategies、content_moves、momentum_formula 等） |
| `state_to_text.py` | 将 AgentState 转为提示词用文本块 |
| `time_context.py` | 当前时间、星期、时段（day_part）等时间上下文 |
| `prompt_helpers.py` | 提示词拼接与格式化 |
| `llm_json.py` | 从 LLM 原始输出解析 JSON |
| `security.py` | 安全检测与敏感词相关 |
| `detailed_logging.py` | 节点/LLM 调用的详细日志 |
| `env_loader.py` | 加载 .env 与项目根路径 |
| `tracing.py` | 可选 tracing 埋点 |
| `busy_schedule.py` | 忙碌/睡眠时段与宏观延迟 |
| `external_text.py` | 外部文本注入与过滤 |
