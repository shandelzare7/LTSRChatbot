# Archive

已废弃或不再在主流程中使用的配置与实现，仅保留供参考或回滚。

## modes/

**Mode 行为策略（心理模式）**：原由 `config/modes/*.yaml` 定义、经 `PsychoEngine` 与 `mode_manager` 节点做 LLM 侧写并写入 `current_mode`。主图已不再使用心理模式切换，`current_mode` 仅由 main 初始化为内置的 normal_mode 默认对象。本目录已移至此处归档。
