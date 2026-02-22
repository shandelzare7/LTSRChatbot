# 已废弃节点（Archive）

本目录存放当前 **graph 未使用** 的节点，仅作保留/参考，不参与构建。

| 文件/目录 | 说明 |
|-----------|------|
| `mode_manager.py` | 原模式管理节点（已从主图移除，路由改由 inner_monologue 的 word_budget / no_reply 处理） |
| `emotion_update.py` | 原情绪更新节点（已从主图移除） |
| `detection/` | 旧版偏离检测子节点（sarcasm / boundary / confusion）；主图现使用 `nodes/detection.py` 单文件感知器 |
| `security_check.py` | 安全检测节点（注入/AI 测试/当助手判断）；主图未使用，devtools 测试见 `devtools/test_detection_security.py` |
| `security_response.py` | 安全响应节点（根据 security_check 生成回复）；主图未使用 |

如需恢复使用，将对应文件移回 `app/nodes/` 并在 `app/graph.py` 中重新挂接即可。
