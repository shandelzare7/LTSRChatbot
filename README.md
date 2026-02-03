# LangGraph Chatbot 流程

这是一个使用 LangGraph 构建的简单 Chatbot 流程示例。

## 功能特性

1. **安全检测** (`safety_check`): 检查用户消息是否包含敏感内容
2. **规划器** (`planner`): 根据对话历史和关系状态生成回复策略
3. **生成器** (`generator`): 根据策略生成最终回复
4. **演化器** (`evolver`): 更新关系统计数据（亲密度等）

## 流程说明

```
起点 -> safety_check -> [条件分支]
                          |
                          ├─ safety_flag=False -> 结束
                          |
                          └─ safety_flag=True -> planner -> generator -> evolver -> 结束
```

## 安装依赖

```bash
pip install -r requirements.txt
```

## 运行示例

### 方法 1: 直接运行（可能显示警告）
```bash
python3 chatbot.py
```

### 方法 2: 使用脚本运行（推荐，无警告）
```bash
./run.sh
```

### 方法 3: 抑制警告运行
```bash
python3 -W ignore chatbot.py
```

## 状态结构 (AgentState)

- `messages`: List[BaseMessage] - 对话消息列表
- `relationship_stats`: dict - 亲密度等关系统计数据
- `safety_flag`: bool - 安全检测结果
- `plan`: str - 思考出的回复策略
- `final_response`: str - 最终回复

## 节点说明

### safety_check
检查最后一条消息是否包含敏感词。如果通过，设置 `safety_flag=True`。

### planner
根据消息数量和当前亲密度生成回复策略，写入 `plan` 字段。

### generator
根据 `plan` 和最后一条消息生成 `final_response`。

### evolver
更新 `relationship_stats`，增加亲密度和对话次数。

## 自定义

所有节点函数都是 mock 实现，你可以根据实际需求替换为：
- 真实的安全检测 API
- LLM 驱动的规划器
- LLM 驱动的生成器
- 数据库存储的关系统计
