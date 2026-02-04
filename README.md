# EmotionalChatBot V5.0

一个基于 LangGraph 构建的高度拟人化 AI 聊天机器人系统，支持大五人格模型、6维关系模型和 PAD 情绪模型。

## 🌟 核心特性

### 1. 心理建模
- **大五人格模型** (Big Five Personality)：开放性、尽责性、外向性、宜人性、神经质
- **6维关系模型**：亲密、信任、喜爱、尊重、暖意、权力
- **PAD 情绪模型**：愉悦度、唤醒度、掌控感

### 2. 动态人设系统
- 支持运行时动态增删机器人属性、爱好、技能
- 灵活的人设结构（attributes, collections, lore）

### 3. 多种心理模式
- **normal_mode**: 正常模式
- **defensive_mode**: 防御模式
- **stress_mode**: 压力模式
- **broken_mode**: 崩溃模式

### 4. 完整的服务架构
- 数据库服务（用户档案、记忆存储）
- LLM 服务（多模型支持）
- 内存管理（RAG + 摘要）
- 12维输出驱动（策略 + 风格）

## 📁 项目结构

```
EmotionalChatBot_V5/
├── app/
│   ├── core/              # 核心引擎
│   │   ├── database.py    # 数据库接口
│   │   ├── engine.py      # 心理引擎
│   │   └── mode_base.py   # 模式基类
│   ├── nodes/             # LangGraph 节点
│   │   ├── loader.py      # 数据加载
│   │   ├── monitor.py     # 模式监控
│   │   ├── reasoner.py    # 深度推理
│   │   ├── style.py       # 风格计算
│   │   ├── generator.py   # 回复生成
│   │   ├── critic.py      # 质量检查
│   │   ├── processor.py   # 后处理
│   │   └── evolver.py     # 关系演化
│   ├── services/          # 服务层
│   │   ├── db_service.py  # 数据库服务
│   │   ├── llm.py         # LLM 服务
│   │   └── memory/        # 内存管理
│   ├── graph.py           # LangGraph 定义
│   └── state.py           # 状态定义
├── config/
│   ├── modes/             # 心理模式配置
│   │   ├── normal.yaml
│   │   ├── defensive.yaml
│   │   ├── stress.yaml
│   │   └── broken.yaml
│   └── settings.yaml      # 全局配置
├── utils/
│   └── yaml_loader.py     # 配置加载
├── main.py                # 程序入口
└── requirements.txt       # 依赖列表
```

## 🚀 快速开始

### 1. 安装依赖

```bash
cd EmotionalChatBot_V5
pip install -r requirements.txt
```

### 2. 配置环境变量（可选）

创建 `.env` 文件：

```bash
# OpenAI API Key（如果使用 OpenAI）
OPENAI_API_KEY=your_api_key_here

# 数据库配置（如果使用真实数据库）
DATABASE_URL=sqlite:///chatbot.db
```

### 3. 运行示例

```bash
python3 main.py
```

## 🔄 工作流程

```
用户输入
  ↓
[Loader] 加载 Bot/User 档案
  ↓
[Monitor] 监控并切换心理模式
  ↓
[Reasoner] 深度推理用户意图
  ↓
[Style] 计算 12 维输出值
  ↓
[Generator] 生成回复初稿
  ↓
[Critic] 检查质量
  ↓
[Processor] 后处理（分段、延迟）
  ↓
[Evolver] 更新关系和情绪
  ↓
输出最终回复
```

## 📊 State 架构

项目使用高度结构化的 State 定义，包含：

- **Identity Layer**: 机器人身份（基本信息、大五人格、动态人设）
- **Perception Layer**: 用户侧写（显性信息、隐性分析）
- **Physics Layer**: 关系和情绪（6维关系、PAD情绪）
- **Memory Layer**: 记忆系统（短期缓冲、长期摘要、RAG检索）
- **Output Layer**: 输出驱动（12维策略+风格）

详细说明请查看 [`STATE_ARCHITECTURE.md`](EmotionalChatBot_V5/STATE_ARCHITECTURE.md)

## 🎯 12维输出驱动

### Strategy 维度
- `self_disclosure`: 自我暴露程度
- `topic_adherence`: 话题粘性
- `initiative`: 主动性
- `advice_style`: 建议风格
- `subjectivity`: 主观性
- `memory_hook`: 记忆钩子

### Style 维度
- `verbal_length`: 语言长度
- `social_distance`: 社交距离
- `tone_temperature`: 语调温度
- `emotional_display`: 情绪表达
- `wit_and_humor`: 机智幽默
- `non_verbal_cues`: 非语言 cues

## 🔧 配置

### 心理模式配置

编辑 `config/modes/*.yaml` 文件来调整心理模式：

```yaml
id: normal_mode
name: 正常模式
trigger_description: 用户语气平和
system_prompt_template: "你是一个陪伴型 Bot。"
monologue_instruction: "理性分析用户意图。"
critic_criteria:
  - "回复自然"
  - "符合当前关系状态"
split_strategy: normal
typing_speed_multiplier: 1.0
```

## 📚 文档

- [State 架构设计](EmotionalChatBot_V5/STATE_ARCHITECTURE.md) - 详细的状态结构说明
- [GitHub 同步指南](GITHUB_SETUP.md) - 如何同步到 GitHub

## 🛠️ 技术栈

- **LangGraph**: 流程图编排
- **LangChain**: LLM 集成
- **Pydantic**: 数据验证
- **SQLAlchemy**: 数据库 ORM
- **PyYAML**: 配置文件解析

## 📝 开发计划

- [ ] 支持微信接口
- [ ] 向量数据库集成（RAG）
- [ ] 多用户会话管理
- [ ] 关系可视化
- [ ] 情绪历史追踪

## 📄 许可证

MIT License

## 🤝 贡献

欢迎提交 Issue 和 Pull Request！
