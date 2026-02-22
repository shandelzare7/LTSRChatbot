# 定时SPT阶段评估脚本

## 概述

`daily_stage_evaluation.py` 是一个独立的Python脚本，用于每天自动评估所有活跃会话的SPT阶段，并更新数据库中的`current_stage`字段。

## 功能

- 查询最近N天内有对话的用户
- 为每个用户加载完整的状态信息
- 使用`KnappStageManager`评估阶段变迁
- 如果阶段发生变化，自动更新数据库
- 记录详细的评估日志（文件和控制台）

## 使用方法

### 1. 环境变量配置

确保设置以下环境变量：

```bash
export DATABASE_URL="postgresql+asyncpg://user:pass@host:port/dbname"
export DAYS_THRESHOLD=7  # 可选，默认7天
export LOG_LEVEL=INFO    # 可选，默认INFO
```

### 2. 手动运行

```bash
# 从项目根目录运行
cd EmotionalChatBot_V5
python scripts/daily_stage_evaluation.py

# 或使用自定义参数
python scripts/daily_stage_evaluation.py --days 14 --log-level DEBUG
```

### 3. 本地Cron设置（Linux/macOS）

使用提供的安装脚本：

```bash
bash scripts/setup_cron.sh
```

这将添加一个cron任务，每天早上5:00执行评估。

手动设置cron：

```bash
# 编辑crontab
crontab -e

# 添加以下行（根据实际路径调整）
0 5 * * * cd /path/to/EmotionalChatBot_V5 && python3 scripts/daily_stage_evaluation.py >> logs/cron.log 2>&1
```

### 4. Render Cron设置

在Render Dashboard中：

1. 创建新的**Cron Job**服务
2. 连接到与Web服务相同的Git仓库
3. **Build Command**: `pip install -r requirements.txt`
4. **Start Command**: `cd EmotionalChatBot_V5 && python scripts/daily_stage_evaluation.py`
5. **Schedule**: `0 21 * * *` (北京时间早上5点，冬令时) 或 `0 22 * * *` (夏令时)
6. **环境变量**: 添加`DATABASE_URL`（与Web服务相同）

## 命令行参数

- `--days N`: 查询最近N天的对话（默认7天，可通过`DAYS_THRESHOLD`环境变量设置）
- `--log-level LEVEL`: 日志级别（DEBUG/INFO/WARNING/ERROR，默认INFO）
- `--no-file-log`: 不写入日志文件，仅输出到stdout（适用于Render等无持久化磁盘的环境）

## 日志

- **控制台输出**: 所有日志都会输出到stdout/stderr，便于在Render等平台查看
- **文件日志**: `logs/stage_evaluation_YYYY-MM-DD.log`（如果可写）

日志内容包括：
- 评估开始/结束时间
- 处理的用户数量
- 每个用户的评估结果（阶段变更、错误等）
- 摘要统计（总用户数、变更数、错误数）

## 错误处理

- 数据库连接失败：记录错误并退出（退出码1）
- 单个用户评估失败：记录错误但继续处理其他用户
- 状态加载失败：跳过该用户并记录警告
- 如果有任何错误，脚本会以非零退出码退出

## 注意事项

1. 脚本需要能够导入项目模块，确保从项目根目录运行或PYTHONPATH正确设置
2. 数据库连接使用独立的连接池，不依赖FastAPI应用
3. 时区：所有时间比较使用UTC时区
4. Render Cron没有持久化磁盘，建议使用`--no-file-log`或依赖Render的日志捕获功能
