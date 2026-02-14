# LangGraph 完整流程图

基于 `app/graph.py` 的节点与条件边编排。

## Mermaid 图（可直接在支持 Mermaid 的编辑器中渲染）

```mermaid
flowchart TD
    Start([开始]) --> loader
    loader[loader<br/>加载状态/历史]
    loader --> detection

    detection[detection<br/>感知与直觉]
    detection -->|NORMAL| reasoner
    detection -->|CREEPY| boundary
    detection -->|KY/BORING| sarcasm
    detection -->|CRAZY| confusion

    reasoner[reasoner<br/>深层思考/策略]
    reasoner --> style
    style[style<br/>12维风格指令]
    style --> generator
    generator[generator<br/>生成回复]
    generator --> critic

    critic[critic<br/>质量检查]
    critic -->|pass| processor
    critic -->|retry| refiner
    refiner[refiner<br/>同 generator 精修]
    refiner --> critic

    processor[processor<br/>拟人化调度/拆句]
    processor --> evolver
    evolver[evolver<br/>6维关系演化]
    evolver --> stage_manager
    stage_manager[stage_manager<br/>Knapp 阶段管理]
    stage_manager --> memory_writer
    memory_writer[memory_writer<br/>持久化写入]
    memory_writer --> End([结束])

    boundary[boundary<br/>防御/边界]
    sarcasm[sarcasm<br/>冷淡/敷衍]
    confusion[confusion<br/>困惑/修正]
    boundary --> End
    sarcasm --> End
    confusion --> End
```

## 简化版（仅主路径 + 分支）

```mermaid
flowchart LR
    subgraph 入口与检测
        loader --> detection
    end
    subgraph 正常主链
        detection -->|NORMAL| reasoner --> style --> generator --> critic
        critic -->|pass| processor --> evolver --> stage_manager --> memory_writer --> END
        critic -->|retry| refiner --> critic
    end
    subgraph 异常出口
        detection -->|CREEPY| boundary --> END
        detection -->|KY/BORING| sarcasm --> END
        detection -->|CRAZY| confusion --> END
    end
```

## 节点说明

| 节点名 | 文件 | 作用 |
|--------|------|------|
| **loader** | `nodes/loader.py` | 从 Memory/DB 加载会话状态、历史、关系数据（Load Early） |
| **detection** | `nodes/detection.py` | 感知与直觉：CoT 分类 NORMAL / CREEPY / KY·BORING / CRAZY |
| **reasoner** | `nodes/reasoner.py` | 深层思考：内心独白 + 回复策略 |
| **style** | `nodes/style.py` | 12 维风格指令，供 Generator 使用 |
| **generator** | `nodes/generator.py` | 根据策略与风格生成最终回复文本 |
| **critic** | `nodes/critic.py` | 质量检查：pass 走 processor，retry 走 refiner |
| **refiner** | 同 generator | 精修后再次进入 critic |
| **processor** | `nodes/processor.py` | 拟人化调度：延迟、拆句（TCU）、HumanizedOutput |
| **evolver** | `nodes/evolver.py` | 6 维关系演化：Analyzer + Updater，更新 relationship_state |
| **stage_manager** | `nodes/stage_manager.py` | Knapp 阶段跃迁：JUMP/DECAY/GROWTH/STAY |
| **memory_writer** | `nodes/memory_writer.py` | 持久化写入 DB/本地（Commit Late） |
| **boundary** | `nodes/detection/boundary.py` | CREEPY 时防御/边界回复 |
| **sarcasm** | `nodes/detection/sarcasm.py` | KY/BORING 时冷淡/敷衍回复 |
| **confusion** | `nodes/detection/confusion.py` | CRAZY 时困惑/修正回复 |

## 条件边汇总

- **detection →**  
  `route_by_detection`: **normal** → reasoner；**creepy** → boundary；**sarcasm** → sarcasm；**confusion** → confusion  
- **critic →**  
  `check_critic_result`: **pass** → processor；**retry** → refiner（refiner 再回到 critic）
