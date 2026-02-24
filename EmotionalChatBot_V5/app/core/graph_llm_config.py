"""
由 app/graph.py 在构建图时写入；各节点（如 reply_planner）只读，不得在此或节点内改值。
修改 temperature、top_p 等请只改 graph.py 中对应赋值。
"""
# ReplyPlanner 采样参数（由 graph.build_graph 在创建 llm_planner_27 前设置）
PLANNER_TEMPERATURE: float = 1.2
PLANNER_TOP_P: float = 0.75
PLANNER_FREQUENCY_PENALTY: float = 0.4
PLANNER_PRESENCE_PENALTY: float = 0.5
