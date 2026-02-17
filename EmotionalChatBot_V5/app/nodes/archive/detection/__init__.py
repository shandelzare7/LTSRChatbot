"""偏离检测相关节点模块（已归档：当前图使用 nodes/detection.py 单文件感知器，此处仅保留供复用）"""
# mute 节点从未实现，已从导出中移除
from app.nodes.archive.detection.boundary import create_boundary_node
from app.nodes.archive.detection.sarcasm import create_sarcasm_node
from app.nodes.archive.detection.confusion import create_confusion_node

__all__ = [
    "create_boundary_node",
    "create_sarcasm_node",
    "create_confusion_node",
]
