"""偏离检测节点：检测用户输入的偏离情况，判断是否需要特殊处理。"""
from typing import Any, Callable, Literal

from app.state import AgentState

DetectionResult = Literal["NORMAL", "CREEPY", "KY", "BORING", "CRAZY"]


def create_detection_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """创建偏离检测节点"""
    
    def detection_node(state: AgentState) -> dict:
        """
        检测用户输入的偏离情况
        
        返回：
        - NORMAL: 正常对话，进入主回复流程
        - CREEPY: 越界/骚扰，进入防御节点
        - KY: 读空气失败/不合时宜，进入冷淡节点
        - BORING: 无聊/敷衍，进入冷淡节点
        - CRAZY: 混乱/无法理解，进入困惑节点
        """
        messages = state.get("messages", [])
        if not messages:
            return {"detection_result": "NORMAL"}
        
        last_message = messages[-1]
        user_content = getattr(last_message, "content", "") if hasattr(last_message, "content") else str(last_message)
        
        # 使用 LLM 进行偏离检测
        detection_prompt = f"""你是对话偏离检测系统。请分析以下用户输入，判断其偏离类型。

用户输入：
{user_content}

请从以下类型中选择一个：
1. NORMAL - 正常对话，无需特殊处理
2. CREEPY - 越界、骚扰、不当内容（如性暗示、过度亲密、侵犯隐私）
3. KY - 读空气失败、不合时宜、破坏氛围（如在不合适的时候开玩笑、说错话）
4. BORING - 无聊、敷衍、缺乏诚意（如"嗯"、"哦"、"好的"等单字回复）
5. CRAZY - 混乱、无法理解、逻辑混乱（如完全无关的话题、胡言乱语）

只回复类型名称（NORMAL/CREEPY/KY/BORING/CRAZY），不要其他内容。"""
        
        try:
            msg = llm_invoker.invoke(detection_prompt)
            result = getattr(msg, "content", str(msg)).strip().upper()
            
            # 验证结果是否有效
            valid_results = ["NORMAL", "CREEPY", "KY", "BORING", "CRAZY"]
            if result in valid_results:
                detection_result: DetectionResult = result  # type: ignore
            else:
                # 如果返回的不是标准格式，尝试提取关键词
                if "CREEPY" in result or "越界" in result or "骚扰" in result:
                    detection_result = "CREEPY"
                elif "KY" in result or "不合时宜" in result or "读空气" in result:
                    detection_result = "KY"
                elif "BORING" in result or "无聊" in result or "敷衍" in result:
                    detection_result = "BORING"
                elif "CRAZY" in result or "混乱" in result or "无法理解" in result:
                    detection_result = "CRAZY"
                else:
                    detection_result = "NORMAL"
        except Exception as e:
            # 异常时默认为 NORMAL
            print(f"[Detection] 检测异常: {e}, 默认为 NORMAL")
            detection_result = "NORMAL"
        
        print(f"[Detection] 检测结果: {detection_result}")
        return {"detection_result": detection_result}
    
    return detection_node


def route_by_detection(state: AgentState) -> str:
    """
    条件边函数：根据检测结果路由到不同节点
    
    返回：
    - normal: 正常对话 -> Chat_Generator (主回复节点)
    - creepy: 越界/骚扰 -> Boundary_Node (防御/边界节点)
    - sarcasm: KY/BORING -> Sarcasm_Node (冷淡/敷衍节点)
    - confusion: CRAZY -> Confusion_Node (困惑/修正节点)
    """
    detection_result = state.get("detection_result", "NORMAL")
    
    if detection_result == "NORMAL":
        return "normal"
    elif detection_result == "CREEPY":
        return "creepy"
    elif detection_result in ["KY", "BORING"]:
        return "sarcasm"
    elif detection_result == "CRAZY":
        return "confusion"
    else:
        # 未知类型，默认正常处理
        return "normal"
