"""
Detection 节点：仅对当轮用户消息输出 6 个量。
- hostility_level: 0-10 敌意/攻击性（扣分项）
- engagement_level: 0-10 投入度/字数/信息量（基础动量）
- topic_appeal: 0-10 话题对 Bot 的吸引力（新东西、好玩的东西、或 Bot 喜欢的东西→高；老生常谈/无聊→低；加分项/自我本位）
- stage_pacing: 正常 | 过分亲密 | 过分生疏（关系节奏，用于策略越界判定）
- urgency: 0-10 紧急程度（求助/危机/需即时回应→高；闲聊/可延后→低）
- subtext: 附加给大模型的上帝视角说明
"""
from __future__ import annotations

from typing import Any, Callable, Dict

from langchain_core.messages import HumanMessage, SystemMessage
from utils.tracing import trace_if_enabled
from utils.detailed_logging import log_prompt_and_params, log_llm_response
from utils.llm_json import parse_json_from_llm
from app.lats.prompt_utils import safe_text
from app.state import AgentState
from src.schemas import DetectionOutput

_STAGE_PACING_VALID = frozenset({"正常", "过分亲密", "过分生疏"})


def _clip_int(x: Any, lo: int, hi: int) -> int:
    try:
        v = int(x)
        return max(lo, min(hi, v))
    except (TypeError, ValueError):
        return lo


def _default_detection() -> Dict[str, Any]:
    return {
        "hostility_level": 0,
        "engagement_level": 5,
        "topic_appeal": 5,
        "stage_pacing": "正常",
        "urgency": 5,
        "subtext": "",
    }


def create_detection_node(llm_invoker: Any) -> Callable[[AgentState], dict]:
    """创建 Detection 节点：单次 LLM 产出 5 个字段。"""

    @trace_if_enabled(
        name="Perception/Detection",
        run_type="chain",
        tags=["node", "perception", "detection"],
        metadata={"state_outputs": ["detection"]},
    )
    def detection_node(state: AgentState) -> dict:
        chat_buffer = list(state.get("chat_buffer") or state.get("messages", [])[-30:])
        stage_id = str(state.get("current_stage") or "initiating")

        if not chat_buffer:
            return {"detection": _default_detection()}

        def _is_user(m) -> bool:
            t = getattr(m, "type", "") or ""
            return "human" in t.lower() or "user" in t.lower()

        last_msg = chat_buffer[-1]
        latest_user_text_raw = (
            (state.get("user_input") or "").strip()
            or (getattr(last_msg, "content", "") or str(last_msg)).strip()
        )
        if not _is_user(last_msg):
            latest_user_text_raw = latest_user_text_raw or "(无用户新句)"

        latest_user_text = safe_text(latest_user_text_raw)
        if len(latest_user_text) > 800:
            latest_user_text = latest_user_text[:800]

        system_content = f"""你是高情商的语义理解专家。请仅对「当轮最新用户消息」输出以下 6 个量（格式由系统约束）。

当前关系阶段：{stage_id}（判定 stage_pacing 时请对照此阶段，判断用户这句话是否与该阶段匹配。）

规则：
- 只针对当轮这条用户消息计分，历史仅作语境。
- hostility_level：敌意/攻击/轻蔑/施压越高分数越高。
- engagement_level：字数多、信息量足、接话、追问→高；敷衍、嗯啊哦、想结束→低。
- topic_appeal：新信息；从 Bot 视角看这话题/这句话是否吸引人、想接。新东西、好玩的东西、或 Bot 喜欢的东西→高(7-10)；老生常谈、无聊、与己无关→低(0-3)；一般(4-6)。自我本位。
- stage_pacing：关系节奏，三选一。正常=与当前关系阶段（{stage_id}）匹配、无越界；过分亲密=交浅言深、过早暧昧或过度自我暴露；过分生疏=突然冷淡、回避、敷衍、想结束。询问姓名/年龄/职业等基础信息属正常破冰，填「正常」。
- urgency：紧急程度。求助/崩溃/危险/等回复→高(7-10)；闲聊/随便说说→低(0-3)；正常(4-6)。
- subtext：一句中文，点出潜台词或真实意图，无则空字符串。
"""

        task_msg = HumanMessage(
            content=f"请对下面这句「当轮最新用户消息」按规则输出 6 个量。\n\n当轮最新用户消息：\n{latest_user_text or '(空)'}"
        )
        messages = [SystemMessage(content=system_content), task_msg]

        log_prompt_and_params("Detection", system_prompt=system_content[:800], user_prompt="[当轮用户消息+JSON]", params={})

        out = _default_detection()
        msg = None
        try:
            result = None
            if hasattr(llm_invoker, "with_structured_output"):
                try:
                    structured = llm_invoker.with_structured_output(DetectionOutput)
                    result = structured.invoke(messages)
                except Exception:
                    result = None
            if result is None:
                msg = llm_invoker.invoke(messages)
                content = (getattr(msg, "content", "") or str(msg)).strip()
                raw = parse_json_from_llm(content)
                if isinstance(raw, dict):
                    result = raw
            if result is not None:
                if hasattr(result, "model_dump"):
                    result = result.model_dump()
                elif hasattr(result, "dict"):
                    result = result.dict()
                if isinstance(result, dict):
                    out["hostility_level"] = _clip_int(result.get("hostility_level"), 0, 10)
                    out["engagement_level"] = _clip_int(result.get("engagement_level"), 0, 10)
                    out["topic_appeal"] = _clip_int(result.get("topic_appeal"), 0, 10)
                    sp = str(result.get("stage_pacing") or "正常").strip()
                    out["stage_pacing"] = sp if sp in _STAGE_PACING_VALID else "正常"
                    out["urgency"] = _clip_int(result.get("urgency"), 0, 10)
                    sub = result.get("subtext")
                    out["subtext"] = str(sub).strip() if sub else ""
                    log_llm_response(
                        "Detection",
                        msg if msg is not None else "(structured_output)",
                        parsed_result=out,
                    )
        except Exception as e:
            print(f"[Detection] 解析异常: {e}，使用默认值")

        print(
            f"[Detection] hostility={out['hostility_level']}, engagement={out['engagement_level']}, "
            f"topic_appeal={out['topic_appeal']}, stage_pacing={out['stage_pacing']}, urgency={out['urgency']}"
        )
        return {"detection": out}

    return detection_node
