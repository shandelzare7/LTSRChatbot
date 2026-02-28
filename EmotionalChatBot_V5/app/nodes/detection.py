"""
Detection 节点：对当轮用户消息输出客观量（不依赖角色/Bot 视角）。
- hostility_level: 0-10 敌意/攻击性
- engagement_level: 0-10 投入度/字数/信息量
- stage_pacing: 正常 | 过分亲密 | 过分生疏
- urgency: 0-10 紧急程度
- knowledge_gap: 用户提到了近期事件/具体事实/专有名词，需要外部搜索
- search_keywords: knowledge_gap=True 时的搜索关键词
注：topic_appeal 和 subtext 已移入 extract 节点
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
        "stage_pacing": "正常",
        "urgency": 5,
        "knowledge_gap": False,
        "search_keywords": "",
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

        system_content = f"""你是客观语义分析专家。请仅对「当轮最新用户消息」输出以下 6 个客观量（不依赖 Bot 人设，只看用户行为）。

当前关系阶段：{stage_id}（判定 stage_pacing 时请对照此阶段）

规则：
- 只针对当轮这条用户消息计分，历史仅作语境。
- hostility_level：敌意/攻击/轻蔑/施压越高分数越高。
- engagement_level：字数多、信息量足、接话、追问→高；敷衍、嗯啊哦、想结束→低。
- stage_pacing：关系节奏，三选一。正常=与当前关系阶段（{stage_id}）匹配、无越界；过分亲密=交浅言深、过早暧昧或过度自我暴露；过分生疏=突然冷淡、回避、敷衍、想结束。询问姓名/年龄/职业等基础信息属正常破冰，填「正常」。
- urgency：紧急程度。求助/崩溃/危险/等回复→高(7-10)；闲聊/随便说说→低(0-3)；正常(4-6)。
- knowledge_gap：用户提到了近期事件、具体事实、专有名词（人名/产品/剧集等）或你可能不了解的内容 → true；纯情感倾诉/闲聊/追问感受 → false。
- search_keywords：knowledge_gap=true 时填写最适合搜索的简短关键词（中文，3-8字）；否则填空字符串。
（注：topic_appeal 和 subtext 已由 extract 节点处理，此处不输出）
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
                    sp = str(result.get("stage_pacing") or "正常").strip()
                    out["stage_pacing"] = sp if sp in _STAGE_PACING_VALID else "正常"
                    out["urgency"] = _clip_int(result.get("urgency"), 0, 10)
                    out["knowledge_gap"] = bool(result.get("knowledge_gap", False))
                    out["search_keywords"] = str(result.get("search_keywords") or "").strip()
                    log_llm_response(
                        "Detection",
                        msg if msg is not None else "(structured_output)",
                        parsed_result=out,
                    )
        except Exception as e:
            print(f"[Detection] 解析异常: {e}，使用默认值")

        print(
            f"[Detection] hostility={out['hostility_level']}, engagement={out['engagement_level']}, "
            f"stage_pacing={out['stage_pacing']}, urgency={out['urgency']}"
        )
        return {"detection": out}

    return detection_node
