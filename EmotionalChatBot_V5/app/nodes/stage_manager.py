"""
stage_manager.py
Knapp 阶段管理器节点：在 evolver 之后执行阶段迁移判定。

输入依赖（来自 AgentState）：
- current_stage
- relationship_state（6维）
- relationship_deltas / relationship_deltas_applied（用于 jump）
- spt_info（可选；缺失时将从 relationship_assets 推导一部分）

输出：
- current_stage（可能更新）
- stage_narrative（原因）
- stage_transition（结构化记录）
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

from app.state import AgentState
from src.state_schema import StageManagerInput
from utils.yaml_loader import get_project_root

logger = logging.getLogger(__name__)


def _safe_check_condition(condition: str, *, closeness: float) -> bool:
    """
    仅支持极简条件：形如 'closeness > 0.7' / 'closeness >= 0.7' 等。
    避免使用 eval。
    注意：closeness 参数和阈值都是 0-1 范围。
    """
    s = (condition or "").strip()
    closeness_val = max(0.0, min(1.0, float(closeness)))
    
    for op in (">=", "<=", "==", "!=", ">", "<"):
        if op in s:
            left, right = [x.strip() for x in s.split(op, 1)]
            if left != "closeness":
                return False
            try:
                threshold = max(0.0, min(1.0, float(right)))
            except Exception:
                return False
            if op == ">=":
                return closeness_val >= threshold
            if op == "<=":
                return closeness_val <= threshold
            if op == "==":
                return abs(closeness_val - threshold) < 0.01  # 浮点比较容差
            if op == "!=":
                return abs(closeness_val - threshold) >= 0.01
            if op == ">":
                return closeness_val > threshold
            if op == "<":
                return closeness_val < threshold
    return False


class KnappStageManager:
    def __init__(self, config_path: Optional[str] = None):
        if config_path is None:
            root = get_project_root()
            config_path = str(Path(root) / "config" / "knapp_rules.yaml")
        self.config_path = config_path
        self.config = self._load_config(config_path)
        self.settings = self.config.get("settings", {}) or {}
        self.stages = self.config.get("stages", {}) or {}

    def _load_config(self, path: str) -> Dict[str, Any]:
        with open(path, "r", encoding="utf-8") as f:
            return yaml.safe_load(f) or {}

    def evaluate_transition(self, current_stage: str, state: Dict[str, Any]) -> Dict[str, Any]:
        """
        Main Entry Point: 决定是否变迁
        Returns:
          { "new_stage": str, "reason": str, "transition_type": "JUMP"|"DECAY"|"GROWTH"|"STAY" }
        """
        parsed = StageManagerInput.model_validate(state)  # pydantic v2
        scores = parsed.relationship_state.model_dump()
        user_turns = self._count_user_turns(state)

        # deltas：优先用 applied；否则用 raw
        deltas_applied = dict(parsed.relationship_deltas_applied or {})
        deltas_raw = dict(parsed.relationship_deltas or {})

        # SPT：缺失则尝试从 relationship_assets 推导 breadth/depth
        spt = parsed.spt_info.model_dump() if parsed.spt_info else self._derive_spt_from_assets(state)

        jump = self._check_jumps(current_stage, scores, deltas_applied or deltas_raw, spt)
        if jump:
            return jump

        decay = self._check_decay(current_stage, scores, spt)
        if decay:
            return decay

        growth = self._check_growth(current_stage, scores, spt, user_turns=user_turns)
        if growth:
            return growth

        return {"new_stage": current_stage, "reason": "Stable state.", "transition_type": "STAY"}

    def _normalize_delta_points(self, x: Any) -> float:
        """
        兼容本项目 delta 标度：
        - 小范围 [-3..3] / [-5..5]：当作“强度”，映射到点数 *10（最大 30/50）
        - 更大值（如 25）：当作点数，保持原样
        """
        try:
            v = float(x)
        except Exception:
            return 0.0
        # 1) 若已是 0-1 量纲，直接返回
        if abs(v) <= 1.0:
            return v
        # 2) 若是强度档位（-3..3 / -5..5），映射到 0-1（例如 3 -> 0.3）
        if abs(v) <= 5.0:
            return v / 10.0
        # 3) 若是旧 points（0-100），归一化
        if abs(v) <= 100.0:
            return v / 100.0
        # 4) 兜底：极端值截断到 [-1, 1]
        return max(-1.0, min(1.0, v))

    def _count_user_turns(self, state: Dict[str, Any]) -> int:
        """
        估算用户轮次：优先从 chat_buffer 中统计 human/user 消息条数。
        目的：避免 initiating->experimenting 过早升级（至少 3 轮用户输入）。
        """
        buf = state.get("chat_buffer") or []
        n = 0
        for m in buf:
            t = getattr(m, "type", "") or ""
            if "human" in str(t).lower() or "user" in str(t).lower():
                n += 1
        # 若本轮 user_input 尚未进入 buffer，也算一轮
        if str(state.get("user_input") or "").strip():
            n = max(n, 1)
        return int(n)

    def _check_jumps(
        self,
        current_stage: str,
        scores: Dict[str, float],
        deltas: Dict[str, Any],
        spt: Dict[str, Any],
    ) -> Optional[Dict[str, Any]]:
        # 阈值（0-1 范围）
        threshold = max(0.0, min(1.0, float(self.settings.get("jump_delta_threshold", 0.25))))
        
        trust_delta = self._normalize_delta_points(deltas.get("trust", 0))
        respect_delta = self._normalize_delta_points(deltas.get("respect", 0))

        # 监控 JUMP 检查过程
        print(f"[MONITOR] stage_jump_check: current_stage={current_stage}, threshold={threshold:.3f}, trust_delta={trust_delta:.3f}, respect_delta={respect_delta:.3f}")

        if trust_delta <= -threshold:
            print(f"[MONITOR] stage_jump_triggered: trust_delta={trust_delta:.3f} <= -threshold={-threshold:.3f}")
            return {
                "new_stage": "terminating",
                "reason": f"Catastrophic trust failure (Event Driven). trust_delta={trust_delta:.3f}",
                "transition_type": "JUMP",
            }
        if respect_delta <= -threshold:
            print(f"[MONITOR] stage_jump_triggered: respect_delta={respect_delta:.3f} <= -threshold={-threshold:.3f}")
            return {
                "new_stage": "differentiating",
                "reason": f"Sudden loss of respect. respect_delta={respect_delta:.3f}",
                "transition_type": "JUMP",
            }

        if current_stage == "initiating" and int(spt.get("depth", 1) or 1) >= 3:
            liking_score = max(0.0, min(1.0, float(scores.get("liking", 0.0) or 0.0)))
            print(f"[MONITOR] stage_jump_check_rapid_intimacy: depth={spt.get('depth', 1)}, liking_score={liking_score:.3f}")
            if liking_score > 0.4:
                print(f"[MONITOR] stage_jump_triggered: rapid_intimacy_acceleration, depth={spt.get('depth', 1)}, liking_score={liking_score:.3f}")
                return {
                    "new_stage": "intensifying",
                    "reason": f"Rapid intimacy acceleration. depth={spt.get('depth', 1)}, liking={liking_score:.3f}",
                    "transition_type": "JUMP",
                }
        return None

    def _check_decay(
        self, current_stage: str, scores: Dict[str, float], spt: Dict[str, Any]
    ) -> Optional[Dict[str, Any]]:
        stage_conf = self.stages.get(current_stage) or {}
        next_stage = stage_conf.get("next_down")
        if not next_stage:
            return None

        triggers = stage_conf.get("decay_triggers", {}) or {}

        # 监控 DECAY 检查过程
        print(f"[MONITOR] stage_decay_check: current_stage={current_stage}, next_down={next_stage}")

        max_scores = triggers.get("max_scores") or {}
        for dim, limit in (max_scores or {}).items():
            limit_val = max(0.0, min(1.0, float(limit)))
            score_val = max(0.0, min(1.0, float(scores.get(dim, 0.0) or 0.0)))
            print(f"[MONITOR] stage_decay_check_max_score: dim={dim}, score={score_val:.3f}, limit={limit_val:.3f}")
            if score_val <= limit_val:
                print(f"[MONITOR] stage_decay_triggered: {dim}={score_val:.3f} <= limit={limit_val:.3f}")
                return {
                    "new_stage": next_stage,
                    "reason": f"Score {dim} dropped below {limit_val:.2f} (actual={score_val:.3f}).",
                    "transition_type": "DECAY",
                }

        if "conditional_drop" in triggers:
            cd = triggers.get("conditional_drop") or {}
            cond_str = str(cd.get("condition") or "")
            closeness_raw = float(scores.get("closeness", 0.0) or 0.0)
            cond_met = _safe_check_condition(cond_str, closeness=closeness_raw)
            print(f"[MONITOR] stage_decay_check_conditional: condition={cond_str}, closeness={closeness_raw:.3f}, condition_met={cond_met}")
            if cond_met:
                sub = cd.get("triggers") or {}
                for dim, limit in sub.items():
                    limit_val = max(0.0, min(1.0, float(limit)))
                    score_val = max(0.0, min(1.0, float(scores.get(dim, 0.0) or 0.0)))
                    print(f"[MONITOR] stage_decay_check_conditional_sub: dim={dim}, score={score_val:.3f}, limit={limit_val:.3f}")
                    if score_val < limit_val:
                        print(f"[MONITOR] stage_decay_triggered: conditional_drop, {dim}={score_val:.3f} < limit={limit_val:.3f}")
                        return {
                            "new_stage": next_stage,
                            "reason": f"High intimacy but low {dim} (Toxic). closeness={closeness_raw:.3f}, {dim}={score_val:.3f}",
                            "transition_type": "DECAY",
                        }

        behavior_required = triggers.get("spt_behavior")
        if behavior_required == "depth_reduction":
            depth_trend = str(spt.get("depth_trend") or "stable")
            print(f"[MONITOR] stage_decay_check_spt_behavior: behavior_required=depth_reduction, depth_trend={depth_trend}")
            if depth_trend == "decreasing":
                print(f"[MONITOR] stage_decay_triggered: depth_reduction, depth_trend=decreasing")
                return {
                    "new_stage": next_stage,
                    "reason": f"User is withdrawing (Depenetration). depth_trend={depth_trend}",
                    "transition_type": "DECAY",
                }
        if behavior_required == "breadth_reduction":
            breadth = int(spt.get("breadth", 0) or 0)
            print(f"[MONITOR] stage_decay_check_spt_behavior: behavior_required=breadth_reduction, breadth={breadth}")
            if breadth <= 1:
                print(f"[MONITOR] stage_decay_triggered: breadth_reduction, breadth={breadth} <= 1")
                return {
                    "new_stage": next_stage,
                    "reason": f"Topic breadth collapsed (breadth reduction). breadth={breadth}",
                    "transition_type": "DECAY",
                }

        return None

    def _check_growth(
        self, current_stage: str, scores: Dict[str, float], spt: Dict[str, Any], *, user_turns: int = 0
    ) -> Optional[Dict[str, Any]]:
        stage_conf = self.stages.get(current_stage) or {}
        next_stage = stage_conf.get("next_up")
        if not next_stage:
            return None

        entry_req = stage_conf.get("up_entry", {}) or {}
        veto_req = stage_conf.get("up_veto", {}) or {}

        # initiating -> experimenting：至少 3 轮用户输入，避免“一句话就升级”
        if str(current_stage) == "initiating" and str(next_stage) == "experimenting":
            if int(user_turns or 0) < 3:
                return None

        for dim, min_val in (entry_req.get("min_scores") or {}).items():
            min_val_norm = max(0.0, min(1.0, float(min_val)))
            score_val = max(0.0, min(1.0, float(scores.get(dim, 0.0) or 0.0)))
            if score_val < min_val_norm:
                return None

        min_spt_depth = int(entry_req.get("min_spt_depth", 0) or 0)
        spt_depth = int(spt.get("depth", 1) or 1)
        print(f"[MONITOR] stage_growth_check_spt_depth: depth={spt_depth}, required={min_spt_depth}")
        if spt_depth < min_spt_depth:
            print(f"[MONITOR] stage_growth_blocked: spt_depth={spt_depth} < required={min_spt_depth}")
            return None

        min_breadth = int(entry_req.get("min_topic_breadth", 0) or 0)
        spt_breadth = int(spt.get("breadth", 0) or 0)
        print(f"[MONITOR] stage_growth_check_spt_breadth: breadth={spt_breadth}, required={min_breadth}")
        if spt_breadth < min_breadth:
            print(f"[MONITOR] stage_growth_blocked: spt_breadth={spt_breadth} < required={min_breadth}")
            return None

        required_signals = entry_req.get("required_signals") or []
        recent = set([str(x) for x in (spt.get("recent_signals") or [])])
        print(f"[MONITOR] stage_growth_check_signals: required={required_signals}, recent={list(recent)}")
        for sig in required_signals:
            if str(sig) not in recent:
                print(f"[MONITOR] stage_growth_blocked: missing_signal={sig}")
                return None

        # 检查 veto requirements
        for dim, min_val in (veto_req.get("min_scores") or {}).items():
            min_val_norm = max(0.0, min(1.0, float(min_val)))
            score_val = max(0.0, min(1.0, float(scores.get(dim, 0.0) or 0.0)))
            print(f"[MONITOR] stage_growth_check_veto_score: dim={dim}, score={score_val:.3f}, veto_threshold={min_val_norm:.3f}")
            if score_val < min_val_norm:
                print(f"[MONITOR] stage_growth_vetoed: {dim}={score_val:.3f} < veto_threshold={min_val_norm:.3f}")
                logger.info(f"Growth vetoed: {dim} too low (score={score_val:.2f} < threshold={min_val_norm:.2f}).")
                return None

        if bool(veto_req.get("check_power_balance")):
            # power balance: 0.5 是平衡点，计算偏离度（0-1 范围）
            power = max(0.0, min(1.0, float(scores.get("power", 0.5) or 0.5)))
            imbalance = abs(power - 0.5) * 2.0  # 0-1 范围
            limit = max(0.0, min(1.0, float(self.settings.get("power_balance_threshold", 0.3) or 0.3)))
            print(f"[MONITOR] stage_growth_check_power_balance: power={power:.3f}, imbalance={imbalance:.3f}, threshold={limit:.3f}")
            if imbalance > limit:
                print(f"[MONITOR] stage_growth_vetoed: power_imbalance={imbalance:.3f} > threshold={limit:.3f}")
                logger.info("Growth vetoed: Power imbalance.")
                return None

        print(f"[MONITOR] stage_growth_triggered: all_entry_criteria_met")
        return {"new_stage": next_stage, "reason": "All entry criteria met.", "transition_type": "GROWTH"}

    def _derive_spt_from_assets(self, state: Dict[str, Any]) -> Dict[str, Any]:
        assets = state.get("relationship_assets") or {}
        topic_list = assets.get("topic_history") or []
        breadth = assets.get("breadth_score")
        if breadth is None:
            breadth = len(set([str(x) for x in topic_list])) if topic_list else 0
        depth = assets.get("max_spt_depth") or 1
        return {
            "depth": int(depth) if isinstance(depth, (int, float, str)) else 1,
            "breadth": int(breadth) if isinstance(breadth, (int, float, str)) else 0,
            "topic_list": [str(x) for x in topic_list],
            "depth_trend": "stable",
            "recent_signals": [],
        }


def create_stage_manager_node(config_path: Optional[str] = None):
    manager = KnappStageManager(config_path=config_path)

    def node(state: AgentState) -> Dict[str, Any]:
        current = str(state.get("current_stage") or "initiating")

        spt_info = state.get("spt_info")
        if not spt_info:
            analysis = state.get("latest_relationship_analysis") or {}
            detected = analysis.get("detected_signals") or []
            assets = state.get("relationship_assets") or {}
            spt_info = {
                "depth": int(assets.get("max_spt_depth") or 1),
                "breadth": int(assets.get("breadth_score") or 0),
                "topic_list": list(assets.get("topic_history") or []),
                "depth_trend": "stable",
                "recent_signals": [str(x) for x in detected],
            }

        # 收集当前状态信息用于监控
        rel_state = state.get("relationship_state") or {}
        rel_deltas = state.get("relationship_deltas_applied") or state.get("relationship_deltas") or {}
        user_turns = manager._count_user_turns(state)

        result = manager.evaluate_transition(current, {**state, "spt_info": spt_info})
        new_stage = result.get("new_stage", current)
        ttype = result.get("transition_type", "STAY")
        reason = result.get("reason", "")

        # ### 6.2 需要监控的参数 - stage 变化触发的详细信息
        if ttype != "STAY" and new_stage != current:
            # Stage 变化发生
            print(f"[MONITOR] stage_transition_triggered:")
            print(f"  from_stage={current}")
            print(f"  to_stage={new_stage}")
            print(f"  transition_type={ttype}")
            print(f"  reason={reason}")
            print(f"  relationship_state: closeness={rel_state.get('closeness', 0):.3f}, trust={rel_state.get('trust', 0):.3f}, liking={rel_state.get('liking', 0):.3f}, respect={rel_state.get('respect', 0):.3f}, warmth={rel_state.get('warmth', 0):.3f}, power={rel_state.get('power', 0):.3f}")
            print(f"  relationship_deltas: {rel_deltas}")
            print(f"  spt_info: depth={spt_info.get('depth', 1)}, breadth={spt_info.get('breadth', 0)}, depth_trend={spt_info.get('depth_trend', 'stable')}, recent_signals={spt_info.get('recent_signals', [])}")
            print(f"  user_turns={user_turns}")
            print(f"🚀 STAGE CHANGE: {current} -> {new_stage} ({reason})")
            print("[StageManager] done")
            return {
                "current_stage": new_stage,
                "stage_narrative": reason,
                "stage_transition": {"from": current, "to": new_stage, "type": ttype, "reason": reason},
                "spt_info": spt_info,
            }
        else:
            # Stage 保持不变，也记录当前状态
            print(f"[MONITOR] stage_no_change:")
            print(f"  current_stage={current}")
            print(f"  transition_type={ttype}")
            print(f"  reason={reason}")
            print(f"  relationship_state: closeness={rel_state.get('closeness', 0):.3f}, trust={rel_state.get('trust', 0):.3f}, liking={rel_state.get('liking', 0):.3f}, respect={rel_state.get('respect', 0):.3f}, warmth={rel_state.get('warmth', 0):.3f}, power={rel_state.get('power', 0):.3f}")
            print(f"  relationship_deltas: {rel_deltas}")
            print(f"  spt_info: depth={spt_info.get('depth', 1)}, breadth={spt_info.get('breadth', 0)}, depth_trend={spt_info.get('depth_trend', 'stable')}, recent_signals={spt_info.get('recent_signals', [])}")
            print(f"  user_turns={user_turns}")
        print("[StageManager] done")
        return {"spt_info": spt_info}

    return node

