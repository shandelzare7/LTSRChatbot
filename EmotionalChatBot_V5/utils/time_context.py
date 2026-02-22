"""
时间上下文与历史切片标记（TIME_CONTEXT / TIME_SLICE）。

- 8 块 day_part 定义（固定边界，Asia/Shanghai）
- TIME_CONTEXT：每次生成时在 prompt 顶部注入「当前时间」块
- TIME_SLICE：在历史消息之间插入「时间断裂」标记，控制密度（gap + day_part 阈值）
"""
from __future__ import annotations

import os
from datetime import datetime, timezone, timedelta
from typing import Any, Dict, List, Optional, Tuple

# 默认时区（与方案一致；可用环境变量覆盖）
DEFAULT_TZ_NAME = os.getenv("LTSR_TIMEZONE", "Asia/Shanghai")

# ---------------------------------------------------------------------------
# 1) 8 块 day_part 定义（固定边界，别让 LLM 自己猜）
# 时间范围均为本地时间 [start, end)，左闭右开
# ---------------------------------------------------------------------------
DAY_PART_RANGES: List[Tuple[str, int, int, str]] = [
    ("P0", 0, 2, "深夜"),    # 00:00–02:00
    ("P1", 2, 5, "凌晨"),   # 02:00–05:00
    ("P2", 5, 8, "清晨"),   # 05:00–08:00
    ("P3", 8, 12, "上午"),  # 08:00–12:00
    ("P4", 12, 14, "中午"), # 12:00–14:00
    ("P5", 14, 18, "下午"), # 14:00–18:00
    ("P6", 18, 20, "傍晚"), # 18:00–20:00
    ("P7", 20, 24, "夜间"), # 20:00–24:00
]


def _get_tz() -> Any:
    try:
        from zoneinfo import ZoneInfo
        return ZoneInfo(DEFAULT_TZ_NAME)
    except Exception:
        return timezone(timedelta(hours=8))  # UTC+8 fallback


def _to_local(dt: datetime) -> datetime:
    """将 naive/aware datetime 转为本地（Asia/Shanghai）naive 时间。"""
    if dt.tzinfo is None:
        dt = dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(_get_tz()).replace(tzinfo=None)


def get_day_part(dt: datetime) -> Tuple[str, str]:
    """
    根据 datetime 返回 (part_id, part_label_zh)。
    dt 可为 UTC 或 naive 本地时间，内部会转到本地后按小时取整判断。
    """
    local = _to_local(dt)
    h = local.hour
    for part_id, start, end, label in DAY_PART_RANGES:
        if start <= h < end:
            return (part_id, label)
    return ("P7", "夜间")


def format_local_datetime(dt: datetime) -> str:
    """格式化为本地时间字符串，用于 TIME_CONTEXT / TIME_SLICE。"""
    local = _to_local(dt)
    return local.strftime("%Y-%m-%d %H:%M")


def local_date_str(dt: datetime) -> str:
    """本地日期 YYYY-MM-DD，用于判断 crossed_day。"""
    local = _to_local(dt)
    return local.strftime("%Y-%m-%d")


def weekday_str(dt: datetime) -> str:
    """英文星期缩写 Mon–Sun。"""
    local = _to_local(dt)
    return local.strftime("%a")


# ---------------------------------------------------------------------------
# 2) TIME_CONTEXT：每次生成只放一块
# long_gap：距上次用户消息 >= 1h；new_session：>= 4h
# ---------------------------------------------------------------------------
GAP_NEW_SESSION = 4 * 3600   # >= 4h → session_mode=new_session
GAP_LONG_GAP = 1 * 3600      # >= 1h → session_mode=long_gap（或 TIME_SLICE SHORT_BREAK）
GAP_2H = 2 * 3600
GAP_60MIN = 60 * 60


def _parse_ts(ts: Any) -> Optional[datetime]:
    """从 additional_kwargs['timestamp'] 或 ISO 字符串解析 datetime。"""
    if ts is None:
        return None
    if isinstance(ts, datetime):
        return ts
    s = str(ts).strip()
    if not s:
        return None
    try:
        s = s.replace("Z", "+00:00")
        if s.endswith("+00:00") or s.endswith("Z") or ("+" in s or "-" in s[-6:]):
            return datetime.fromisoformat(s)
        return datetime.fromisoformat(s)
    except Exception:
        return None


# session_mode -> 中文（供可读时间模块）
SESSION_MODE_ZH = {
    "continuation": "连续对话",
    "long_gap": "间隔重连",
    "new_session": "全新会话",
}


def _time_gap_readable(sec: Optional[int], *, no_history: bool = False) -> str:
    """将秒数转为人类可读：不到1分钟 / 约X分钟 / 约X小时 / X天以上。无历史时可为（本会话首条）。"""
    if no_history or sec is None or sec < 0:
        return "（本会话首条）" if (sec is None and no_history) else "无"
    if sec < 60:
        return "不到 1 分钟"
    if sec < 3600:
        return f"约 {int(round(sec / 60))} 分钟"
    if sec < 86400:
        return f"约 {int(round(sec / 3600))} 小时"
    return f"{int(round(sec / 86400))} 天以上"


def build_time_context_block(state: Dict[str, Any]) -> str:
    """
    构建可读的时间与会话上下文一句话（核心占位符模板）。
    模板：当前现实时间是 {now_local}（{weekday} {now_day_part}）。当前处于【{session_mode_zh}】状态，距离上一次交谈已过去 {time_gap_readable}。{optional_speaker_note}
    """
    now = _parse_ts(state.get("current_time"))
    if now is None:
        now = datetime.now(timezone.utc)
    if now.tzinfo is None:
        now = now.replace(tzinfo=timezone.utc)

    _, part_label = get_day_part(now)

    since_last_user_sec: Optional[int] = None
    since_last_assistant_sec: Optional[int] = None
    session_mode = "continuation"

    chat_buffer = state.get("chat_buffer") or []
    if isinstance(chat_buffer, list) and chat_buffer:
        last_user_ts: Optional[datetime] = None
        last_ai_ts: Optional[datetime] = None
        for m in reversed(chat_buffer):
            kwargs = getattr(m, "additional_kwargs", None) or {}
            ts = _parse_ts(kwargs.get("timestamp"))
            if ts is None:
                continue
            t = getattr(m, "type", "") or ""
            if "human" in str(t).lower() or "user" in str(t).lower():
                if last_user_ts is None:
                    last_user_ts = ts
            else:
                if last_ai_ts is None:
                    last_ai_ts = ts
            if last_user_ts is not None and last_ai_ts is not None:
                break

        if last_user_ts is not None:
            delta = now - last_user_ts if last_user_ts.tzinfo else now - last_user_ts.replace(tzinfo=timezone.utc)
            since_last_user_sec = int(delta.total_seconds())
        if last_ai_ts is not None:
            delta = now - last_ai_ts if last_ai_ts.tzinfo else now - last_ai_ts.replace(tzinfo=timezone.utc)
            since_last_assistant_sec = int(delta.total_seconds())

        if since_last_user_sec is not None and since_last_user_sec >= GAP_NEW_SESSION:
            session_mode = "new_session"
        elif since_last_user_sec is not None and since_last_user_sec >= GAP_LONG_GAP:
            session_mode = "long_gap"

    now_local = format_local_datetime(now)
    weekday = weekday_str(now)
    now_day_part = part_label
    session_mode_zh = SESSION_MODE_ZH.get(session_mode, "连续对话")

    # 距离上一次交谈：取两者中较小的一个（即距最后一条消息的间隔）；无历史时用（本会话首条）
    gap_sec: Optional[int] = None
    if since_last_user_sec is not None and since_last_assistant_sec is not None:
        gap_sec = min(since_last_user_sec, since_last_assistant_sec)
    elif since_last_user_sec is not None:
        gap_sec = since_last_user_sec
    elif since_last_assistant_sec is not None:
        gap_sec = since_last_assistant_sec
    no_history = since_last_user_sec is None and since_last_assistant_sec is None
    time_gap_readable = _time_gap_readable(gap_sec, no_history=no_history)

    # 断联且上一条是 AI 发出时（用户更久没说话，即 since_last_user_sec > since_last_assistant_sec），补充旁白
    optional_speaker_note = ""
    if session_mode in ("long_gap", "new_session") and since_last_user_sec is not None and since_last_assistant_sec is not None:
        if since_last_user_sec > since_last_assistant_sec:
            optional_speaker_note = "注：断联前的最后一条消息由你发出。"

    return (
        f"当前现实时间是 {now_local}（{weekday} {now_day_part}）。"
        f"当前处于【{session_mode_zh}】状态，距离上一次交谈已过去 {time_gap_readable}。"
        f"{optional_speaker_note}"
    ).strip()


# ---------------------------------------------------------------------------
# 3) TIME_SLICE：插入规则与 label
# ---------------------------------------------------------------------------
def _time_slice_label(gap_sec: int, crossed_day: bool, part_changed: bool) -> str:
    if gap_sec >= GAP_NEW_SESSION:
        return "LONG_GAP"
    if crossed_day:
        return "NEXT_DAY"
    if gap_sec >= GAP_LONG_GAP:
        return "SHORT_BREAK"
    if part_changed and gap_sec >= GAP_2H:
        return "PART_SHIFT"
    return "CONTINUATION"


def should_insert_time_slice(prev_ts: Optional[datetime], curr_ts: Optional[datetime]) -> Tuple[bool, Optional[str]]:
    """
    判断是否在 prev 与 curr 之间插入 TIME_SLICE，以及 label。
    返回 (insert, label)。若 insert 为 False，label 可为 None。
    """
    if prev_ts is None or curr_ts is None:
        return (False, None)

    if prev_ts.tzinfo is None:
        prev_ts = prev_ts.replace(tzinfo=timezone.utc)
    if curr_ts.tzinfo is None:
        curr_ts = curr_ts.replace(tzinfo=timezone.utc)
    gap_sec = int((curr_ts - prev_ts).total_seconds())
    if gap_sec < 0:
        return (False, None)

    crossed_day = local_date_str(prev_ts) != local_date_str(curr_ts)
    part_changed = get_day_part(prev_ts)[0] != get_day_part(curr_ts)[0]

    # 强制插入
    if crossed_day:
        return (True, _time_slice_label(gap_sec, True, part_changed))
    if gap_sec >= GAP_NEW_SESSION:
        return (True, "LONG_GAP")
    if gap_sec >= GAP_LONG_GAP:
        return (True, "SHORT_BREAK")

    # 条件插入：跨时段且 >= 2h
    if part_changed and gap_sec >= GAP_2H:
        return (True, "PART_SHIFT")

    return (False, None)


def build_time_slice_marker(
    from_ts: datetime,
    to_ts: datetime,
    gap_sec: int,
    crossed_day: bool,
    label: str,
) -> str:
    """生成一条 TIME_SLICE 的 XML 字符串（不展示给用户，只给 LLM）。"""
    from_part = get_day_part(from_ts)[1]
    to_part = get_day_part(to_ts)[1]
    return (
        f'<TIME_SLICE\n'
        f'  from="{format_local_datetime(from_ts)}"\n'
        f'  to="{format_local_datetime(to_ts)}"\n'
        f'  from_part="{from_part}"\n'
        f'  to_part="{to_part}"\n'
        f'  gap_sec="{gap_sec}"\n'
        f'  crossed_day="{"true" if crossed_day else "false"}"\n'
        f'  label="{label}"\n'
        f'/>'
    )


# ---------------------------------------------------------------------------
# 4) 跨段策略规则（写进 system，供 generator 遵守）
# ---------------------------------------------------------------------------
TIME_SLICE_BEHAVIOR_RULES = """
Behavior by TIME_SLICE label (do not repeat these markers verbatim):
- PART_SHIFT: keep continuity, but avoid "just now". Light re-alignment allowed.
- SHORT_BREAK: brief re-entry (no deep recap), then continue.
- NEXT_DAY: greet according to day_part; avoid assuming last topic is still active; offer quick alignment.
- LONG_GAP: greet + ask one alignment question or provide a short recap before continuing.

Time rules:
- Do NOT mention exact timestamps unless the user explicitly asks.
- If session_mode=new_session: do a natural re-entry (brief greeting + quick alignment), avoid strong continuation.
""".strip()


# ---------------------------------------------------------------------------
# 5) 在历史消息之间插入 TIME_SLICE（返回新列表，含 SystemMessage）
# ---------------------------------------------------------------------------
def inject_time_slices_into_messages(messages: List[Any]) -> List[Any]:
    """
    遍历相邻两条消息 prev -> curr，按规则在之间插入一条 SystemMessage(TIME_SLICE)。
    返回新列表，元素为 BaseMessage 或 SystemMessage(TIME_SLICE)。
    调用方需传入 langchain_core.messages 的 BaseMessage 列表。
    """
    from langchain_core.messages import BaseMessage, SystemMessage

    out: List[Any] = []
    prev_ts: Optional[datetime] = None

    for i, msg in enumerate(messages):
        if not isinstance(msg, BaseMessage):
            out.append(msg)
            continue

        kwargs = getattr(msg, "additional_kwargs", None) or {}
        curr_ts = _parse_ts(kwargs.get("timestamp"))

        if prev_ts is not None and curr_ts is not None:
            p_ts = prev_ts.replace(tzinfo=timezone.utc) if prev_ts.tzinfo is None else prev_ts
            c_ts = curr_ts.replace(tzinfo=timezone.utc) if curr_ts.tzinfo is None else curr_ts
            insert, label = should_insert_time_slice(p_ts, c_ts)
            if insert and label and label != "CONTINUATION":
                gap_sec = int((c_ts - p_ts).total_seconds())
                crossed_day = local_date_str(p_ts) != local_date_str(c_ts)
                marker = build_time_slice_marker(p_ts, c_ts, gap_sec, crossed_day, label)
                out.append(SystemMessage(content=marker))

        out.append(msg)
        if curr_ts is not None:
            prev_ts = curr_ts

    return out
