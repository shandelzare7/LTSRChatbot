"""
Bot 忙碌度兜底：按当前时间与工作日/休息日返回 0~1 的 busy 值。
用于每轮会话开始时为 mood_state.busyness 赋兜底值，模拟上班族/学生作息。
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Optional

# 时间段 (start_min, end_min, busy)，分钟从 0 计（0:00=0, 23:59=1439）
# 左闭右开 [start, end)，最后一档 end=24*60 表示到 24:00

def _minute_of_day(dt: datetime) -> int:
    """当天 0 点起的分钟数，0~1439。"""
    return dt.hour * 60 + dt.minute


# 工作日 (周一=0 至 周五=4)
_WEEKDAY_SLOTS = [
    (0, 7 * 60, 0.95),         # 00:00-07:00 睡眠
    (7 * 60, 8 * 60 + 30, 0.60),   # 07:00-08:30 起床洗漱
    (8 * 60 + 30, 9 * 60 + 30, 0.75),  # 08:30-09:30 早高峰通勤
    (9 * 60 + 30, 12 * 60, 0.85),   # 09:30-12:00 上午工作
    (12 * 60, 13 * 60 + 30, 0.30),  # 12:00-13:30 午餐午休
    (13 * 60 + 30, 16 * 60, 0.85),  # 13:30-16:00 下午工作
    (16 * 60, 17 * 60, 0.60),      # 16:00-17:00 短暂摸鱼
    (17 * 60, 18 * 60, 0.85),      # 17:00-18:00 下午工作
    (18 * 60, 19 * 60 + 30, 0.70), # 18:00-19:30 晚高峰通勤
    (19 * 60 + 30, 20 * 60 + 30, 0.40),  # 19:30-20:30 晚餐杂事
    (20 * 60 + 30, 23 * 60 + 30, 0.10),  # 20:30-23:30 晚间娱乐
    (23 * 60 + 30, 24 * 60, 0.50), # 23:30-24:00 睡前酝酿
]

# 休息日 (周六、周日；法定节假日暂按周末处理)
_WEEKEND_SLOTS = [
    (0, 9 * 60 + 30, 0.95),        # 00:00-09:30 睡眠/赖床
    (9 * 60 + 30, 11 * 60, 0.20),  # 09:30-11:00 缓慢起床/早午餐
    (11 * 60, 17 * 60, 0.50),      # 11:00-17:00 外出/兴趣爱好
    (17 * 60, 19 * 60, 0.30),      # 17:00-19:00 晚餐
    (19 * 60, 24 * 60, 0.10),      # 19:00-24:00 晚间重度娱乐
]


def get_busy_fallback_from_schedule(
    dt: Optional[datetime] = None,
    *,
    use_utc: bool = False,
) -> float:
    """
    按给定时间（默认当前时间）返回 0~1 的 busy 兜底值。
    - 周一~五：工作日作息
    - 周六、周日：休息日作息（法定节假日暂按周末）
    - dt 为 None 时使用当前时间；若 use_utc=True 用 UTC，否则用本地时间。
    """
    if dt is None:
        dt = datetime.now(timezone.utc) if use_utc else datetime.now()
    if dt.tzinfo and use_utc:
        # 保持 UTC
        pass
    elif dt.tzinfo and not use_utc:
        dt = dt.astimezone()
    elif not dt.tzinfo:
        # naive 视为本地
        pass

    minute = _minute_of_day(dt)
    weekday = dt.weekday()  # 0=Mon, 6=Sun
    slots = _WEEKDAY_SLOTS if weekday < 5 else _WEEKEND_SLOTS

    for start, end, busy in slots:
        if start <= minute < end:
            return busy
    return 0.5  # fallback
