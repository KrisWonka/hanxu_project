"""
Time, date, and simple timer utilities.
"""

from __future__ import annotations

import logging
import threading
import time
from datetime import datetime

logger = logging.getLogger(__name__)

_timers: dict[str, threading.Timer] = {}


def get_current_time() -> dict:
    """Return current date and time in Asia/Shanghai timezone."""
    now = datetime.now()
    weekdays = ["星期一", "星期二", "星期三", "星期四", "星期五", "星期六", "星期日"]
    return {
        "success": True,
        "date": now.strftime("%Y年%m月%d日"),
        "time": now.strftime("%H:%M:%S"),
        "weekday": weekdays[now.weekday()],
    }


def set_timer(seconds: int, label: str = "定时器") -> dict:
    """
    Set a simple countdown timer. When it fires, logs and prints an alert.
    Returns immediately.
    """
    if seconds <= 0 or seconds > 86400:
        return {"success": False, "error": "时间范围: 1秒 ~ 24小时"}

    timer_id = f"{label}_{int(time.time())}"

    def _on_fire():
        logger.info("Timer fired: %s", label)
        print(f"\n⏰ 定时器到！{label} ({seconds}秒)")
        _timers.pop(timer_id, None)

    t = threading.Timer(seconds, _on_fire)
    t.daemon = True
    t.start()
    _timers[timer_id] = t

    if seconds >= 3600:
        display = f"{seconds // 3600}小时{(seconds % 3600) // 60}分钟"
    elif seconds >= 60:
        display = f"{seconds // 60}分钟{seconds % 60}秒" if seconds % 60 else f"{seconds // 60}分钟"
    else:
        display = f"{seconds}秒"

    logger.info("Timer set: %s for %s", label, display)
    return {"success": True, "timer_id": timer_id, "duration": display, "label": label}


def cancel_timer(timer_id: str | None = None) -> dict:
    """Cancel a specific timer or the most recent one."""
    if not _timers:
        return {"success": False, "error": "没有正在运行的定时器"}

    if timer_id and timer_id in _timers:
        _timers[timer_id].cancel()
        _timers.pop(timer_id)
        return {"success": True}

    last_id = list(_timers.keys())[-1]
    _timers[last_id].cancel()
    _timers.pop(last_id)
    return {"success": True, "cancelled": last_id}
