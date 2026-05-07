"""火焰检测后处理（stub，P3 阶段实现）。

按 PLAN §4 的三层过滤：
  4.1 单帧级：置信度门 + HSV 颜色复核 + 尺寸/长宽比
  4.2 时序级：闪烁 FFT + 运动一致性 + Tracker
  4.3 告警级：N/M 帧投票 + 冷却期
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Optional
import json


@dataclass
class FireConfig:
    fire_conf_thresh: float = 0.45
    smoke_conf_thresh: float = 0.40
    hsv_red_ratio_min: float = 0.30
    hsv_v_min: int = 150
    bbox_area_min_ratio: float = 5e-4
    bbox_aspect_max: float = 8.0
    flicker_freq_lo: float = 3.0
    flicker_freq_hi: float = 20.0
    vote_window_frames: int = 30
    vote_threshold: int = 18
    cooldown_seconds: float = 10.0


@dataclass
class FireEvent:
    ts: float
    track_id: int
    type: str  # "fire_alarm" | "smoke_alarm"
    bbox: List[float]
    conf: float
    frames_voted: int

    def to_jsonl(self) -> str:
        return json.dumps(self.__dict__, ensure_ascii=False)


class FireDetector:
    """E2E 后处理：detections → events."""

    def __init__(self, config: Optional[FireConfig] = None):
        self.config = config or FireConfig()
        # TODO P3: 初始化 tracker、滑窗 buffer、冷却 dict

    def process(self, frame, detections, ts: Optional[float] = None) -> List[FireEvent]:
        """frame: BGR ndarray; detections: list of (x1,y1,x2,y2,score,cls).

        返回新触发的告警事件（已通过单帧 + 时序 + 投票 + 冷却全套过滤）。
        """
        # TODO P3: HSV 复核 + tracker 更新 + 闪烁 FFT + 投票 + 冷却
        return []
