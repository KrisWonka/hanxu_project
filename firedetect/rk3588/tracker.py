"""SimpleIoUTracker — 多目标跟踪器（stub，P3 阶段从 handwaving 拆出实现）。

设计来自 ``handwaving/huishou/rk3588/wave_detector.py``。
本文件先放骨架让 ``fire_detector.py`` 能 import；真实实现 P3 补全
（IoU 匹配 + 归一化中心距兜底 + 短时恒速预测桥接 dropout）。
"""
from __future__ import annotations
from dataclasses import dataclass, field
from typing import List, Tuple


@dataclass
class Track:
    track_id: int
    bbox: Tuple[float, float, float, float]   # xyxy
    score: float
    cls: int
    age: int = 0
    lost: int = 0
    history: list = field(default_factory=list)  # [(cx, cy, frame_id), ...]


class SimpleIoUTracker:
    def __init__(self, iou_thresh: float = 0.4, max_lost: int = 10):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.tracks: List[Track] = []
        self._next_id = 0

    def update(self, detections):
        """detections: list of (x1, y1, x2, y2, score, cls). 返回当前活跃 tracks."""
        # TODO P3：从 handwaving/huishou/rk3588/wave_detector.py 移植 SimpleIoUTracker 完整逻辑
        raise NotImplementedError("SimpleIoUTracker.update will be implemented in P3")
