"""火焰检测后处理 — PLAN §4 三层过滤实装。

输入：模型每帧输出的 detections (x1,y1,x2,y2,conf,cls)
输出：触发的告警事件 FireEvent

三层过滤：
  §4.1 单帧级
    - 置信度门（fire ≥ 0.45 / smoke ≥ 0.40）
    - HSV 颜色复核（fire bbox 内火色像素占比 ≥ 30%）
    - 尺寸 + 长宽比过滤
  §4.2 时序级
    - IoU tracker（IoU + 中心距匹配 + 短时丢失桥接）
    - 静止物体过滤（位置稳定 ∧ bbox 内像素 MAD 低 = 误报源）
  §4.3 告警级
    - N/M 帧投票（30 帧窗口至少 18 帧通过 §4.1+§4.2 才升级）
    - 冷却期（同 track 触发后 10s 内不再升级）

光是 Light/zfalsebug 类直接丢弃，不参与告警逻辑（它们是模型内部的"误报抑制类"）。
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

import numpy as np

try:
    import cv2
except ImportError:
    cv2 = None  # 允许 import 时 cv2 不在场（板上单独装）


# ─────────────────────────────────────────────────────────────────
# 配置
# ─────────────────────────────────────────────────────────────────
@dataclass
class FireConfig:
    # § 4.1 单帧级
    fire_conf_thresh: float = 0.45
    smoke_conf_thresh: float = 0.40
    hsv_red_ratio_min: float = 0.30   # bbox 内火色像素至少占比
    hsv_v_min: int = 150              # HSV V 通道最低（够亮）
    hsv_s_min: int = 80               # HSV S 通道最低（足够饱和）
    bbox_area_min_ratio: float = 5e-4 # bbox 面积最小占图比例
    bbox_aspect_max: float = 8.0      # 长宽比上限

    # § 4.2 时序级 - tracker
    iou_thresh: float = 0.4
    max_lost: int = 10

    # § 4.2 时序级 - 静止过滤
    stationary_window: int = 30           # 累积多少帧才判
    stationary_pos_std_thresh: float = 0.012  # 中心点位移 std / 短边
    stationary_mad_thresh: float = 8.0    # bbox 内 16x16 缩略图帧间 MAD

    # § 4.3 告警级
    vote_window_frames: int = 30
    vote_threshold: int = 18
    cooldown_seconds: float = 10.0

    # 内部
    crop_hash_size: int = 16  # 静止过滤用的下采样 patch 大小


# ─────────────────────────────────────────────────────────────────
# 数据结构
# ─────────────────────────────────────────────────────────────────
@dataclass
class TrackHistory:
    ts: float
    bbox: Tuple[float, float, float, float]
    conf: float
    crop_patch: np.ndarray  # 下采样后的 bbox 区域，用于像素差分


@dataclass
class Track:
    track_id: int
    cls: int
    bbox: Tuple[float, float, float, float]
    conf: float
    age: int = 0
    lost: int = 0
    history: List[TrackHistory] = field(default_factory=list)
    pass_frames: int = 0       # 累积通过单帧+时序过滤的次数（投票用）
    last_alarm_ts: float = 0.0 # 上次告警时间（冷却用）

    def center(self) -> Tuple[float, float]:
        return ((self.bbox[0] + self.bbox[2]) / 2,
                (self.bbox[1] + self.bbox[3]) / 2)

    def short_edge(self) -> float:
        return min(self.bbox[2] - self.bbox[0], self.bbox[3] - self.bbox[1])


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


# ─────────────────────────────────────────────────────────────────
# 工具函数
# ─────────────────────────────────────────────────────────────────
def iou_xyxy(b1, b2) -> float:
    x1, y1 = max(b1[0], b2[0]), max(b1[1], b2[1])
    x2, y2 = min(b1[2], b2[2]), min(b1[3], b2[3])
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    return inter / max(a1 + a2 - inter, 1e-9)


def hsv_red_ratio(frame: np.ndarray, bbox, v_min: int, s_min: int) -> float:
    """bbox 区域内"火色像素"占比（OpenCV BGR 输入）。

    定义：H∈[0,30]∪[330,360] (360 制) ∧ S≥s_min ∧ V≥v_min
    """
    if cv2 is None:
        return 1.0  # 无 cv2 时直接放过（仅用于无 cv2 环境的 stub）
    x1, y1, x2, y2 = (max(0, int(b)) for b in bbox)
    H, W = frame.shape[:2]
    x2 = min(W, x2)
    y2 = min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return 0.0
    hsv = cv2.cvtColor(frame[y1:y2, x1:x2], cv2.COLOR_BGR2HSV)
    h, s, v = hsv[:, :, 0], hsv[:, :, 1], hsv[:, :, 2]
    # OpenCV H 是 0-180，对应 0-360°：H∈[0,30]∪[330,360] → [0,15]∪[165,180]
    mask = ((h <= 15) | (h >= 165)) & (s >= s_min) & (v >= v_min)
    return float(mask.mean())


def make_crop_patch(frame: np.ndarray, bbox, size: int = 16) -> np.ndarray:
    """把 bbox 区域下采样到 size×size 灰度，用于像素差分。

    返回 (size, size) uint8 ndarray。失败返回零阵。
    """
    if cv2 is None:
        return np.zeros((size, size), dtype=np.uint8)
    x1, y1, x2, y2 = (max(0, int(b)) for b in bbox)
    H, W = frame.shape[:2]
    x2 = min(W, x2)
    y2 = min(H, y2)
    if x2 <= x1 or y2 <= y1:
        return np.zeros((size, size), dtype=np.uint8)
    crop = frame[y1:y2, x1:x2]
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return cv2.resize(gray, (size, size), interpolation=cv2.INTER_AREA)


# ─────────────────────────────────────────────────────────────────
# IoU Tracker
# ─────────────────────────────────────────────────────────────────
class IoUTracker:
    """同类 IoU 贪心匹配 tracker。

    - 同类 + 最高 IoU 优先匹配
    - 没匹配上的 detection → 新 track
    - 连续 lost > max_lost 帧 → 删除 track
    """

    def __init__(self, iou_thresh: float = 0.4, max_lost: int = 10):
        self.iou_thresh = iou_thresh
        self.max_lost = max_lost
        self.tracks: List[Track] = []
        self._next_id = 0

    def update(self, detections: List, frame: np.ndarray, ts: float,
               crop_size: int = 16) -> List[Track]:
        """detections: [(x1,y1,x2,y2,conf,cls), ...]"""
        used = set()
        # 现有 track 找匹配
        for tr in self.tracks:
            best_iou, best_idx = 0.0, -1
            for i, det in enumerate(detections):
                if i in used:
                    continue
                if int(det[5]) != tr.cls:
                    continue
                iou = iou_xyxy(tr.bbox, det[:4])
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            if best_iou >= self.iou_thresh:
                det = detections[best_idx]
                tr.bbox = tuple(map(float, det[:4]))
                tr.conf = float(det[4])
                tr.age += 1
                tr.lost = 0
                tr.history.append(TrackHistory(
                    ts=ts,
                    bbox=tr.bbox,
                    conf=tr.conf,
                    crop_patch=make_crop_patch(frame, tr.bbox, crop_size),
                ))
                used.add(best_idx)
            else:
                tr.lost += 1

        # 没匹配上的新 detection → 创建新 track
        for i, det in enumerate(detections):
            if i in used:
                continue
            bbox = tuple(map(float, det[:4]))
            tr = Track(
                track_id=self._next_id,
                cls=int(det[5]),
                bbox=bbox,
                conf=float(det[4]),
                age=1,
                history=[TrackHistory(
                    ts=ts, bbox=bbox, conf=float(det[4]),
                    crop_patch=make_crop_patch(frame, bbox, crop_size),
                )],
            )
            self.tracks.append(tr)
            self._next_id += 1

        # 清理超时 track
        self.tracks = [tr for tr in self.tracks if tr.lost <= self.max_lost]
        return [tr for tr in self.tracks if tr.lost == 0]


# ─────────────────────────────────────────────────────────────────
# FireDetector 主类
# ─────────────────────────────────────────────────────────────────
class FireDetector:
    def __init__(self, config: Optional[FireConfig] = None):
        self.config = config or FireConfig()
        self.tracker = IoUTracker(
            iou_thresh=self.config.iou_thresh,
            max_lost=self.config.max_lost,
        )

    # § 4.1 单帧级
    def passes_frame_filters(self, frame: np.ndarray, track: Track) -> bool:
        c = self.config
        # 置信度门
        thresh = c.fire_conf_thresh if track.cls == 0 else c.smoke_conf_thresh
        if track.conf < thresh:
            return False
        # 尺寸 / 长宽比
        bw = track.bbox[2] - track.bbox[0]
        bh = track.bbox[3] - track.bbox[1]
        H, W = frame.shape[:2]
        area_ratio = (bw * bh) / max(H * W, 1)
        if area_ratio < c.bbox_area_min_ratio:
            return False
        if max(bw, bh) / max(min(bw, bh), 1) > c.bbox_aspect_max:
            return False
        # HSV 颜色复核（仅 fire 类。smoke 没典型颜色，跳过）
        if track.cls == 0:
            if hsv_red_ratio(frame, track.bbox, c.hsv_v_min, c.hsv_s_min) < c.hsv_red_ratio_min:
                return False
        return True

    # § 4.2 时序级 - 静止物体过滤
    def is_stationary(self, track: Track) -> bool:
        """位置稳定 ∧ bbox 内像素几乎不变 = 静止误报源（LED/红窗帘等）"""
        c = self.config
        if track.age < c.stationary_window:
            return False
        recent = track.history[-c.stationary_window:]

        # A. 位置稳定性（中心点 std / 短边）
        centers = np.array([
            ((h.bbox[0] + h.bbox[2]) / 2, (h.bbox[1] + h.bbox[3]) / 2)
            for h in recent
        ])
        short_edges = np.array([
            min(h.bbox[2] - h.bbox[0], h.bbox[3] - h.bbox[1])
            for h in recent
        ])
        pos_std = centers.std(axis=0).max() / max(short_edges.mean(), 1.0)

        # C. bbox 内像素差分（16x16 灰度图帧间 MAD）
        mads = []
        for i in range(1, len(recent)):
            d = np.abs(recent[i].crop_patch.astype(np.int16)
                       - recent[i - 1].crop_patch.astype(np.int16)).mean()
            mads.append(d)
        avg_mad = float(np.mean(mads)) if mads else 0.0

        return (pos_std < c.stationary_pos_std_thresh
                and avg_mad < c.stationary_mad_thresh)

    # § 主流程
    def process(self, frame: np.ndarray, detections: List,
                ts: Optional[float] = None) -> List[FireEvent]:
        """frame: BGR ndarray
        detections: [(x1,y1,x2,y2,conf,cls), ...]
        返回触发的新 FireEvent（已通过单帧 + 时序 + 投票 + 冷却）。
        """
        if ts is None:
            ts = time.time()
        c = self.config

        # 只关注 fire (0) 和 smoke (1)，Light(2)/zfalsebug(3) 是模型内部抑制类
        det_filtered = [d for d in detections if int(d[5]) in (0, 1)]

        active = self.tracker.update(det_filtered, frame, ts, c.crop_hash_size)

        events = []
        for tr in active:
            # § 4.1 单帧
            if not self.passes_frame_filters(frame, tr):
                tr.pass_frames = max(0, tr.pass_frames - 1)
                continue
            # § 4.2 静止过滤
            if self.is_stationary(tr):
                tr.pass_frames = max(0, tr.pass_frames - 1)
                continue
            # 通过：累积
            tr.pass_frames += 1
            # § 4.3 投票
            if tr.pass_frames < c.vote_threshold:
                continue
            # § 4.3 冷却
            if ts - tr.last_alarm_ts < c.cooldown_seconds:
                continue
            # 升级告警
            tr.last_alarm_ts = ts
            events.append(FireEvent(
                ts=ts,
                track_id=tr.track_id,
                type="fire_alarm" if tr.cls == 0 else "smoke_alarm",
                bbox=list(tr.bbox),
                conf=tr.conf,
                frames_voted=tr.pass_frames,
            ))

        return events
