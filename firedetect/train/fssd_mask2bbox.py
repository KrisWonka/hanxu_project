"""FSSD 语义分割 mask → YOLO bbox（→ Tier 2 室内增强）。

算法（PLAN §3.2 + §4.4）：
1. 读取 FSSD 原始 mask（每像素 class id）
2. 对 fire / smoke 分别做 cv2.connectedComponentsWithStats 取连通域
3. 过滤面积 < min_area、长宽比 > max_aspect 的连通域
4. 输出 YOLO txt 格式（cx cy w h 全归一化）
"""
from __future__ import annotations
import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="FSSD 原始数据根目录（含 images/ + masks/）")
    parser.add_argument("--out", required=True, help="转换后输出目录（YOLO 格式）")
    parser.add_argument("--min-area", type=int, default=64,
                        help="连通域最小像素面积，过滤碎渣")
    parser.add_argument("--max-aspect", type=float, default=8.0,
                        help="bbox 长宽比上限，过滤窄条噪声")
    args = parser.parse_args()

    # TODO P4
    print("fssd_mask2bbox stub — implement in P4")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
