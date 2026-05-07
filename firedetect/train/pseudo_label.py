"""用 baseline 模型对 Home Fire / Material-Auxiliary 等图像级分类集自动伪标注。

流程（PLAN §3.2 Tier 3 + §6 P6）：
1. 加载 P2 baseline best.pt
2. 对目标集每张图跑推理，置信度 > thresh 的 bbox 写入 YOLO txt
3. 标记 needs_review 输出供人工抽检（前 1–2k 张）
4. 修正后并入 data/merged/，做第三轮 fine-tune
"""
from __future__ import annotations
import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="待伪标的数据根目录")
    parser.add_argument("--model", required=True, help="baseline best.pt 路径")
    parser.add_argument("--out", required=True, help="伪标注输出目录")
    parser.add_argument("--conf", type=float, default=0.5, help="伪标置信度阈值")
    parser.add_argument("--review-sample", type=int, default=2000,
                        help="抽几张给人工审核")
    args = parser.parse_args()

    # TODO P6
    print("pseudo_label stub — implement in P6")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
