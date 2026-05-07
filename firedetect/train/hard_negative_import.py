"""把同事旧模型的误报截图导入训练集作为硬负样本。

为什么要这步：上一版模型踩的"看到红色就报火"的坑，最高效的根治方法是把
那些误报截图直接喂回训练，让模型学到"这些不是火"。

输入：一个目录，里面是旧模型在生产中的误报 case 截图（jpg/png）。
处理：
  1. 每张图配一个**空 YOLO txt**（YOLO 协定：空 txt = 全图都是负样本）
  2. 写入 data/merged/{train,val} 对应 source=hard_negatives 子集
  3. 训练时通过 weighted sampling 过采样 ×3 ~ ×5
"""
from __future__ import annotations
import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--src", required=True, help="同事提供的 FP 截图目录")
    parser.add_argument("--out", default="../data/merged",
                        help="目标合并数据集根目录")
    parser.add_argument("--weight-multiplier", type=int, default=3,
                        help="训练时这部分样本的过采样倍数")
    args = parser.parse_args()

    # TODO P1（等同事提供截图后）
    print("hard_negative_import stub — implement in P1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
