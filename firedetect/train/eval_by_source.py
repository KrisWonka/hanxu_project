"""按 source tag / bbox 尺寸 / 场景切片评估。

为什么要切片：合并集训完看整体 mAP 不够用，**养老室内场景**的 mAP 才是真正的部署指标，
**小火召回**才是真正回应同事痛点的指标。
"""
from __future__ import annotations
import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="best.pt 或 fire.rknn")
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--by", choices=["source", "size", "scene"], default="source",
                        help="切片维度：source（公开集来源） / size（bbox 尺寸） / scene（室内/户外）")
    parser.add_argument("--size-bins", default="0,0.005,0.02,0.10,1.0",
                        help="bbox 面积占比分桶边界")
    args = parser.parse_args()

    # TODO P2/P5
    print("eval_by_source stub — implement in P2/P5")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
