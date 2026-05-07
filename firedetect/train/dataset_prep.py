"""下载并合并 FASDD_CV + D-Fire（→ Tier 1 主训练集）。

P1 阶段任务：
1. 从 SciDB 下载 FASDD_CV，从 GitHub/OneDrive 下载 D-Fire
2. pHash 去重（imagededup）—— FASDD 部分来源就是 D-Fire，必须去重
3. 类 ID 统一（0=fire, 1=smoke）
4. 7:1:2 切分到 data/merged/{train,val,test}
5. 每张图打 source tag，写入 data/merged/sources.csv
"""
from __future__ import annotations
import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sources", nargs="+",
                        default=["fasdd_cv", "dfire"],
                        help="要拉取并合并的数据集名")
    parser.add_argument("--out", default="../data/merged")
    parser.add_argument("--cache", default="../data/raw")
    parser.add_argument("--dedup", action="store_true", default=True,
                        help="启用 pHash 去重（推荐）")
    parser.add_argument("--split", default="0.7,0.1,0.2", help="train/val/test 比例")
    args = parser.parse_args()

    # TODO P1 实现
    print("dataset_prep stub — implement in P1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
