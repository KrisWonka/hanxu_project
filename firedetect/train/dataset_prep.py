"""dataset_prep.py — P1 数据集合并: FASDD_CV + D-Fire → data/merged/

按 6 步走（详见 Obsidian 笔记《CV 学习笔记 — dataset_prep 数据集合并 6 步》）：
  1. FASDD COCO JSON → YOLO 内存映射
  2. D-Fire 类 ID 翻转（在 step 3 内联，避免破坏 raw/）
  3. 合并 + hardlink + 加前缀 → data/merged/{train,val,test}/{images,labels}/
  4. pHash 跨集去重（跨 split 重复优先删 val/test，同 split 删较小分辨率）
  5. 写 sources.csv
  6. 生成 data.yaml

用法：
    python dataset_prep.py                       # 全部 6 步
    python dataset_prep.py --steps 3,5,6         # 只跑指定步骤
    python dataset_prep.py --skip-dedup          # 跳过最慢的 step 4
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import sys
import time
from collections import defaultdict
from pathlib import Path
from typing import Dict, List, Tuple

import yaml


# 标准类 ID 约定：0=fire, 1=smoke（匹配 FASDD 官方；D-Fire 镜像与之相反，需翻转）
DFIRE_FLIP_MAP = {"0": "1", "1": "0"}
SPLITS = ("train", "val", "test")


# ─────────────────────────────────────────────────────────────────
# Step 1: FASDD COCO → YOLO（内存映射）
# ─────────────────────────────────────────────────────────────────
def step1_load_fasdd_coco(fasdd_root: Path) -> Dict[str, Dict[str, List[str]]]:
    """读 FASDD COCO JSON，返回 {split: {image_filename: [yolo_lines]}}。

    yolo_lines 元素形如 "0 0.512 0.341 0.082 0.045"（cls cx cy w h，全归一化）。
    在内存中搞定，不落 txt（step 3 合并时直接写到 merged/）。
    """
    print("=" * 60)
    print("Step 1: FASDD COCO → YOLO (in-memory)")
    print("=" * 60)

    result: Dict[str, Dict[str, List[str]]] = {}
    for split in SPLITS:
        json_path = fasdd_root / "annotations" / f"{split}.json"
        if not json_path.exists():
            print(f"  [skip] {json_path} 不存在")
            continue

        t0 = time.time()
        with open(json_path) as f:
            data = json.load(f)

        # 类 ID 校验：必须 0=fire, 1=smoke
        cats = {c["id"]: c["name"] for c in data.get("categories", [])}
        if cats:
            assert cats.get(0) == "fire" and cats.get(1) == "smoke", \
                f"FASDD {split}.json 类 ID 异常: {cats}"

        images = {img["id"]: img for img in data["images"]}
        anns_by_img = defaultdict(list)
        for ann in data.get("annotations", []):
            anns_by_img[ann["image_id"]].append(ann)

        split_result: Dict[str, List[str]] = {}
        for img_id, img in images.items():
            W, H = img["width"], img["height"]
            yolo_lines: List[str] = []
            for ann in anns_by_img.get(img_id, []):
                x, y, w, h = ann["bbox"]
                # COCO [x_topleft, y_topleft, w, h] → YOLO [cx, cy, w, h] 归一化
                cx = (x + w / 2) / W
                cy = (y + h / 2) / H
                wn = w / W
                hn = h / H
                # clamp 到 [0, 1] 防越界
                cx, cy = max(0.0, min(1.0, cx)), max(0.0, min(1.0, cy))
                wn, hn = max(0.0, min(1.0, wn)), max(0.0, min(1.0, hn))
                yolo_lines.append(f"{ann['category_id']} {cx:.6f} {cy:.6f} {wn:.6f} {hn:.6f}")
            split_result[img["file_name"]] = yolo_lines

        result[split] = split_result
        n_pos = sum(1 for v in split_result.values() if v)
        n_total = len(split_result)
        print(f"  {split}: {n_total} 图（{n_pos} 含 bbox / {n_total - n_pos} 负样本）"
              f" — {time.time() - t0:.1f}s")

    return result


# ─────────────────────────────────────────────────────────────────
# Step 3: 合并（含 D-Fire 类 ID 翻转 = step 2 内联）
# ─────────────────────────────────────────────────────────────────
def step3_merge(
    dfire_root: Path,
    fasdd_root: Path,
    fasdd_yolo: Dict[str, Dict[str, List[str]]],
    merged_root: Path,
) -> int:
    """合并到 merged_root/<split>/{images,labels}/。

    - 加前缀避免命名冲突：`dfire__<stem>.jpg`、`fasdd__<stem>.jpg`
    - 图像用 hardlink（同文件系统零额外空间）
    - D-Fire label 读取时翻转类 ID（0↔1）
    - FASDD label 从内存映射写出
    返回总合并文件数。
    """
    print("=" * 60)
    print("Step 3: 合并到 data/merged/（含 D-Fire 类 ID 翻转）")
    print("=" * 60)

    total = 0
    for split in SPLITS:
        img_dst = merged_root / split / "images"
        lbl_dst = merged_root / split / "labels"
        img_dst.mkdir(parents=True, exist_ok=True)
        lbl_dst.mkdir(parents=True, exist_ok=True)

        n_dfire = n_fasdd = 0
        t0 = time.time()

        # ---- D-Fire ----
        dfire_imgs = dfire_root / "data" / split / "images"
        dfire_lbls = dfire_root / "data" / split / "labels"
        if dfire_imgs.exists():
            for img_file in dfire_imgs.iterdir():
                if not img_file.is_file():
                    continue
                stem = img_file.stem
                new_img = img_dst / f"dfire__{stem}{img_file.suffix}"
                new_lbl = lbl_dst / f"dfire__{stem}.txt"

                if not new_img.exists():
                    os.link(img_file, new_img)

                # 读 + 翻转类 ID + 写
                old_lbl = dfire_lbls / f"{stem}.txt"
                if old_lbl.exists():
                    flipped = []
                    for line in old_lbl.read_text().splitlines():
                        parts = line.strip().split()
                        if not parts:
                            continue
                        parts[0] = DFIRE_FLIP_MAP.get(parts[0], parts[0])
                        flipped.append(" ".join(parts))
                    new_lbl.write_text("\n".join(flipped))
                else:
                    new_lbl.write_text("")  # 负样本：空 txt

                n_dfire += 1

        # ---- FASDD ----
        fasdd_imgs = fasdd_root / "images" / split
        if fasdd_imgs.exists():
            split_yolo = fasdd_yolo.get(split, {})
            for img_file in fasdd_imgs.iterdir():
                if not img_file.is_file():
                    continue
                stem = img_file.stem
                new_img = img_dst / f"fasdd__{stem}{img_file.suffix}"
                new_lbl = lbl_dst / f"fasdd__{stem}.txt"

                if not new_img.exists():
                    os.link(img_file, new_img)

                yolo_lines = split_yolo.get(img_file.name, [])
                new_lbl.write_text("\n".join(yolo_lines))

                n_fasdd += 1

        total += n_dfire + n_fasdd
        print(f"  {split}: dfire={n_dfire}, fasdd={n_fasdd}, total={n_dfire + n_fasdd}"
              f" — {time.time() - t0:.1f}s")

    return total


# ─────────────────────────────────────────────────────────────────
# Step 4: pHash 跨集去重
# ─────────────────────────────────────────────────────────────────
def step4_phash_dedup(merged_root: Path, threshold: int = 6) -> int:
    """跨 split + 跨集 pHash 去重。

    优先级：
      1. 跨 split 重复（如 train/X 与 val/X 同图）→ 删 val/test 那个（防训练泄漏）
      2. 同 split 重复 → 删较小分辨率的
    """
    print("=" * 60)
    print(f"Step 4: pHash 去重 (汉明距离阈值 = {threshold})")
    print("=" * 60)

    from imagededup.methods import PHash
    phasher = PHash()

    all_encodings: Dict[str, str] = {}     # filename → hash
    file_to_split: Dict[str, str] = {}     # filename → split

    t0 = time.time()
    for split in SPLITS:
        img_dir = merged_root / split / "images"
        if not img_dir.exists():
            continue
        encs = phasher.encode_images(image_dir=str(img_dir))
        for fname, h in encs.items():
            all_encodings[fname] = h
            file_to_split[fname] = split
        print(f"  {split}: {len(encs)} 图 hash 完成（累计 {time.time() - t0:.1f}s）")

    print(f"  跨集查重 ({len(all_encodings)} 张图)...")
    t1 = time.time()
    duplicates = phasher.find_duplicates(
        encoding_map=all_encodings,
        max_distance_threshold=threshold,
    )
    print(f"  查重完成 — {time.time() - t1:.1f}s")

    SPLIT_PRIORITY = {"train": 0, "val": 1, "test": 2}  # 越小越优先保留

    def _file_size(fname: str) -> int:
        try:
            return (merged_root / file_to_split[fname] / "images" / fname).stat().st_size
        except (FileNotFoundError, KeyError):
            return 0

    to_remove: set = set()
    for img, dup_list in duplicates.items():
        if not dup_list or img in to_remove:
            continue
        group = ({img} | set(dup_list)) - to_remove
        if len(group) <= 1:
            continue
        # 选保留：先按 split 优先级（train > val > test），再按文件大小（大优先）
        keep = min(group, key=lambda f: (
            SPLIT_PRIORITY.get(file_to_split.get(f, "train"), 99),
            -_file_size(f),
        ))
        to_remove.update(group - {keep})

    n_removed = 0
    for f in to_remove:
        split = file_to_split.get(f)
        if not split:
            continue
        img_path = merged_root / split / "images" / f
        lbl_path = merged_root / split / "labels" / (Path(f).stem + ".txt")
        try:
            img_path.unlink(missing_ok=True)
            lbl_path.unlink(missing_ok=True)
            n_removed += 1
        except Exception as e:
            print(f"    warn: 删除 {f} 失败: {e}")

    pct = (n_removed * 100 / len(all_encodings)) if all_encodings else 0
    print(f"  共删除 {n_removed} 张重复图（占 {pct:.2f}%）")
    return n_removed


# ─────────────────────────────────────────────────────────────────
# Step 5: 写 sources.csv（自包含，从 merged/ 反推）
# ─────────────────────────────────────────────────────────────────
def step5_write_source_csv(merged_root: Path) -> int:
    """从 merged/ 现有内容反推 source 写 CSV。

    依据文件名前缀（`dfire__` / `fasdd__`）判断来源；orig_path 也按前缀拼出。
    自包含设计：不依赖 step 3 的中间数据，单独跑也能补齐。
    """
    print("=" * 60)
    print("Step 5: 写 sources.csv")
    print("=" * 60)

    csv_path = merged_root / "sources.csv"
    n = 0
    with open(csv_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["filename", "source", "split", "orig_path_hint"])
        for split in SPLITS:
            img_dir = merged_root / split / "images"
            if not img_dir.exists():
                continue
            for img_file in sorted(img_dir.iterdir()):
                if not img_file.is_file():
                    continue
                fname = img_file.name
                if fname.startswith("dfire__"):
                    src = "dfire"
                    stem = fname[len("dfire__"):]
                    orig = f"data/raw/dfire/data/{split}/images/{stem}"
                elif fname.startswith("fasdd__"):
                    src = "fasdd_cv"
                    stem = fname[len("fasdd__"):]
                    orig = f"data/raw/fasdd_cv/images/{split}/{stem}"
                else:
                    src = "unknown"
                    orig = ""
                w.writerow([fname, src, split, orig])
                n += 1
    print(f"  写入 {n} 行 → {csv_path}")
    return n


# ─────────────────────────────────────────────────────────────────
# Step 6: 生成 data.yaml
# ─────────────────────────────────────────────────────────────────
def step6_write_data_yaml(merged_root: Path, out_yaml: Path) -> None:
    print("=" * 60)
    print("Step 6: 生成 data.yaml")
    print("=" * 60)

    cfg = {
        "path": str(merged_root.resolve()),
        "train": "train/images",
        "val": "val/images",
        "test": "test/images",
        "nc": 2,
        "names": {0: "fire", 1: "smoke"},
        "sources_csv": "sources.csv",
        "generated_by": "dataset_prep.py",
        "generated_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "class_id_convention": "0=fire, 1=smoke (matches FASDD; D-Fire flipped)",
    }
    with open(out_yaml, "w") as f:
        yaml.safe_dump(cfg, f, allow_unicode=True, sort_keys=False)
    print(f"  写入 {out_yaml}")
    print(f"  数据根: {cfg['path']}")


# ─────────────────────────────────────────────────────────────────
# 汇总
# ─────────────────────────────────────────────────────────────────
def report(merged_root: Path) -> None:
    print("=" * 60)
    print("最终汇总")
    print("=" * 60)
    grand_total = 0
    for split in SPLITS:
        img_dir = merged_root / split / "images"
        lbl_dir = merged_root / split / "labels"
        if not img_dir.exists():
            continue
        n_imgs = sum(1 for f in img_dir.iterdir() if f.is_file())
        n_dfire = sum(1 for f in img_dir.iterdir() if f.is_file() and f.name.startswith("dfire__"))
        n_fasdd = sum(1 for f in img_dir.iterdir() if f.is_file() and f.name.startswith("fasdd__"))
        n_pos = sum(1 for f in lbl_dir.iterdir() if f.is_file() and f.stat().st_size > 0)
        n_neg = sum(1 for f in lbl_dir.iterdir() if f.is_file() and f.stat().st_size == 0)
        print(f"  {split:>5}: {n_imgs:>6} 图（dfire={n_dfire}, fasdd={n_fasdd}）"
              f" / labels: {n_pos} 含 bbox + {n_neg} 负样本")
        grand_total += n_imgs
    print(f"  合计: {grand_total} 图")


# ─────────────────────────────────────────────────────────────────
# 入口
# ─────────────────────────────────────────────────────────────────
def main() -> int:
    p = argparse.ArgumentParser(description="P1 数据集合并：FASDD_CV + D-Fire → data/merged/")
    p.add_argument("--data-root", default="../data", help="data/ 目录路径（相对 train/）")
    p.add_argument("--steps", default="all", help="逗号分隔步骤号 / all")
    p.add_argument("--dedup-thresh", type=int, default=6, help="pHash 汉明距离阈值")
    p.add_argument("--skip-dedup", action="store_true", help="跳过最慢的 step 4")
    args = p.parse_args()

    train_dir = Path(__file__).resolve().parent
    data_root = (train_dir / args.data_root).resolve()
    dfire_root = data_root / "raw" / "dfire"
    fasdd_root = data_root / "raw" / "fasdd_cv"
    merged_root = data_root / "merged"

    print(f"[路径] data_root  = {data_root}")
    print(f"[路径] dfire_root = {dfire_root}  (exists: {dfire_root.exists()})")
    print(f"[路径] fasdd_root = {fasdd_root}  (exists: {fasdd_root.exists()})")
    print(f"[路径] merged     = {merged_root}\n")

    if args.steps == "all":
        steps = [1, 2, 3, 4, 5, 6]
    else:
        steps = [int(s) for s in args.steps.split(",")]
    if args.skip_dedup and 4 in steps:
        steps.remove(4)
    print(f"[执行] 步骤: {steps}\n")

    fasdd_yolo: Dict[str, Dict[str, List[str]]] = {}
    if 1 in steps or 3 in steps:
        fasdd_yolo = step1_load_fasdd_coco(fasdd_root)

    if 3 in steps:
        step3_merge(dfire_root, fasdd_root, fasdd_yolo, merged_root)

    if 4 in steps:
        step4_phash_dedup(merged_root, threshold=args.dedup_thresh)

    if 5 in steps:
        step5_write_source_csv(merged_root)

    if 6 in steps:
        step6_write_data_yaml(merged_root, train_dir / "data.yaml")

    print()
    report(merged_root)
    return 0


if __name__ == "__main__":
    sys.exit(main())
