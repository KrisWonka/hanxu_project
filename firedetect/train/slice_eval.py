"""slice_eval.py — 按 bbox 尺寸 + source 切片评估模型 mAP/recall。

直接回应 PLAN §6 P2 的"小火召回 ≥ 70%"验收，比 ultralytics 默认的整体 mAP 更细。
为什么需要：整体 mAP 0.84 看着好，但可能是大火（>10%）拉的，小火（<0.5%）召回可能很低。

输出维度：
- 按 bbox 尺寸分桶（<0.1% / 0.1-0.5% / 0.5-2% / 2-10% / >10%）
- 按 source（dfire / fasdd）
- source × size 双重切片

用法:
    python slice_eval.py [--model best.pt] [--split test] [--imgsz 960] [--conf 0.25]
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from collections import defaultdict


def iou_xyxy(b1, b2):
    x1 = max(b1[0], b2[0])
    y1 = max(b1[1], b2[1])
    x2 = min(b1[2], b2[2])
    y2 = min(b1[3], b2[3])
    inter = max(0, x2 - x1) * max(0, y2 - y1)
    a1 = (b1[2] - b1[0]) * (b1[3] - b1[1])
    a2 = (b2[2] - b2[0]) * (b2[3] - b2[1])
    union = a1 + a2 - inter
    return inter / max(union, 1e-9)


def yolo_to_xyxy(cx, cy, w, h, W, H):
    return [(cx - w / 2) * W, (cy - h / 2) * H, (cx + w / 2) * W, (cy + h / 2) * H]


def size_bucket(area_norm):
    if area_norm < 0.001:
        return "<0.1%"
    if area_norm < 0.005:
        return "0.1-0.5%"
    if area_norm < 0.02:
        return "0.5-2%"
    if area_norm < 0.10:
        return "2-10%"
    return ">10%"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", required=True, help="best.pt 路径")
    parser.add_argument("--data-root", default="../data/merged", help="merged/ 目录")
    parser.add_argument("--split", choices=["train", "val", "test"], default="test")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--iou-thresh", type=float, default=0.5,
                        help="IoU 命中阈值（≥ 这个值算 TP）")
    parser.add_argument("--small-target-thresh", type=float, default=0.005,
                        help="'小火'定义：bbox 面积占比 < 此值（默认 0.5%）")
    args = parser.parse_args()

    train_dir = Path(__file__).resolve().parent
    data_root = (train_dir / args.data_root).resolve()

    from ultralytics import YOLO
    from PIL import Image

    print(f"[加载 {args.split} set ground-truth...]")
    img_dir = data_root / args.split / "images"
    lbl_dir = data_root / args.split / "labels"
    test_imgs = list(img_dir.iterdir())
    print(f"  共 {len(test_imgs)} 张图")

    gt_by_img = {}
    for img_path in test_imgs:
        try:
            with Image.open(img_path) as im:
                W, H = im.size
        except Exception:
            continue
        lbl = lbl_dir / (img_path.stem + ".txt")
        gts = []
        if lbl.exists():
            for line in lbl.read_text().splitlines():
                p = line.strip().split()
                if len(p) != 5:
                    continue
                cls = int(p[0])
                cx, cy, w, h = map(float, p[1:])
                if w == 0 or h == 0:
                    continue
                gts.append((cls, yolo_to_xyxy(cx, cy, w, h, W, H), w * h))
        gt_by_img[img_path.name] = gts

    print("[加载模型 + 跑推理...]")
    model = YOLO(args.model)
    t0 = time.time()
    BATCH = 256
    preds_by_img = {}
    for i in range(0, len(test_imgs), BATCH):
        batch_paths = test_imgs[i:i + BATCH]
        results = model.predict(
            source=[str(p) for p in batch_paths],
            imgsz=args.imgsz, conf=args.conf, iou=0.45,
            verbose=False, device=0,
        )
        # 关键：用 zip 而非 r.path，因为 ultralytics 在批量模式下会把 r.path 改成 image0/1/2...
        for batch_p, r in zip(batch_paths, results):
            name = batch_p.name
            preds = []
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    preds.append((
                        int(b.cls.item()),
                        b.xyxy.tolist()[0],
                        float(b.conf.item()),
                    ))
            preds_by_img[name] = preds
        if i % 1024 == 0:
            print(f"  inference {i}/{len(test_imgs)} ({time.time()-t0:.1f}s)")
    print(f"  完成，{time.time()-t0:.1f}s")

    print("[匹配 GT vs Pred...]")
    bucket_stats = defaultdict(lambda: {"tp": 0, "fn": 0})

    for img_name, gts in gt_by_img.items():
        if img_name.startswith("dfire__"):
            source = "dfire"
        elif img_name.startswith("fasdd__"):
            source = "fasdd"
        else:
            source = "other"
        preds = preds_by_img.get(img_name, [])
        used = set()
        for gt_cls, gt_box, gt_area in gts:
            best_iou = 0
            best_idx = -1
            for i, (p_cls, p_box, _) in enumerate(preds):
                if i in used or p_cls != gt_cls:
                    continue
                iou = iou_xyxy(gt_box, p_box)
                if iou > best_iou:
                    best_iou = iou
                    best_idx = i
            key = (source, gt_cls, size_bucket(gt_area))
            if best_iou >= args.iou_thresh:
                used.add(best_idx)
                bucket_stats[key]["tp"] += 1
            else:
                bucket_stats[key]["fn"] += 1

    print("\n" + "=" * 78)
    print(f"  切片评估报告  |  model={Path(args.model).name}  split={args.split}  imgsz={args.imgsz}")
    print("=" * 78)

    SIZES = ["<0.1%", "0.1-0.5%", "0.5-2%", "2-10%", ">10%"]
    SOURCES = ["dfire", "fasdd"]
    CLASSES = [(0, "fire"), (1, "smoke")]

    print("\n>>> 按 bbox 尺寸切片（recall = TP / (TP+FN)）<<<")
    print(f"  {'size':<10}{'fire_tp':<10}{'fire_fn':<10}{'fire_R':<10}{'smoke_tp':<10}{'smoke_fn':<10}{'smoke_R':<10}")
    for sz in SIZES:
        line = f"  {sz:<10}"
        for cls, _ in CLASSES:
            tp = sum(bucket_stats[(s, cls, sz)]["tp"] for s in SOURCES)
            fn = sum(bucket_stats[(s, cls, sz)]["fn"] for s in SOURCES)
            r = tp / max(tp + fn, 1)
            line += f"{tp:<10}{fn:<10}{r:<10.4f}"
        print(line)

    print("\n>>> 按 source 切片（合并所有尺寸）<<<")
    print(f"  {'source':<10}{'cls':<8}{'tp':<8}{'fn':<8}{'recall':<10}")
    for src in SOURCES:
        for cls, cname in CLASSES:
            tp = sum(bucket_stats[(src, cls, sz)]["tp"] for sz in SIZES)
            fn = sum(bucket_stats[(src, cls, sz)]["fn"] for sz in SIZES)
            r = tp / max(tp + fn, 1)
            print(f"  {src:<10}{cname:<8}{tp:<8}{fn:<8}{r:<10.4f}")

    print("\n>>> 小火（fire <0.5%）双重切片：source × size <<<")
    print(f"  {'source':<10}{'<0.1%':<22}{'0.1-0.5%':<22}")
    for src in SOURCES:
        line = f"  {src:<10}"
        for sz in ["<0.1%", "0.1-0.5%"]:
            tp = bucket_stats[(src, 0, sz)]["tp"]
            fn = bucket_stats[(src, 0, sz)]["fn"]
            r = tp / max(tp + fn, 1)
            line += f"{tp}/{tp+fn} (R={r:.3f})        "
        print(line)

    # PLAN 验收
    fire_small_tp = sum(bucket_stats[(s, 0, sz)]["tp"]
                        for s in SOURCES for sz in SIZES
                        if (sz == "<0.1%") or (sz == "0.1-0.5%"))
    fire_small_fn = sum(bucket_stats[(s, 0, sz)]["fn"]
                        for s in SOURCES for sz in SIZES
                        if (sz == "<0.1%") or (sz == "0.1-0.5%"))
    small_recall = fire_small_tp / max(fire_small_tp + fire_small_fn, 1)
    target = 0.70
    status = "✓ 达标" if small_recall >= target else "✗ 未达标"
    print(f"\n>>> PLAN §6 P2 验收对照 <<<")
    print(f"  小火（bbox 面积 <0.5%）召回: {small_recall:.4f} (目标 ≥ {target})  {status}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
