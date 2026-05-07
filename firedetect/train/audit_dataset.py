"""audit_dataset.py — 深度审计 firedetect/data/merged 的数据集质量。

检查项：
  1. 标签格式扫描 + bbox 几何 + 类分布（每行 5 字段、cls∈{0,1}、coords∈[0,1]、无 0 尺寸）
  2. bbox 尺寸分桶（针对'小火'痛点：<0.1% / 0.1-0.5% / 0.5-2% / 2-10% / >10%）
  3. FASDD 文件名 vs 类 ID 一致性（金钥匙——D-Fire 翻转 + COCO→YOLO 转换是否对齐）
  4. 图像-标签配对完整性
  5. source 分布（dedup 后）
  6. 图像可读性 + 分辨率抽样

用法:
    python audit_dataset.py
"""
from __future__ import annotations

import sys
import time
import random
from pathlib import Path
from collections import defaultdict, Counter

ROOT = Path(__file__).resolve().parents[1] / "data" / "merged"
SPLITS = ("train", "val", "test")


def banner(t: str) -> None:
    print()
    print("=" * 78)
    print(f"  {t}")
    print("=" * 78)


# === 1. 格式 + bbox 几何 + 类分布 ===
def audit_labels():
    banner("1. 标签格式扫描 + bbox 几何 + 类分布")
    parse_errors = []
    out_of_bounds = []
    bbox_per_split = defaultdict(list)  # split → [(cls, area, aspect)]
    cls_count = defaultdict(Counter)
    img_with_label = defaultdict(int)
    img_no_label = defaultdict(int)

    for split in SPLITS:
        lbl_dir = ROOT / split / "labels"
        if not lbl_dir.exists():
            continue
        for txt in lbl_dir.iterdir():
            if txt.suffix != ".txt":
                continue
            try:
                content = txt.read_text()
            except Exception as e:
                parse_errors.append(f"{txt}: {e}")
                continue
            n_bbox = 0
            for ln, line in enumerate(content.splitlines(), 1):
                line = line.strip()
                if not line:
                    continue
                parts = line.split()
                if len(parts) != 5:
                    parse_errors.append(f"{txt.name}:{ln} 字段数={len(parts)}: {line}")
                    continue
                try:
                    cls = int(parts[0])
                    cx, cy, w, h = (float(x) for x in parts[1:])
                except ValueError:
                    parse_errors.append(f"{txt.name}:{ln} 数值解析失败: {line}")
                    continue
                if cls not in (0, 1):
                    parse_errors.append(f"{txt.name}:{ln} 类 ID 异常: {cls}")
                    continue
                if not (0 <= cx <= 1 and 0 <= cy <= 1 and 0 <= w <= 1 and 0 <= h <= 1):
                    out_of_bounds.append((txt.name, cls, cx, cy, w, h))
                    continue
                if w == 0 or h == 0:
                    parse_errors.append(f"{txt.name}:{ln} 0 尺寸 bbox")
                    continue
                area = w * h
                aspect = max(w, h) / min(w, h)
                bbox_per_split[split].append((cls, area, aspect))
                cls_count[split][cls] += 1
                n_bbox += 1
            if n_bbox > 0:
                img_with_label[split] += 1
            else:
                img_no_label[split] += 1

    print(f"\n[格式错误] 总数: {len(parse_errors)}")
    for e in parse_errors[:10]:
        print(f"  {e}")
    if len(parse_errors) > 10:
        print(f"  ... 还有 {len(parse_errors) - 10} 条")

    print(f"\n[bbox 出界 (cx/cy/w/h ∉ [0,1])] 总数: {len(out_of_bounds)}")
    for ob in out_of_bounds[:5]:
        print(f"  {ob}")

    print("\n[类分布]")
    print(f"  {'split':<8}{'fire':<10}{'smoke':<10}{'fire/smoke':<12}{'含 bbox 图':<12}{'负样本图':<10}")
    for split in SPLITS:
        c = cls_count[split]
        ratio = (c[0] / c[1]) if c[1] else 0
        print(f"  {split:<8}{c[0]:<10}{c[1]:<10}{ratio:<12.3f}{img_with_label[split]:<12}{img_no_label[split]:<10}")

    return bbox_per_split


# === 2. bbox 尺寸分桶 ===
def bbox_size_dist(bps):
    banner("2. bbox 尺寸分布（针对'小火'痛点）")
    print(f"  {'split':<8}{'class':<8}{'<0.1%':<10}{'0.1-0.5%':<12}{'0.5-2%':<10}{'2-10%':<10}{'>10%':<10}{'total':<10}")
    for split in SPLITS:
        for cls in (0, 1):
            cn = "fire" if cls == 0 else "smoke"
            sizes = [a for c, a, _ in bps[split] if c == cls]
            if not sizes:
                continue
            b = [
                sum(1 for a in sizes if a < 0.001),
                sum(1 for a in sizes if 0.001 <= a < 0.005),
                sum(1 for a in sizes if 0.005 <= a < 0.02),
                sum(1 for a in sizes if 0.02 <= a < 0.10),
                sum(1 for a in sizes if a >= 0.10),
            ]
            print(f"  {split:<8}{cn:<8}{b[0]:<10}{b[1]:<12}{b[2]:<10}{b[3]:<10}{b[4]:<10}{len(sizes):<10}")

    extreme = sum(1 for s in SPLITS for _, _, a in bps[s] if a > 15)
    print(f"\n[极端长宽比 > 15:1] {extreme} 个（标注质量警报阈值）")


# === 3. FASDD 文件名 vs 类 ID 一致性 ===
def verify_fasdd_naming():
    banner("3. FASDD 文件名 vs 类 ID 一致性（金钥匙）")
    print("  FASDD 命名规则:")
    print("    fire_*               → 应只含 cls 0")
    print("    smoke_*              → 应只含 cls 1")
    print("    bothFireAndSmoke_*   → 应同时有 0 和 1")
    print("    neitherFireNorSmoke_*→ 应空 txt")
    print()

    stats = defaultdict(lambda: {"ok": 0, "bad": 0, "ex": []})
    for split in SPLITS:
        lbl_dir = ROOT / split / "labels"
        if not lbl_dir.exists():
            continue
        for txt in lbl_dir.iterdir():
            if not txt.name.startswith("fasdd__"):
                continue
            stem = txt.stem.replace("fasdd__", "")
            cls_set = set()
            try:
                for line in txt.read_text().splitlines():
                    p = line.strip().split()
                    if p:
                        cls_set.add(int(p[0]))
            except Exception:
                continue

            if stem.startswith("fire_"):
                k, ok = "fire", cls_set == {0}
            elif stem.startswith("smoke_"):
                k, ok = "smoke", cls_set == {1}
            elif stem.startswith("bothFireAndSmoke_"):
                k, ok = "bothFireAndSmoke", cls_set == {0, 1}
            elif stem.startswith("neitherFireNorSmoke_"):
                k, ok = "neitherFireNorSmoke", len(cls_set) == 0
            else:
                continue

            if ok:
                stats[k]["ok"] += 1
            else:
                stats[k]["bad"] += 1
                if len(stats[k]["ex"]) < 3:
                    stats[k]["ex"].append((txt.name, sorted(cls_set)))

    total_ok = total_bad = 0
    print(f"  {'前缀':<22}{'正确':<10}{'错误':<10}错误示例（实际类）")
    for k in ("fire", "smoke", "bothFireAndSmoke", "neitherFireNorSmoke"):
        s = stats[k]
        ex_str = ""
        if s["ex"]:
            first = s["ex"][0]
            ex_str = f"{first[0]} cls={first[1]}"
        print(f"  {k:<22}{s['ok']:<10}{s['bad']:<10}{ex_str}")
        total_ok += s["ok"]
        total_bad += s["bad"]

    if total_ok + total_bad == 0:
        print("  (无 fasdd 数据)")
    else:
        acc = total_ok * 100 / (total_ok + total_bad)
        print(f"\n  合计: 正确 {total_ok} / 错误 {total_bad} / 准确率 {acc:.2f}%")


# === 4. 配对完整性 ===
def pair_check():
    banner("4. 图像-标签配对完整性")
    for split in SPLITS:
        img_dir = ROOT / split / "images"
        lbl_dir = ROOT / split / "labels"
        if not img_dir.exists() or not lbl_dir.exists():
            continue
        img_stems = {f.stem for f in img_dir.iterdir() if f.is_file()}
        lbl_stems = {f.stem for f in lbl_dir.iterdir() if f.is_file()}
        only_img = img_stems - lbl_stems
        only_lbl = lbl_stems - img_stems
        common = img_stems & lbl_stems
        print(f"  {split:<8}: 配对={len(common)}, 仅图={len(only_img)}, 仅标={len(only_lbl)}")
        for f in list(only_img)[:2]:
            print(f"    only_img: {f}")
        for f in list(only_lbl)[:2]:
            print(f"    only_lbl: {f}")


# === 5. source 分布 ===
def source_dist():
    banner("5. source 分布（dedup 后）")
    print(f"  {'split':<8}{'dfire':<14}{'fasdd':<14}{'other':<8}{'total':<8}")
    grand = {"dfire": 0, "fasdd": 0, "total": 0}
    for split in SPLITS:
        img_dir = ROOT / split / "images"
        if not img_dir.exists():
            continue
        nd = nf = no = 0
        for f in img_dir.iterdir():
            if not f.is_file():
                continue
            if f.name.startswith("dfire__"):
                nd += 1
            elif f.name.startswith("fasdd__"):
                nf += 1
            else:
                no += 1
        tot = nd + nf + no
        grand["dfire"] += nd
        grand["fasdd"] += nf
        grand["total"] += tot
        print(f"  {split:<8}{nd:>5} ({nd*100/max(tot,1):4.1f}%)  "
              f"{nf:>5} ({nf*100/max(tot,1):4.1f}%)  {no:<8}{tot:<8}")
    print(f"  {'合计':<8}{grand['dfire']:>5} ({grand['dfire']*100/max(grand['total'],1):4.1f}%)  "
          f"{grand['fasdd']:>5} ({grand['fasdd']*100/max(grand['total'],1):4.1f}%)")


# === 6. 图像可读性 + 分辨率 ===
def image_sanity():
    banner("6. 图像可读性 + 分辨率抽样（每 split 随机 200 张）")
    try:
        from PIL import Image
    except ImportError:
        print("  PIL 未装，跳过")
        return

    random.seed(42)
    for split in SPLITS:
        img_dir = ROOT / split / "images"
        if not img_dir.exists():
            continue
        all_imgs = list(img_dir.iterdir())
        sample = random.sample(all_imgs, min(200, len(all_imgs)))
        bad = []
        sizes = []
        for p in sample:
            try:
                with Image.open(p) as im:
                    sizes.append(im.size + (p.name,))
            except Exception as e:
                bad.append((p.name, str(e)[:40]))
        if not sizes:
            continue
        ws = sorted(s[0] for s in sizes)
        hs = sorted(s[1] for s in sizes)
        avg_w = sum(ws) / len(ws)
        avg_h = sum(hs) / len(hs)
        med_w = ws[len(ws) // 2]
        med_h = hs[len(hs) // 2]
        min_short = min(min(w, h) for w, h, _ in sizes)
        print(f"  {split:<8}: 抽 {len(sample)} 张, {len(bad)} 不可读")
        print(f"           avg={avg_w:.0f}×{avg_h:.0f}, median={med_w}×{med_h}, 最短边={min_short}")
        for b in bad[:3]:
            print(f"    bad: {b[0]} ({b[1]})")


def main() -> int:
    t0 = time.time()
    bps = audit_labels()
    bbox_size_dist(bps)
    verify_fasdd_naming()
    pair_check()
    source_dist()
    image_sanity()
    print(f"\n[审计完成，用时 {time.time()-t0:.1f}s]")
    return 0


if __name__ == "__main__":
    sys.exit(main())
