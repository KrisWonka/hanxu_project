"""export_rknn.py — best.onnx → fire.rknn (INT8 PTQ for RK3588)。

仅在 krix-su（x86 Linux）上跑，需独立 venv（rknn-toolkit2 与 ultralytics 的 torch 版本冲突）：
    python3 -m venv ~/.venv/firedetect_rknn
    ~/.venv/firedetect_rknn/bin/pip install rknn-toolkit2 numpy opencv-python-headless onnx

用法（典型）：
    ~/.venv/firedetect_rknn/bin/python export_rknn.py \\
        --onnx ../runs/.../best.onnx \\
        --out ../rk3588/model/fire.rknn \\
        --calib-set ../data/merged/train/images \\
        --calib-num 200 \\
        --calib-small-fire-ratio 0.3

校准集策略（PLAN §2 决定）：
- 从训练集随机抽 200 张
- 强制 ≥ 30% 是含小火（bbox 面积 <0.5%）的图，避免量化把小火信号 round 没了
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path
from typing import List


def build_calib_list(
    img_dir: Path,
    label_dir: Path,
    n_total: int,
    small_fire_ratio: float,
    out_txt: Path,
    small_thresh: float = 0.005,
) -> int:
    """从 img_dir 抽 n_total 张图作校准集，写到 out_txt（每行一个绝对路径）。

    保证至少 ratio*n_total 张是含小火（fire bbox 面积 < small_thresh）的图。
    """
    print(f"[校准集采样] 目标 {n_total} 张，含小火 ≥ {small_fire_ratio*100:.0f}%")
    all_imgs = [p for p in img_dir.iterdir() if p.is_file()]
    print(f"  候选池: {len(all_imgs)} 张")

    # 第一轮：扫所有 label 找含小火的图
    small_fire_imgs: List[Path] = []
    rest_imgs: List[Path] = []
    for img in all_imgs:
        lbl = label_dir / (img.stem + ".txt")
        if not lbl.exists():
            rest_imgs.append(img)
            continue
        has_small_fire = False
        try:
            for line in lbl.read_text().splitlines():
                p = line.strip().split()
                if len(p) != 5:
                    continue
                cls = int(p[0])
                if cls != 0:  # 0=fire
                    continue
                w, h = float(p[3]), float(p[4])
                if w * h < small_thresh and w > 0 and h > 0:
                    has_small_fire = True
                    break
        except Exception:
            pass
        (small_fire_imgs if has_small_fire else rest_imgs).append(img)

    print(f"  含小火图 (bbox <{small_thresh*100:.1f}%): {len(small_fire_imgs)}")

    # 决定各类各取多少
    n_small_target = int(n_total * small_fire_ratio)
    n_small_actual = min(n_small_target, len(small_fire_imgs))
    n_rest = n_total - n_small_actual

    if n_small_actual < n_small_target:
        print(f"  ⚠ 小火图不够 {n_small_target} 张，仅有 {n_small_actual} 张")

    random.seed(42)
    pick_small = random.sample(small_fire_imgs, n_small_actual)
    pick_rest = random.sample(rest_imgs, min(n_rest, len(rest_imgs)))
    picked = pick_small + pick_rest
    random.shuffle(picked)

    with open(out_txt, "w") as f:
        for p in picked:
            f.write(f"{p.resolve()}\n")
    print(f"  写入 {out_txt}: {len(picked)} 张（{n_small_actual} 小火 + {len(pick_rest)} 其他）")
    return len(picked)


def main() -> int:
    p = argparse.ArgumentParser(description="best.onnx → fire.rknn INT8 (RK3588)")
    p.add_argument("--onnx", required=True, help="ultralytics 导出的 best.onnx")
    p.add_argument("--out", default="../rk3588/model/fire.rknn",
                   help="输出 rknn 路径")
    p.add_argument("--platform", default="rk3588",
                   choices=["rk3562", "rk3566", "rk3568", "rk3576", "rk3588"])
    p.add_argument("--quant", choices=["fp16", "i8", "u8"], default="i8")
    # 校准集
    p.add_argument("--calib-set", default="../data/merged/train/images",
                   help="校准集图像目录（也用 ../labels/ 自动反推 small fire 占比）")
    p.add_argument("--calib-num", type=int, default=200)
    p.add_argument("--calib-small-fire-ratio", type=float, default=0.3,
                   help="校准集中含小火的图最低占比")
    p.add_argument("--calib-list", default="/tmp/calib_list.txt",
                   help="校准列表 txt 输出路径（rknn-toolkit 需要）")
    p.add_argument("--imgsz", type=int, default=960, help="模型训练时的输入尺寸")
    p.add_argument("--mean", type=float, nargs="+", default=[0.0, 0.0, 0.0])
    p.add_argument("--std", type=float, nargs="+", default=[255.0, 255.0, 255.0])
    p.add_argument("--verbose", action="store_true")
    args = p.parse_args()

    onnx_path = Path(args.onnx).resolve()
    out_path = Path(args.out).resolve()
    calib_dir = Path(args.calib_set).resolve()
    calib_lbl_dir = calib_dir.parent / "labels"
    calib_list = Path(args.calib_list)

    if not onnx_path.exists():
        print(f"ERROR: ONNX not found: {onnx_path}")
        return 1

    out_path.parent.mkdir(parents=True, exist_ok=True)

    # === 1. 准备校准集 ===
    if args.quant == "i8":
        if not calib_dir.exists():
            print(f"ERROR: calib dir not found: {calib_dir}")
            return 1
        build_calib_list(
            calib_dir, calib_lbl_dir,
            n_total=args.calib_num,
            small_fire_ratio=args.calib_small_fire_ratio,
            out_txt=calib_list,
        )

    # === 2. 调 rknn-toolkit2 ===
    try:
        from rknn.api import RKNN
    except ImportError:
        print("ERROR: rknn-toolkit2 未装。在专用 venv 里跑：")
        print("    ~/.venv/firedetect_rknn/bin/python export_rknn.py ...")
        return 2

    rknn = RKNN(verbose=args.verbose)

    print(f"\n[1/4] config (platform={args.platform}, quant={args.quant})")
    rknn.config(
        mean_values=[args.mean],
        std_values=[args.std],
        target_platform=args.platform,
        quantized_dtype="asymmetric_quantized-8" if args.quant == "i8" else "asymmetric_quantized-u8",
        optimization_level=3,
    )

    print(f"\n[2/4] load_onnx: {onnx_path}")
    ret = rknn.load_onnx(model=str(onnx_path))
    if ret != 0:
        print(f"ERROR: load_onnx failed (ret={ret})")
        return 3

    print(f"\n[3/4] build (do_quantization={args.quant != 'fp16'})")
    ret = rknn.build(
        do_quantization=(args.quant != "fp16"),
        dataset=str(calib_list) if args.quant != "fp16" else None,
    )
    if ret != 0:
        print(f"ERROR: build failed (ret={ret})")
        return 4

    print(f"\n[4/4] export_rknn: {out_path}")
    ret = rknn.export_rknn(str(out_path))
    if ret != 0:
        print(f"ERROR: export_rknn failed (ret={ret})")
        return 5

    rknn.release()

    size_mb = out_path.stat().st_size / 1024 / 1024
    print(f"\n=== 完成 ===")
    print(f"  ONNX:  {onnx_path.stat().st_size / 1024 / 1024:.1f} MB")
    print(f"  RKNN:  {size_mb:.1f} MB")
    print(f"  压缩比: {onnx_path.stat().st_size / out_path.stat().st_size:.2f}x")
    return 0


if __name__ == "__main__":
    sys.exit(main())
