"""train_yolo.py — P2 baseline 训练 YOLOv8n@960。

针对同事痛点的小目标友好配置：
- 输入 960×960（比默认 640 大，小火像素更多）
- mosaic + copy-paste 增广强迫模型见多尺度小火
- fliplr=0.5（左右翻转 OK）、flipud=0（火不会上下颠倒）
- patience=20（早停，节省时间）

运行（krix-su 上）：
    cd ~/hanxu_project/firedetect/train
    ~/.venv/firedetect/bin/python train_yolo.py
后台：
    nohup ~/.venv/firedetect/bin/python train_yolo.py > ~/train.log 2>&1 < /dev/null &
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    p = argparse.ArgumentParser(description="P2 baseline: YOLOv8n@960 训练")
    p.add_argument("--data", default="data.yaml")
    p.add_argument("--model", default="yolov8n.pt", help="预训练权重（COCO）")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--batch", type=int, default=-1, help="-1 = ultralytics auto")
    p.add_argument("--project", default="runs/train")
    p.add_argument("--name", default="firedetect_v1")
    p.add_argument("--device", default="0")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--patience", type=int, default=20, help="早停 patience（epoch 数）")
    p.add_argument("--cache", default="False", help="ram / disk / False")
    p.add_argument("--resume", action="store_true")
    p.add_argument("--mosaic", type=float, default=1.0)
    p.add_argument("--copy-paste", type=float, default=0.3,
                   help="copy-paste 增广概率，强迫模型见多尺度小火")
    p.add_argument("--scale", type=float, default=0.5, help="mosaic 内尺度变化范围")
    p.add_argument("--fliplr", type=float, default=0.5)
    p.add_argument("--flipud", type=float, default=0.0, help="火不会上下颠倒")
    # loss 权重（ultralytics 默认值，留参可调）
    p.add_argument("--box", type=float, default=7.5)
    p.add_argument("--cls", type=float, default=0.5)
    p.add_argument("--dfl", type=float, default=1.5)
    args = p.parse_args()

    # 处理 cache 参数：ultralytics 期望 bool 或 'ram'/'disk'
    cache_value: object
    if args.cache.lower() in ("false", "0", "no"):
        cache_value = False
    elif args.cache.lower() == "ram":
        cache_value = "ram"
    elif args.cache.lower() == "disk":
        cache_value = "disk"
    else:
        cache_value = False

    print(f"=" * 60)
    print(f"P2 baseline 训练")
    print(f"=" * 60)
    print(f"  data:      {args.data}")
    print(f"  model:     {args.model}")
    print(f"  imgsz:     {args.imgsz}")
    print(f"  epochs:    {args.epochs} (patience={args.patience})")
    print(f"  batch:     {args.batch}")
    print(f"  device:    {args.device}")
    print(f"  cache:     {cache_value}")
    print(f"  project:   {args.project}/{args.name}")
    print(f"  增广:      mosaic={args.mosaic}, copy_paste={args.copy_paste}, "
          f"scale={args.scale}, fliplr={args.fliplr}, flipud={args.flipud}")
    print(f"  loss 权重: box={args.box}, cls={args.cls}, dfl={args.dfl}")
    print()

    from ultralytics import YOLO

    model = YOLO(args.model)

    results = model.train(
        data=args.data,
        imgsz=args.imgsz,
        epochs=args.epochs,
        batch=args.batch,
        project=args.project,
        name=args.name,
        device=args.device,
        workers=args.workers,
        patience=args.patience,
        cache=cache_value,
        resume=args.resume,
        # 增广
        mosaic=args.mosaic,
        copy_paste=args.copy_paste,
        scale=args.scale,
        fliplr=args.fliplr,
        flipud=args.flipud,
        # loss 权重
        box=args.box,
        cls=args.cls,
        dfl=args.dfl,
        # 其他
        plots=True,           # 训完出 PR 曲线、confusion matrix 等图
        save=True,
        save_period=10,       # 每 10 epoch 存一次 checkpoint
        verbose=True,
    )

    print()
    print("=" * 60)
    print("训练完成")
    print("=" * 60)
    if hasattr(model, "trainer"):
        print(f"best model: {model.trainer.best}")
        print(f"last model: {model.trainer.last}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
