"""ultralytics YOLOv8n 训练入口（P2 baseline）。

默认配置：
- 模型：yolov8n.pt（COCO 预训练权重起步）
- 输入：960×960
- batch：自动（GPU 显存允许的最大）
- epochs：100（开早停）
- 数据：data.yaml
- 小目标 loss 加权 ×2–3（针对 bbox 面积占比 < 1% 的样本，对应同事痛点）
"""
from __future__ import annotations
import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", default="data.yaml")
    parser.add_argument("--model", default="yolov8n.pt")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--batch", type=int, default=-1, help="-1 = ultralytics auto")
    parser.add_argument("--project", default="runs/train")
    parser.add_argument("--name", default="firedetect_v1")
    parser.add_argument("--device", default="0", help="GPU id（krix-su 上 5090 = 0）")
    parser.add_argument("--small-obj-weight", type=float, default=2.5,
                        help="bbox 面积占比 <1%% 样本的 loss 权重倍数")
    parser.add_argument("--mosaic", type=float, default=1.0)
    parser.add_argument("--copy-paste", type=float, default=0.3,
                        help="copy-paste 增广比例，强迫模型见多尺度小火")
    args = parser.parse_args()

    # TODO P2:
    # from ultralytics import YOLO
    # model = YOLO(args.model)
    # model.train(
    #     data=args.data, imgsz=args.imgsz, epochs=args.epochs,
    #     batch=args.batch, project=args.project, name=args.name,
    #     device=args.device, mosaic=args.mosaic, copy_paste=args.copy_paste,
    #     # small-obj 加权: 用自定义 sampler 或修改 loss
    # )
    print("train_yolo stub — implement in P2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
