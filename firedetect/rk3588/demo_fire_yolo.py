"""E2E 火焰检测 demo CLI（在 Orange Pi 5 Pro 上跑）。

用法：
  python demo_fire_yolo.py --source 0                                  # USB 摄像头
  python demo_fire_yolo.py --source rtsp://user:pass@cam-ip/stream     # 网络摄像头
  python demo_fire_yolo.py --source video.mp4 --model model/fire.rknn
  python demo_fire_yolo.py --dummy                                      # P0 烟雾测试
"""
from __future__ import annotations
import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(description="Fire/smoke detection on Orange Pi 5 Pro")
    parser.add_argument("--source", default=None, help="0 / rtsp:// / 视频文件路径")
    parser.add_argument("--model", default="model/fire.rknn", help="RKNN 模型路径")
    parser.add_argument("--out", default="fire_events.jsonl", help="事件输出 jsonl 路径")
    parser.add_argument("--imgsz", type=int, default=960, help="推理输入尺寸（与训练对齐）")
    parser.add_argument("--conf", type=float, default=0.45, help="fire 置信度门限")
    parser.add_argument("--dummy", action="store_true",
                        help="P0 验证：不加载模型 / 不读视频，输出空事件流")
    args = parser.parse_args()

    if args.dummy:
        Path(args.out).write_text("")
        print(f"[dummy] wrote empty event stream to {args.out}")
        return 0

    # 真实推理路径在 P2/P3 接通：
    #   from rknnlite_model import RknnliteModel
    #   from fire_detector import FireDetector
    #   model = RknnliteModel(args.model, img_size=(args.imgsz, args.imgsz), obj_thresh=args.conf)
    #   detector = FireDetector()
    #   for frame in stream(args.source):
    #       dets = model.detect(frame)
    #       events = detector.process(frame, dets)
    #       for ev in events:
    #           jsonl_writer.write(ev.to_jsonl() + "\n")
    print("real inference not yet implemented (P2/P3 pending)", file=sys.stderr)
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
