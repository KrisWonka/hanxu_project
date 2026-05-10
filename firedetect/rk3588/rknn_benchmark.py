"""rknn_benchmark.py — 在 RK3588 板上跑连续推理统计真实 FPS。

用法（在 Orange Pi 5 Pro 上跑）：
    python3 rknn_benchmark.py --model fire.rknn --imgsz 960 --duration 60

参数：
  --core-mask  NPU 核心掩码：1=core0 / 2=core1 / 4=core2 / 7=三核全开（默认）
  --source     可选测试视频；不给就用随机像素（纯算力测试）
  --warmup     warmup 推理次数（默认 10）
"""
from __future__ import annotations
import argparse
import sys
import time
from pathlib import Path

import numpy as np

try:
    from rknnlite.api import RKNNLite
except ImportError:
    print("ERROR: rknnlite 未装。Orange Pi 上需要：")
    print("  从 https://github.com/airockchip/rknn-toolkit2 release 找 rknn_toolkit_lite2 wheel")
    sys.exit(1)

try:
    import cv2
except ImportError:
    cv2 = None


CORE_MASK_MAP = {
    0: RKNNLite.NPU_CORE_AUTO,
    1: RKNNLite.NPU_CORE_0,
    2: RKNNLite.NPU_CORE_1,
    4: RKNNLite.NPU_CORE_2,
    7: RKNNLite.NPU_CORE_0_1_2,
}


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True, help=".rknn 文件路径")
    p.add_argument("--imgsz", type=int, default=960)
    p.add_argument("--duration", type=int, default=60, help="跑多久（秒）")
    p.add_argument("--warmup", type=int, default=10)
    p.add_argument("--core-mask", type=int, default=7, choices=[0, 1, 2, 4, 7],
                   help="0=auto, 1=core0, 2=core1, 4=core2, 7=三核全开")
    p.add_argument("--source", default=None, help="测试视频路径，留空用随机像素")
    args = p.parse_args()

    print(f"=== rknn_benchmark ===")
    print(f"  model:       {args.model}")
    print(f"  imgsz:       {args.imgsz}")
    print(f"  duration:    {args.duration}s")
    print(f"  core-mask:   {args.core_mask}")
    print(f"  source:      {args.source or '(random pixels)'}")

    if not Path(args.model).exists():
        print(f"ERROR: model not found: {args.model}")
        return 1

    rknn = RKNNLite()
    print("\n[1/3] 加载 .rknn 模型...")
    if rknn.load_rknn(args.model) != 0:
        print("ERROR: load_rknn failed")
        return 2

    print("[2/3] 初始化 runtime...")
    if rknn.init_runtime(core_mask=CORE_MASK_MAP[args.core_mask]) != 0:
        print("ERROR: init_runtime failed")
        return 3

    # 数据源
    cap = None
    if args.source:
        if cv2 is None:
            print("ERROR: cv2 未装但指定了 --source")
            return 4
        cap = cv2.VideoCapture(args.source)
        if not cap.isOpened():
            print(f"ERROR: 无法打开 {args.source}")
            return 5

    def get_frame():
        if cap is not None:
            ok, img = cap.read()
            if not ok:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)  # loop
                ok, img = cap.read()
            if cv2 is not None and img is not None:
                if img.shape[:2] != (args.imgsz, args.imgsz):
                    img = cv2.resize(img, (args.imgsz, args.imgsz))
            return img
        return np.random.randint(0, 255, (args.imgsz, args.imgsz, 3), dtype=np.uint8)

    print(f"[3/3] warmup {args.warmup} 次...")
    for _ in range(args.warmup):
        img = get_frame()
        if img is None:
            continue
        rknn.inference(inputs=[img])

    print(f"\n=== benchmark 开始 ({args.duration}s) ===")
    t0 = time.time()
    n = 0
    times = []
    while time.time() - t0 < args.duration:
        img = get_frame()
        if img is None:
            continue
        t1 = time.time()
        rknn.inference(inputs=[img])
        times.append(time.time() - t1)
        n += 1

    elapsed = time.time() - t0
    times = np.array(times)

    print(f"\n=== 完成 ===")
    print(f"  推理次数:   {n}")
    print(f"  实际时长:   {elapsed:.1f}s")
    print(f"  平均 FPS:   {n / elapsed:.2f}")
    print(f"\n  推理耗时分布:")
    print(f"    平均:     {times.mean() * 1000:.1f} ms")
    print(f"    p50:      {np.median(times) * 1000:.1f} ms")
    print(f"    p95:      {np.percentile(times, 95) * 1000:.1f} ms")
    print(f"    p99:      {np.percentile(times, 99) * 1000:.1f} ms")
    print(f"    max:      {times.max() * 1000:.1f} ms")

    rknn.release()
    if cap is not None:
        cap.release()
    return 0


if __name__ == "__main__":
    sys.exit(main())
