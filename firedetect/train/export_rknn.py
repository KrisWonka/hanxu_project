"""best.pt → best.onnx → fire.rknn（INT8 PTQ）。

仅在 krix-su（x86 Linux）上跑，需 rknn-toolkit2（不能在 ARM 板上跑）。
校准集从训练集随机抽 200 张，含小火（bbox 面积 < 1%）样本占比 ≥ 30%
（这点对解决同事的小火误报问题很关键，量化校准要见过小火分布）。
"""
from __future__ import annotations
import argparse


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pt", required=True, help="ultralytics best.pt 路径")
    parser.add_argument("--out", default="../rk3588/model/fire.rknn")
    parser.add_argument("--imgsz", type=int, default=960)
    parser.add_argument("--platform", default="rk3588")
    parser.add_argument("--quant", choices=["fp16", "i8", "u8"], default="i8")
    parser.add_argument("--calib-set", default="../data/merged/images/train")
    parser.add_argument("--calib-num", type=int, default=200)
    parser.add_argument("--calib-small-fire-ratio", type=float, default=0.3,
                        help="校准集中 bbox 面积 <1%% 样本的强制最低占比")
    args = parser.parse_args()

    # TODO P2:
    # 1. ultralytics: YOLO(args.pt).export(format="onnx", imgsz=args.imgsz, opset=12)
    # 2. rknn-toolkit2:
    #    rknn = RKNN()
    #    rknn.config(mean_values=[[0,0,0]], std_values=[[255,255,255]],
    #                target_platform=args.platform)
    #    rknn.load_onnx(model=onnx_path)
    #    rknn.build(do_quantization=(args.quant=="i8"), dataset=calib_list_txt)
    #    rknn.export_rknn(args.out)
    print("export_rknn stub — implement in P2")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
