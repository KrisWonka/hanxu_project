# firedetect — 火焰/烟雾检测模块

含旭家居养老系统子模块。设计文档见 [PLAN.md](./PLAN.md)，标注规范见 [ANNOTATION_GUIDE.md](./ANNOTATION_GUIDE.md)。

## 双机分工

| 机器 | 角色 | 跑什么 |
|---|---|---|
| krix-su（RTX 5090 + Ubuntu 22.04 + CUDA 13）| 训练机 | `train/` 全部脚本 |
| Orange Pi 5 Pro（RK3588S）| 部署机 | `rk3588/` 全部脚本 |

## krix-su 上跑训练（开发者）

```bash
ssh krix-su
cd hanxu_project/firedetect/train
pip install -r requirements.txt
python dataset_prep.py        # P1: 拉公开集 + 合并去重
python train_yolo.py          # P2: 训 YOLOv8n@960
python export_rknn.py         # P2: 导出 INT8 .rknn
scp ../rk3588/model/fire.rknn pi@<orangepi-ip>:/home/pi/firedetect/rk3588/model/
```

## Orange Pi 5 Pro 上跑推理（部署）

```bash
cd ~/firedetect/rk3588
python demo_fire_yolo.py --source rtsp://user:pass@cam-ip/stream --model model/fire.rknn
```

## P0 验证（任何机器都能跑）

```bash
python -c "import firedetect"                          # 无错
python firedetect/rk3588/demo_fire_yolo.py --dummy     # 出空 fire_events.jsonl
```

## 阶段进度

详见 [PLAN.md §6](./PLAN.md#6-实施分阶段)。

- [x] P0 骨架 — 当前
- [ ] P1 数据 — 待 `dataset_prep.py` 实现
- [ ] P2 baseline — 待 `train_yolo.py` + `export_rknn.py`
- [ ] P3 后处理 — 待 `fire_detector.py`
- [ ] P4–P6 见 PLAN
