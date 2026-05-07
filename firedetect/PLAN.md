# 火焰识别模块 PLAN

> 含旭家居养老系统 · 火焰/烟雾检测子模块
> 部署目标板：Orange Pi 5 Pro（Rockchip RK3588S，ARM64，6 TOPS NPU）
> 推理 runtime：RKNN（与现有 `handwaving/huishou/rk3588/` 同款链路）
> 创建日期：2026-05-06

---

## 1. 目标与边界

- **任务**：室内监控视角下检测**早期火焰 + 烟雾**两类，实时输出告警事件
- **场景**：养老住宅室内 —— 客厅、厨房、卧室、走廊。监控摄像头 RGB 视频流（IPC RTSP 或 USB UVC）
- **关键 KPI**（自定，待客户确认）
  - 早期小火苗（bbox 面积占比 ≥ 0.5%）召回 ≥ 90%
  - 室内 24h 不间断运行误报 ≤ 1 次/天
  - 端到端延迟 ≤ 500ms（采集→告警）
  - NPU 推理：v8n@960 目标 ≤ 50 ms/帧（≥ 20 FPS）；v8n@1280 目标 ≤ 80 ms/帧（≥ 12 FPS）
- **不做**：户外大范围野火、无人机航拍、卫星遥感（这些是 FASDD_UAV / FASDD_RS 的事，我们只用 `FASDD_CV` 子集）

---

## 2. 模型选型

**主力**：**YOLOv8n**（fallback / plan B：YOLOv11n）

> 2026-05-06 决策反转：原选 v11n，结合 RKNN 平台实测后改 v8n。理由见下。

**为什么是 v8n 而不是 v11n（RKNN 平台特定原因）**：
- RKNN 工具链对 v8 支持最成熟：`airockchip/rknn_model_zoo` 的 v8n demo 是基线参考实现，INT8 转换 + 量化 + 后处理都跑通；v11n 虽然 2025 年加进 zoo（`examples/yolo11/`），但 C2PSA / C3k2 新模块的部分算子 NPU 加速不全，会 fallback 到 CPU
- **板上实测：RK3588 上 v11n 比 v8n 推理时间多 ~2.4 ms**，mAP 提升微小（1–2%），且 INT8 量化后差距几乎抹平
- 同事反馈"小火误报多"是核心痛点 → 解决方法是**提高输入分辨率**而非换架构。v8n 省下来的算力可以上 v8n@960 / @1280，把小火从 ~10 像素涨到 ~20 像素，效果远大于 v11 的微小架构优势
- 社区 fire/smoke 示例 v8 资源更多

**模型配置**：
- 输入：**960×960 起评，960 跑通后实测 1280×1280 是否仍能 ≥ 15 FPS**（板上 P0 阶段 benchmark 决定）
- 类 head：2 类（`fire` / `smoke`）
- 量化：INT8 PTQ，校准集从训练集随机抽 200 张（含小火样本占比 ≥ 30%）
- 训练时给小目标 loss 加权（bbox 面积占比 < 1% 的样本权重 ×2–3）
- **小目标增强**：训练阶段 mosaic + copy-paste augmentation 强迫模型见多尺度小火

**不选的方案**：
- v11n：见上，留作 v8n 基线达不到指标时的备选
- YOLO-seg：分割是好，但 RKNN 上 mask 后处理慢，2x 推理时间换不来真实增益
- 双流 RGB+IR：硬件成本翻倍，养老场景单 RGB 够用，靠后处理压 FP（写在 §4）
- 大模型（v8s/m）：板子带不动 25 FPS @ 960，且小目标增益主要靠分辨率不靠模型容量

**不达标时的升级路径**：
1. v8n@960 baseline → 看 mAP 和小火召回
2. 若小火召回 < 80%：上 v8n@1280（NPU 会更慢，可能掉到 12–15 FPS 但仍可用）
3. 若仍不够：v8n@1280 + 加 P2 检测头（在更浅特征层加一个检测分支专攻小目标）
4. 终极备选：换 v11n@960，或干脆上多模态（IR 热阵列）

---

## 3. 数据集策略

### 3.1 五个备选集的真实情况

| 数据集 | 标注类型 | 体量 | 估算体积 | License | 直接用 | 说明 |
|---|---|---|---|---|---|---|
| FASDD（CV 子集） | bbox（YOLO） | **95,314 图**、~14 万 bbox | ~25–40 GB | **CC-BY 4.0** ✅ 商用 OK | ✅ | 主集，最大最全；含户外，需筛 indoor 视角加权。**注**：原 ESSD preprint 因 RS 子集质量问题被作者撤回，但我们只用 CV 子集（已发表于 Geo-spatial Information Science 2024） |
| D-Fire | bbox（YOLO） | 21,527（fire-only 1,164 / smoke-only 5,867 / both 4,658 / 负样本 9,838） | ~3–5 GB | **CC0** ✅ 公共领域 | ✅ | 经典 benchmark，与 FASDD 互补；零商用限制 |
| FSSD | **语义分割 mask** | 1,968 图 / 2,971 实例（室内/其他） | ~0.5–1 GB | MDPI 默认 CC-BY 4.0（待人工核） | ⚠️ 需转换 | mask → bbox 后并入；P4 室内增强阶段使用 |
| **MDPI Fire 2025 室内监控视频集**（替代 Material-Aux） | 视频片段（待确认是否带 bbox） | **1108 段视频**，含早期火 + 真实误报触发 | TBD | MDPI 默认 CC-BY 4.0（待人工核） | ❓ 视频，需帧抽取 | **场景与同事痛点完全对应**：pre-flashover 早期火 / 火色物品 / 人造光源 / 多种室内布局。来源：[Reliable Indoor Fire Detection Using Attention-Based 3D CNNs](https://www.mdpi.com/2571-6255/8/7/285)。P3 用作 FFT 闪烁验证 + P5 holdout 误报评估 |
| Home Fire（Kaggle pengbo00） | 多为分类（待 Kaggle 登录确认） | TBD | ~0.5–2 GB（估） | TBD | ⚠️ 需重标 | 居家场景，P6 伪标注后并入 |
| ~~Indoor Lab Fire（MDPI Fire 2022）~~ | ❌ 传感器时序数据，非图像 | 8 组实验 CSV | — | — | ❌ 不适用 | 2026-05-06 修正：原误列。可留作未来多模态融合的传感器基线 |
| ~~Material-Auxiliary Fire Dataset~~ | — | — | — | — | ❌ 弃用 | 2026-05-06 替换：被 MDPI Fire 2025 室内视频集（场景匹配度更高）取代 |

### 3.2 三层使用策略

**Tier 1 — 主训练集（直接合）**
- `FASDD_CV` + `D-Fire`
- 注意：图像 hash 去重（FASDD 部分来源就是早期公开集，避免 train/val 漏题）
- 类 ID 统一：`0=fire, 1=smoke`
- 切分：7:1:2 → train/val/test

**Tier 2 — 室内增强（转换后合）**
- FSSD 语义分割 mask → bbox：
  - 算法：`cv2.connectedComponentsWithStats` 取连通域外接矩形
  - 过滤：面积阈值（去碎渣）、长宽比上限（去窄条噪声）
  - 输出：YOLO txt 格式
- 用于 baseline 训完后的第二轮 fine-tune，专攻室内召回

**Tier 3 — 自动伪标注（半监督）**
- 候选：Home Fire（Kaggle pengbo00）+ MDPI Fire 2025 1108 视频集帧抽取
- 分类标签直接训会让模型学坏（崩成"看到红色就画全图框"）
- 流程：Tier 1+2 baseline → 跑这些集做伪标注 → 人工抽检 1–2k → 修正 → 第三轮 fine-tune
- **MDPI Fire 2025 视频集的特殊价值**：含真实误报触发器（火色物品、人造光源），是同事旧 FP 痛点的"公开版替代品"，可显著降低 holdout 误报率
- 备选：只当**图像级评估集**（看 max-confidence 准确率），不进训练
- ~~Indoor Lab Fire~~：剔除（2026-05-06：传感器时序数据，非图像）
- ~~Material-Auxiliary~~：剔除（2026-05-06：被 MDPI Fire 2025 视频集替换）

**Tier 4 — 真实场景 holdout（必须）**
- 5–10% 客户家自录数据，全程不进训练，只做最终评估
- 必录场景（误报来源）：厨房灶火、香炉/蜡烛、夕阳直射、红色窗帘/灯笼、电视画面里的火、红色衣物

### 3.3 风险与对策

- **类不平衡**：FASDD 量级碾压其他集 → 按 source 加权 sampling，FSSD 室内样本权重 ×3
- **分布漂移**：每张图打 `source` tag（fasdd_cv / dfire / fssd_indoor / lab / home / local），训完按 source 切片看指标
- **License**：FASDD/D-Fire 学术开放，FSSD/Indoor Lab 来自论文需看作者声明，**商用前出 license 确认表**

---

## 4. 后处理与误报抑制

模型输出框 ≠ 告警。**误报是养老场景头号杀手**（alarm fatigue），层层过滤：

### 4.1 单帧级
1. **置信度门**：fire conf ≥ 0.45，smoke conf ≥ 0.40（待 PR 曲线调）
2. **HSV 颜色复核**：fire bbox 内 HSV 像素满足 `(H ∈ [0,30]∪[330,360]) ∧ (S ≥ 80) ∧ (V ≥ 150)` 的占比 ≥ 30%
3. **尺寸/长宽比**：bbox 面积 ≥ 全图 0.05%（过滤碎像素），长宽比 < 8（过滤窄条误检）

### 4.2 时序级
4. **闪烁频率**：火焰自然闪烁 5–15 Hz。对同一 track 取最近 1s 的 bbox 中心 y 坐标做 FFT，主频在 [3, 20] Hz 才算真火
5. **运动一致性**：smoke 一定向上/向侧扩散，bbox 中心 dy/dt 长期 ≤ 0（向上）才算
6. **Tracker**：复用挥手模块的 `SimpleIoUTracker`（IoU + 归一化中心距 + 短时预测桥接 dropout）

### 4.3 告警级
7. **N/M 帧投票**：连续 30 帧（~1s @ 25fps）中至少 18 帧检出且通过 §4.1+4.2 → 升级为 `wave_events.jsonl` 同款的 `fire_events.jsonl` 事件
8. **冷却期**：同一 track 触发后 10s 内不再升级，防止刷屏

---

## 5. 项目结构（镜像 handwaving 模块）

```
firedetect/
├── PLAN.md                     # 本文件（设计 + 决策）
├── ANNOTATION_GUIDE.md         # 标注规范（针对同事"框太紧"问题，标注员看）
├── README.md                   # 工作流说明（双机分工、部署步骤）
├── .gitignore                  # 排除 data/、*.rknn、*.pt、runs/
├── rk3588/                     # ⬇️ 部署到 Orange Pi 5 Pro，板上推理
│   ├── __init__.py
│   ├── model.py                # 抽象基类（复用挥手模块）
│   ├── coco_utils.py           # letterbox / COCO 助手（复用）
│   ├── rknn_model_postprocess.py  # YOLO DFL 3-branch 解码 + NMS（复用）
│   ├── rknnlite_model.py       # RKNN 推理壳（复用，v8n 通用）
│   ├── tracker.py              # SimpleIoUTracker（从挥手模块拆出）
│   ├── fire_detector.py        # 后处理核心（HSV / 闪烁 FFT / 投票 / 冷却）
│   ├── demo_fire_yolo.py       # E2E demo（视频/RTSP/UVC → 检测 → 后处理 → 事件）
│   └── model/
│       ├── fire.rknn           # INT8 量化产物（gitignore）
│       └── fire.onnx           # ONNX 中间产物（gitignore）
├── train/                      # ⬇️ 在 krix-su（RTX 5090）上跑，训练 + 转换
│   ├── __init__.py
│   ├── requirements.txt        # ultralytics / opencv / imagededup / rknn-toolkit2 等
│   ├── data.yaml               # YOLO 数据集配置（类名、路径）
│   ├── dataset_prep.py         # 拉取 FASDD+D-Fire / pHash 去重 / 类对齐 / 切分
│   ├── fssd_mask2bbox.py       # FSSD 分割 mask → YOLO bbox
│   ├── pseudo_label.py         # Home Fire / Material-Auxiliary 自动伪标注
│   ├── hard_negative_import.py # 同事旧模型 FP 截图 → 训练集硬负样本
│   ├── train_yolo.py           # ultralytics YOLOv8n 训练入口
│   ├── export_rknn.py          # ONNX → RKNN（INT8 PTQ，校准集 200 张含 ≥30% 小火）
│   └── eval_by_source.py       # 按 source tag 切片看 mAP + 小火召回
└── data/                       # ⬇️ krix-su 本地存储，gitignore（合计 ~35 GB+）
    ├── raw/                    # 公开集原始下载
    ├── merged/                 # 合并后标准 YOLO 数据集
    └── local/                  # 客户家自录数据 + 同事旧 FP 截图
```

---

## 5.5 工作流与机器分工

```
┌─────────────────────────────┐         ┌──────────────────────────┐
│   krix-su (RTX 5090, x86)   │         │   Orange Pi 5 Pro (ARM)  │
│   训练 + 数据 + RKNN 转换    │         │   推理 + 后处理 + 告警    │
├─────────────────────────────┤         ├──────────────────────────┤
│ 1. dataset_prep.py          │         │                          │
│    → 拉公开集 + 去重 + 切分  │         │                          │
│ 2. train_yolo.py            │         │                          │
│    → best.pt（FP32）        │         │                          │
│ 3. ultralytics export onnx  │         │                          │
│    → fire.onnx              │         │                          │
│ 4. export_rknn.py           │         │                          │
│    → fire.rknn (INT8) ──────┼────────►│ 5. rknnlite_model.py     │
│                             │  scp/git│    加载 .rknn 推理        │
│                             │         │ 6. fire_detector.py      │
│                             │         │    HSV+FFT+投票          │
│                             │         │ 7. demo_fire_yolo.py     │
│                             │         │    → fire_events.jsonl   │
└─────────────────────────────┘         └──────────────────────────┘
```

**为什么这样分**：
- **训练在 krix-su**：RTX 5090 32GB + 128GB RAM，跑 YOLOv8n@960 一个 epoch < 5 min；Pi 上跑训练就是不可能的事
- **RKNN 转换在 krix-su**：RKNN-Toolkit2（转换器）只支持 x86 Linux + Python 3.8/3.10，**不能在 ARM 板上跑**。板上跑的是 RKNN-Lite（runtime）
- **数据全部在 krix-su**：35GB+ 公开集 + 自录数据存板子 SSD 浪费且慢
- **代码用 git 同步**：本地（这台 Mac）写代码 push → krix-su pull → 跑训练；产出的 `.rknn` 文件再 scp 到 Orange Pi

**数据回流**：
- 客户家 Pi 跑出的 `fire_events.jsonl` 定期 rsync 回 krix-su，作为下一轮迭代的真实场景样本（误报案例尤其值钱）

**SSH 与同步**：
- 本机 → krix-su：`ssh krix-su`（密码认证，密码用户在对话里给）
- 本机 → Orange Pi：待客户现场配置后补充 SSH alias
- krix-su → Orange Pi：krix-su 公网可达 100.119.144.98，但 Pi 在客户内网，反向通道（frp/zerotier）以后再说

---

## 6. 实施分阶段

| 阶段 | 主机 | 工作 | 产出 | 验收 |
|---|---|---|---|---|
| P0 骨架 | 本机 / git | 建目录、复用挥手模块通用层、写 stub、写标注规范 | 模块能 import 不报错；同步到 krix-su 跑得通 | `python -c "import firedetect"` 无错；`demo_fire_yolo.py --dummy` 出空事件流 |
| P1 数据 | krix-su | `dataset_prep.py` 跑通，FASDD+D-Fire 合并去重；同事旧 FP 导入 | `data/merged/` 标准 YOLO 数据集 | val 集 ≥ 1 万图；类别均衡；硬负样本 ≥ 200 张 |
| P2 baseline | krix-su | 训 YOLOv8n@960，导出 ONNX→RKNN，scp 到 Pi 板上 benchmark | `model/fire.rknn` + 板上推理时间报告 | dfire test 上 mAP@.5 ≥ 0.65；小火（bbox 面积 <1%）召回 ≥ 70%；板上 ≥ 20 FPS |
| P3 后处理 | Pi（板上调试） | 写 `fire_detector.py` 全套过滤 | E2E 事件流 | 挥手模块同款 jsonl 输出 |
| P4 室内增强 | krix-su | FSSD mask→bbox，第二轮 fine-tune | v2 模型 | 室内切片召回提升 ≥ 5pp |
| P5 现场 | 客户家 + krix-su | Pi 录 1–2k 张，回流 krix-su 做 holdout 评估 | 误报率报告 | 24h 误报 ≤ 1 次 |
| P6 伪标注循环 | krix-su | Home Fire（+ Material-Auxiliary 若可用）伪标→人工抽检→第三轮 | v3 模型 | 早期小火召回 ≥ 90% |

---

## 7. 与现有模块的协同

- **复用**：`handwaving/huishou/rk3588/` 的 `rknnlite_model.py`、`rknn_model_postprocess.py`、`coco_utils.py`、`SimpleIoUTracker`
- **不复用**：挥手判定逻辑（`HandWaveDetector` / `PoseWaveFilter` / `WaveParams`）—— 火焰判定逻辑独立写在 `fire_detector.py`
- **未来抽公共库**：等第三个 CV 模块（摔倒检测）落地后，把三家公用的 RKNN 壳 + tracker 抽到 `common/cv/`

---

## 8. 待定 / 风险点

- [x] ~~FASDD license~~ → 已确认 **CC-BY 4.0**（2026-05-06）；下载源 [SciDB DOI](https://doi.org/10.57760/sciencedb.j00104.00103) 国内访问速度待测；下载完打印实测体积更新 25–40 GB 估算
- [x] ~~D-Fire license~~ → 已确认 **CC0**（公共领域）
- [ ] FSSD license 人工确认（MDPI Electronics，默认 CC-BY 4.0，去 [论文页](https://www.mdpi.com/2079-9292/12/18/3778) 看 "License" 字样确认）
- [ ] **MDPI Fire 2025 1108 视频集**验真：从 [论文页](https://www.mdpi.com/2571-6255/8/7/285) 找 Data Availability 链接（Zenodo / GitHub 仓库 / 邮件作者）；确认 license + 标注格式
- [ ] Home Fire（Kaggle pengbo00）登录后确认：图像数 / 标注类型 / 体积
- [ ] 客户家摄像头型号 / 分辨率 / RTSP 协议未定，影响推理输入预处理
- [ ] 是否需要烟雾"早期检测"（火苗未起前）—— 现 baseline 只做火焰+烟雾的视觉特征，纯烟雾早期检测建议加烟感传感器互补
- [ ] 夜间（红外补光摄像头）效果未验证，可能需要单独训练夜间分支
