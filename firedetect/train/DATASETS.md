# 数据集下载手册

> 在 **krix-su** 上跑（训练机）。所有数据落地到 `firedetect/data/raw/<dataset_name>/`。
> 完成后 `dataset_prep.py` 负责合并、去重、切分（P1）。
> License 状态见 [PLAN.md §3.1](../PLAN.md)。

---

## 0. 准备工作（krix-su 上一次就行）

```bash
ssh krix-su
cd ~/hanxu_project/firedetect/train

# 创建 raw 目录
mkdir -p ../data/raw/{fasdd_cv,dfire,fssd,fire2025_video}

# 装环境（如已装可跳过）
python3 -m venv ~/.venv/firedetect && source ~/.venv/firedetect/bin/activate
pip install -r requirements.txt

# Kaggle CLI（拉 D-Fire / Home Fire 用）
pip install kaggle
# 把 kaggle.json 放到 ~/.kaggle/，chmod 600
# 从 https://www.kaggle.com/settings → Create New API Token 下
```

---

## 1. D-Fire（21,527 图，~4 GB，CC0）

最快路径是 **Kaggle 镜像**（headless 一行命令拿）：

```bash
cd ../data/raw/dfire
kaggle datasets download -d sayedgamal99/smoke-fire-detection-yolo
unzip smoke-fire-detection-yolo.zip -d .
rm smoke-fire-detection-yolo.zip
ls   # 应看到 train/ valid/ test/ 三个目录，images + labels 子目录
```

**官方备选**（如 Kaggle 不通）：

```bash
# GitHub 仓库（只有元信息和 utils，**没有数据**，数据在 OneDrive）
git clone https://github.com/gaiasd/DFireDataset.git
# OneDrive 下载链接见仓库 README，需要浏览器；不适合 headless
```

**验证**：
```bash
find . -name "*.txt" | wc -l  # 应 ≈ 21527
du -sh .                       # 应 ~3-5 GB
```

---

## 2. FASDD_CV（95,314 图，~25–40 GB，CC-BY 4.0）

**首选：SciDB DOI 直链**（中科院数据银行，国内最快）：

```bash
cd ../data/raw/fasdd_cv

# DOI 解析后的 SciDB 页面：
#   https://www.scidb.cn/en/detail?dataSetId=ce9c9400b44148e1b0a749f5c3eb0bda
# 或直接 DOI:
#   https://doi.org/10.57760/sciencedb.j00104.00103

# SciDB 的下载需要 https 调用 + 可能要 SciDB 账号（免费注册）
# 推荐：先在浏览器拿到下载 URL（可能是分卷 .zip 或 .tar.gz），再用 wget 拉

# 示例（占位，需要先在浏览器拿到真实 URL）：
# wget -c "https://download.scidb.cn/<token>/FASDD_CV_part1.zip"

# SciDB 提供 IDM / 迅雷支持的多线程下载，大文件用浏览器 + 客户端更稳
```

**备选 1：Kaggle 镜像（COCO 格式）**：

```bash
kaggle datasets download -d yuulind/fasdd-cv-coco
# COCO json 而非 YOLO txt，dataset_prep.py 会负责转换
```

**备选 2：Roboflow 镜像**：

```bash
# 需要 Roboflow API key
pip install roboflow
python -c "
from roboflow import Roboflow
rf = Roboflow(api_key='YOUR_KEY')
ds = rf.workspace('forestfiresmoke').project('fasdd_cv-dx83j').version(1).download('yolov8')
"
```

**注意**：FASDD 原始有三个子集（CV / UAV / RS），**只下 CV**，UAV 和 RS 跟养老室内场景无关。

**验证**：
```bash
ls   # 找 images/ + labels/ 或 train/ val/ test/
du -sh .   # 应 25-40 GB
```

---

## 3. FSSD（1,968 图，~1 GB，CC-BY 4.0 待人工核）

> **License 待人工核**：去 [论文页](https://www.mdpi.com/2079-9292/12/18/3778) 看页面 footer 的 "License" 字样，MDPI Electronics 默认 CC-BY 4.0。

```bash
cd ../data/raw/fssd

# 论文页有 Supplementary Materials 链接，通常是 .zip
# 下载后放在这里：
# wget "https://www.mdpi.com/.../supplementary/fssd_dataset.zip"
# unzip fssd_dataset.zip

# 如果论文没释放数据集（只发了样本图），需邮件作者要：
# 论文通讯作者 Email 在论文最末尾
```

**P4 才用，先不下也没事**。

---

## 4. MDPI Fire 2025 — 1108 段室内监控视频集（待验真）

> **场景与同事痛点完全对应，但需先验真**：
> 论文：[Reliable Indoor Fire Detection Using Attention-Based 3D CNNs](https://www.mdpi.com/2571-6255/8/7/285)
> 待确认 Data Availability 链接（Zenodo / GitHub / 邮件作者）+ license + 标注格式。

**P3 / P5 才用，先不动。** 等 baseline 出来后回头再处理。

---

## 5. 同事旧 FP 截图（硬负样本）

```bash
cd ../data/local
mkdir -p hard_negatives_old_fp/

# 跟同事要旧模型在生产环境的误报截图
# 拷过来后跑：
cd ../../train
python hard_negative_import.py --src ../data/local/hard_negatives_old_fp/
```

---

## 6. Home Fire (Kaggle pengbo00) —— P6 才用

```bash
cd ../data/raw/home_fire
kaggle datasets download -d pengbo00/home-fire-dataset
unzip home-fire-dataset.zip
```

---

## 总下载量估算（P1 阶段必下的）

| 数据集 | 体积 | 必下 |
|---|---|---|
| D-Fire | ~4 GB | ✅ |
| FASDD_CV | ~30 GB | ✅ |
| **总计** | **~34 GB** | **P1 必备** |

后续按阶段补：
- P4：FSSD ~1 GB
- P3/P5：1108 视频集 TBD
- P6：Home Fire ~1 GB

---

## 完成清单（krix-su 上跑）

```bash
cd ~/hanxu_project/firedetect/data/raw

du -sh dfire fasdd_cv  # 确认体积
ls dfire/              # 应有 train/ valid/ test/
ls fasdd_cv/           # 应有 images/ + labels/
```

下完一并 `git status`（应该都被 .gitignore 排除），然后跑 `python ../../train/dataset_prep.py` 进入 P1。
