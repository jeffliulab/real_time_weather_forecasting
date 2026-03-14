# Pipeline Documentation

## 目录 / Table of Contents

1. [项目概述](#项目概述)
2. [任务清单与进度](#任务清单与进度)
3. [数据集详解](#数据集详解)
4. [数据清洗与质量分析](#数据清洗与质量分析)
5. [训练 Pipeline](#训练-pipeline)
6. [分析 Pipeline](#分析-pipeline)
7. [实时预测 Pipeline](#实时预测-pipeline)
8. [HPC 操作指南](#hpc-操作指南)
9. [故障排除](#故障排除)

---

## 项目概述

### 目标

构建一个基于深度学习的区域天气预报系统：输入新英格兰地区的空间天气快照 `(450, 449, 42)`，
预测目标位置（Tufts Jumbo Statue, 42.408°N, 71.120°W）24 小时后的 6 个天气变量 + 1 个二分类标签。

### 四部分任务

| 阶段 | 内容 | 进度 |
|------|------|------|
| Part 1 | 训练 Baseline CNN，实现 6 变量回归 + 降水二分类 | 进行中 |
| Part 2 | 梯度 Saliency 分析，可视化模型对地理区域的关注度 | 待进行 |
| Part 3 | 独立研究：多架构对比（多帧 CNN、3D CNN、ViT） | 待进行 |
| Part 4 | Demo：实时天气预测系统（bonus） | 待进行 |

### 研究假设（Part 3）

> 多时刻输入和更先进的架构（3D CNN、ViT）能提升 24h 区域天气预报的精度，
> 但其优势程度取决于目标变量类型，且受限于数据量（~26k 样本）。

从两个维度展开：
- **时间维度**：同一 CNN 架构，增加输入帧数（1 帧 vs 4 帧）
- **架构维度**：同一输入，换不同模型（2D CNN → 3D CNN → ViT）

### 评估指标

| 指标 | 范围 | 说明 |
|------|------|------|
| RMSE | 每个连续目标 | 预报精度（denormalized 到物理单位后计算） |
| 条件 RMSE | APCP（真值 > 2mm 时） | 强降水预报精度 |
| AUC | 二分类标签 | 降水检测能力 |
| 参数量 | 所有模型 | 模型复杂度 |
| 每 epoch 时间 | 所有模型 | 计算成本 |

### 时间线与交付

| 时间 | 任务 |
|------|------|
| 第 1 周（3/12 - 3/19） | 数据探索，搭建数据管线，训练 baseline CNN |
| 第 2 周（3/19 - 3/26） | Saliency 分析，开始架构对比实验 |
| 第 3 周（3/26 - 4/2） | 完成多模型训练，整理实验结果 |
| 4/3 截止 | 提交代码、报告（PDF）、Slides（≤7 页）、最佳 checkpoint |
| 4/9 | 课堂展示 |

---

## 任务清单与进度

### Part 1: Baseline CNN (60 pts)

- [x] 数据探索：确认 42 通道、26280 样本、NaN 分布
- [x] 数据管线：`WeatherDataset` + `DataLoader`，NaN 过滤、归一化、多帧支持
- [x] 模型设计：2D CNN + Residual blocks + GAP + 回归头（MSE loss）
- [x] 训练脚本：`train.py` with AdamW, CosineAnnealing, grad clip, checkpointing
- [x] 评测接口：`evaluation/baseline_cnn/model.py` 实现 `get_model()`
- [ ] 训练完成（当前 epoch 16/30，val_loss=0.62）
- [ ] 在测试集上运行 `evaluate.py` 生成最终指标

### Part 2: Saliency Analysis (20 pts)

- [x] 实现 `saliency.py`：梯度计算、平均 saliency map、可视化
- [ ] 在训练完成后运行 saliency 分析
- [ ] 为每个目标变量生成 saliency 热力图
- [ ] 叠加到新英格兰地理地图上（使用 metadata.pt 投影信息）
- [ ] 分析讨论：温度是否关注西/西北方向（盛行西风），降水是否关注海岸线

### Part 3: Architecture Comparison (20 pts)

- [x] 实现 4 个模型：`cnn_baseline`, `cnn_multi_frame`, `cnn_3d`, `vit`
- [x] 统一训练脚本，通过 `--model` 参数切换
- [ ] 训练所有模型（需逐个提交 SLURM 任务）
- [ ] 每个模型跑 3 次不同随机种子，报告 mean ± std
- [ ] 对所有模型做 saliency map，对比不同模型"看的区域"
- [ ] 画 train/val loss 对比曲线
- [ ] 逐变量拆解 RMSE，看哪些架构对哪些变量帮助更大

### Part 4: Real-Time Demo (Bonus)

- [x] 搭建 `inference/predict.py` 骨架
- [ ] 实现 `fetch_gfs_data()`：从 NOAA NOMADS 下载 GFS 数据
- [ ] 实现 GFS → HRRR 网格重投影
- [ ] 实现变量名映射（GFS → HRRR 42 通道格式）
- [ ] 搭建简单 Web UI 展示预测结果

### 提交清单

- [ ] 代码（训练脚本、notebook、model.py）
- [ ] 报告（PDF，覆盖 Part 1/2/3）
- [ ] 演示 slides（≤ 7 页）
- [ ] 最佳模型 checkpoint

---

## 数据集详解

### 数据来源

HRRR（High-Resolution Rapid Refresh）是 NOAA 提供的高分辨率天气再分析数据。
原始数据为 Zarr 格式，通过 `data_preparation/generate_dataset.py` 转换为 PyTorch `.pt` 文件。

### 数据集生成流程

```
HRRR Zarr 存档（xarray/zarr 格式）
├── hrrr_ne_anl.zarr     # 大气分析场（41 个变量）
└── hrrr_ne_apcp.zarr    # 累积降水场
         │
         ↓  generate_dataset.py
         │
PyTorch .pt 文件
├── dataset/inputs/YYYY/X_YYYYMMDDHH.pt   # 每小时空间快照
├── dataset/targets.pt                      # 所有时间步的预测目标
└── dataset/metadata.pt                     # 数据集元信息
```

### 文件结构与规模

```
dataset/                             总计约 416 GB，26,280 个文件
├── inputs/
│   ├── 2018/                        4,128 个文件，~65 GB（2018-07-13 ~ 2018-12-31）
│   │   ├── X_2018071300.pt          单文件 ~16.2 MB (bfloat16)
│   │   ├── X_2018071301.pt
│   │   └── ...
│   ├── 2019/                        8,760 个文件，~139 GB（2019-01-01 ~ 2019-12-31）
│   ├── 2020/                        8,784 个文件，~139 GB（2020-01-01 ~ 2020-12-31，闰年）
│   └── 2021/                        4,608 个文件，~73 GB（2021-01-01 ~ 2021-07-12）
├── targets.pt                       ~1.2 MB（26,280 × 6 float32 + 元数据）
└── metadata.pt                      ~4 MB（变量名、网格坐标、投影信息）
```

### 输入张量 (`X_YYYYMMDDHH.pt`)

| 属性 | 值 |
|------|-----|
| 形状 | `(450, 449, 42)` — (Height, Width, Channels) |
| 数据类型 | `torch.bfloat16`（加载后转为 `float32`）|
| 单文件大小 | ~16 MB |
| 文件总数 | 26,280（3 年逐小时，2018-07 至 2021-07） |
| 总数据量 | ~416 GB（26,280 × 16.2 MB）|
| 坐标系 | Lambert Conformal Conic 投影 |
| 空间分辨率 | 3 km |
| 空间范围 | 新英格兰地区（449 × 450 像素） |
| 数值范围 | -39 ~ 5920（跨越温度 K、高度 m、百分比 %、风速 m/s 等不同量纲） |

#### 42 个输入通道详细定义

通道按 `VAR_LEVELS = TARGET_VARS + ATMOS_VARS` 顺序排列（定义在 `data_preparation/data_spec.py`）。

**通道 0–6: 近地面 / 地表变量**（与预测目标部分重叠）

| 通道 | 变量名 | 说明 | 单位 | 典型值范围 |
|------|--------|------|------|-----------|
| 0 | `TMP@2m_above_ground` | 2 米温度 | K | 250–310 |
| 1 | `RH@2m_above_ground` | 2 米相对湿度 | % | 0–100 |
| 2 | `UGRD@10m_above_ground` | 10 米东向风 | m/s | -15–15 |
| 3 | `VGRD@10m_above_ground` | 10 米北向风 | m/s | -15–15 |
| 4 | `GUST@surface` | 地面阵风 | m/s | 0–30 |
| 5 | `DSWRF@surface` | 地表短波辐射 | W/m² | 0–1000 |
| 6 | `APCP_1hr_acc_fcst@surface` | 1 小时累积降水 | mm | 0–50 |

> **注意**：DSWRF（通道 5）在 42 个输入通道中但不在 6 个预测目标中。

**通道 7–41: 高空大气变量**（35 个通道）

| 通道 | 变量名 | 说明 |
|------|--------|------|
| 7 | `CAPE@surface` | 对流有效位能 (J/kg) |
| 8–12 | `DPT@{1000,500,700,850,925}mb` | 露点温度，5 个气压层 |
| 13–17 | `HGT@{1000,500,700,850,surface}` | 位势高度，5 层 |
| 18–22 | `TMP@{1000,500,700,850,925}mb` | 温度，5 个气压层 |
| 23–28 | `UGRD@{1000,250,500,700,850,925}mb` | 东向风，6 个气压层 |
| 29–34 | `VGRD@{1000,250,500,700,850,925}mb` | 北向风，6 个气压层 |
| 35 | `TCDC@entire_atmosphere` | 总云量 (%) |
| 36 | `HCDC@high_cloud_layer` | 高云量 (%) |
| 37 | `MCDC@middle_cloud_layer` | 中云量 (%) |
| 38 | `LCDC@low_cloud_layer` | 低云量 (%) |
| 39 | `PWAT@entire_atmosphere_single_layer` | 可降水量 (kg/m²) |
| 40 | `RHPW@entire_atmosphere` | 全大气层相对湿度 (%) |
| 41 | `VIL@entire_atmosphere` | 垂直积分液态水含量 (kg/m²) |

### 预测目标 (`targets.pt`)

存储为 Python 字典：

```python
{
    "time":           np.ndarray (26280,) datetime64[ns],
    "variable_names": [6 个变量名],
    "values":         Tensor (26280, 6) float32,
    "binary_label":   Tensor (26280,) bool,      # APCP > 2mm
    "grid_y_idx":     177,                        # 预测目标点 y 索引
    "grid_x_idx":     263,                        # 预测目标点 x 索引
    "grid_proj_x":    2141479.86,                 # 投影坐标 x (m)
    "grid_proj_y":    743693.85,                  # 投影坐标 y (m)
}
```

#### 6 个预测目标的统计分布

| 索引 | 变量 | 单位 | 均值 | 标准差 | 最小值 | 最大值 | NaN 数量 | NaN 比例 |
|------|------|------|------|--------|--------|--------|----------|----------|
| 0 | TMP@2m | K | 284.17 | 10.32 | 256.25 | 309.25 | 494 | 1.9% |
| 1 | RH@2m | % | 68.94 | 19.92 | 13.00 | 100.00 | 494 | 1.9% |
| 2 | UGRD@10m | m/s | 0.98 | 2.46 | -10.31 | 11.41 | 494 | 1.9% |
| 3 | VGRD@10m | m/s | 0.06 | 2.18 | -9.59 | 10.60 | 494 | 1.9% |
| 4 | GUST@sfc | m/s | 6.77 | 3.89 | 0.14 | 29.14 | 526 | 2.0% |
| 5 | APCP | mm | 0.14 | 0.72 | 0.00 | 26.40 | 0 | 0% |

二分类标签（APCP > 2mm）：正样本约占 3-5%（极度不平衡）。

### 元数据 (`metadata.pt`)

```python
{
    "variable_names":  list[str],    # 42 个输入变量名（按通道顺序）
    "n_vars":          42,
    "input_shape":     (450, 449, 42),
    "times":           np.ndarray,   # (26280,) 时间序列
    "grid_x":          np.ndarray,   # (449,) x 坐标 (投影空间, 米)
    "grid_y":          np.ndarray,   # (450,) y 坐标 (投影空间, 米)
    "projection":      "LambertConformal(central_lon=262.5, central_lat=38.5)",
    "target_vars":     list[str],    # 6 个目标变量名
    "jumbo_y_idx":     177,          # 预测目标点的网格 y 索引
    "jumbo_x_idx":     263,          # 预测目标点的网格 x 索引
}
```

### 归一化统计量 (`norm_stats.pt`)

由 `data_preparation/dataset.py` 的 `compute_norm_stats()` 从训练集抽样 1000 个样本计算。

```python
{
    "input_mean":  Tensor (42, 1, 1),   # 每通道均值（空间取平均后）
    "input_std":   Tensor (42, 1, 1),   # 每通道标准差
    "target_mean": Tensor (6,),         # 每目标变量均值
    "target_std":  Tensor (6,),         # 每目标变量标准差
}
```

| 目标变量 | norm 均值 | norm 标准差 |
|----------|-----------|------------|
| TMP@2m | 286.65 K | 9.93 |
| RH@2m | 69.18 % | 19.42 |
| UGRD@10m | 0.83 m/s | 2.40 |
| VGRD@10m | 0.21 m/s | 2.14 |
| GUST@sfc | 6.61 m/s | 3.48 |
| APCP | 0.14 mm | 0.75 |

### 数据集划分

| 集合 | 年份 | 样本数 (约) | 用途 |
|------|------|------------|------|
| 训练集 | 2018-07 ~ 2019-12 | ~13,000 | 模型训练 |
| 验证集 | 2020 | ~8,784 | 超参数调整、early stopping、RMSE/AUC 评估 |
| 测试集 | 2021 | ~4,500 | 最终评估（由 `evaluate.py` 使用，训练过程中不接触）|

### 投影坐标系

数据使用 Lambert Conformal Conic 投影（非经纬度），参数：

- 中心经度：262.5°E (= 97.5°W)
- 中心纬度：38.5°N
- 标准纬线：38.5°N
- 地球半径：6,371,229 m
- 空间分辨率：3,000 m/pixel
- x 范围：1,352,480 ~ 2,696,480 m（449 点）
- y 范围：212,694 ~ 1,559,694 m（450 点）

---

## 数据清洗与质量分析

### NaN 问题概览

数据中存在两种 NaN 来源，对训练有不同影响：

| NaN 来源 | 位置 | 数量/比例 | 原因 | 处理方式 |
|----------|------|-----------|------|----------|
| 输入 NaN | `X_YYYYMMDDHH.pt` 文件 | ~40-60% 文件含 NaN | HRRR 原始数据缺失 | `_load_frame()` 检测 → 返回 None → `collate_skip_none` 过滤 |
| 目标 NaN | `targets.pt` 的 values 列 | 494-526 / 26280 (1.9-2.0%) | 目标网格点数据缺失 | MSE loss 为 NaN → `if torch.isnan(loss): continue` 跳过 |

#### 输入 NaN 详情

- 约 40-60% 的 `.pt` 文件含 NaN 值（整个 450×449×42 张量中至少有一个 NaN）
- `WeatherDataset._load_frame()` 用 `torch.isnan(x).any()` 检测，含 NaN 的帧返回 None
- `collate_skip_none` 在 DataLoader 层面过滤掉 None 样本
- **有效训练样本数约为名义样本数的一半**

#### 目标 NaN 详情

- TMP、RH、UGRD、VGRD 的 NaN **同时出现**（同一时间步缺失）：494 个时间步
- GUST 额外有 32 个 NaN（总共 526）
- APCP **完全没有 NaN**（0 个）
- NaN 出现的原因：HRRR 在某些时间步的目标网格点无有效观测
- 2020 验证集中：517 个样本含任何 NaN（占 8784 的 5.9%），8267 个完全有效

#### bfloat16 精度

输入存储为 `bfloat16`（7 位尾数，约 3 位十进制精度），加载后转 `float32`。
对温度 (~280K) 精度约 ±0.5K，对风速 (~1 m/s) 精度约 ±0.01 m/s。不会引入数值问题。

### NaN 对训练的影响分析

**结论：NaN 对训练的影响很小，不会造成梯度爆炸或梯度消失。**

依据（来自当前 baseline 训练 17 个 epoch 的实际数据）：

#### 1. NaN batch 极少，不影响梯度流

- 17 个 epoch 共 27,387 个 batch（1611 × 17），其中 NaN loss 仅 120 次 = **0.44%**
- 每 epoch 约 7 个 NaN batch（占 1611 的 0.43%）
- NaN batch 直接 `continue` 跳过：**不执行 backward()、不更新权重**
- 由于不参与反向传播，NaN 值不会污染梯度或模型参数

#### 2. 训练稳定收敛，无爆炸/消失迹象

```
Epoch  0: train=0.894  val=0.863  → 正常初始化
Epoch  5: train=0.653  val=0.647  → 稳定下降
Epoch 10: train=0.565  val=0.629  → 开始出现 train/val gap
Epoch 15: train=0.467  val=0.621  → 持续收敛，*best*
Epoch 16: train=0.458  val=0.634  → 轻微过拟合但无发散
```

- **Train loss 单调递减**：0.894 → 0.458（下降 49%），无突然跳升（无梯度爆炸）
- **Val loss 持续下降**：0.863 → 0.621（下降 28%），无 NaN（无数值不稳定）
- **每 epoch 时间稳定**：1001-1131s（无内存泄漏或计算异常）

#### 3. 梯度裁剪提供额外保护

```python
torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
```

即使偶尔出现极端梯度（例如因为 bfloat16 精度问题），梯度裁剪也会将其限制在 L2 范数 ≤ 1.0 内。

#### 4. 为什么之前 RMSE 显示 nan

这不是 NaN 造成的训练问题，而是 **指标计算的 bug**：

- `targets.pt` 中 TMP/RH/UGRD/VGRD/GUST 列含有 NaN
- 旧版 `compute_metrics` 在 denormalize 后直接对整列做 `((pred - target)**2).mean()`
- NaN 参与 `.mean()` → 结果为 NaN
- APCP 列无 NaN → 其条件 RMSE 正常计算
- **已修复**：改为逐变量过滤 `torch.isfinite(p) & torch.isfinite(t)` 后再计算 RMSE
- 修复后实测：TMP=2.27K, RH=13.33%, UGRD=2.09m/s, VGRD=1.33m/s, GUST=3.82m/s

### 数据清洗流程总结

```
原始 .pt 文件
    │
    ↓ ① bfloat16 → float32 转换
    │
    ↓ ② NaN 输入检测（torch.isnan(x).any()）
    │   ├── 含 NaN → 返回 None → collate_skip_none 过滤
    │   └── 无 NaN → 继续
    │
    ↓ ③ 通道维度转置 (H,W,C) → (C,H,W)
    │
    ↓ ④ Z-score 归一化（input_mean, input_std 逐通道）
    │
    ↓ ⑤ 目标值归一化（target_mean, target_std）
    │   └── 目标含 NaN 时：归一化结果仍为 NaN → 训练时 NaN loss 被跳过
    │
    ↓ 进入模型
```

---

## 训练 Pipeline

### 整体流程

```
┌─────────────┐     ┌──────────────┐     ┌──────────────┐
│ WeatherData │────▶│  Model       │────▶│ MSE Loss     │
│ (B,C,H,W)   │     │ (CNN/ViT)    │     │ + Backprop   │
└─────────────┘     └──────────────┘     └──────────────┘
                           │                     │
                           ▼                     ▼
                    ┌──────────────┐     ┌──────────────┐
                    │ Predictions  │     │ AdamW + Grad  │
                    │ (B, 6)       │     │ Clip + CosLR  │
                    └──────────────┘     └──────────────┘
```

### 数据加载 (`data_preparation/dataset.py`)

```
WeatherDataset
    ├── 按需加载 .pt 文件（避免一次性加载所有数据到内存）
    ├── 支持单帧 / 多帧模式
    │   ├── single:   (C, H, W)           — 1 个时间步
    │   ├── channel:  (C*k, H, W)         — k 帧沿通道维度拼接（多帧 2D CNN）
    │   └── temporal: (k, C, H, W)        — k 帧保持时间维度（3D CNN）
    ├── NaN 输入过滤（返回 None → collate_skip_none 过滤）
    ├── NaN 目标保留（传给模型，由 loss 层处理）
    └── Z-score 归一化（均值=0, 标准差=1）
```

### 四个模型架构

| 模型 | 输入形状 | 参数量 | 核心思路 |
|------|----------|--------|----------|
| `cnn_baseline` | `(B, 42, 450, 449)` | ~1.8M | 单帧 2D CNN + Residual blocks + GAP |
| `cnn_multi_frame` | `(B, 42*4, 450, 449)` | ~2.5M | 4 帧通道拼接 → 2D CNN |
| `cnn_3d` | `(B, 4, 42, 450, 449)` | ~1.2M | Conv3d 时空三维卷积 |
| `vit` | `(B, 42, 450, 449)` → patches | ~3.5M | Patch embedding + Transformer |

### 运行训练

```bash
# 本地同步代码到 HPC
powershell -File scripts/sync.ps1

# SSH 到 HPC
ssh <hpc-login>

# 提交训练任务（从项目根目录）
sbatch scripts/train.slurm                              # Baseline CNN
sbatch --job-name=multi scripts/train.slurm cnn_multi_frame
sbatch --job-name=cnn3d scripts/train.slurm cnn_3d
sbatch --job-name=vit   scripts/train.slurm vit

# 监控任务
squeue -u $USER
tail -f runs/weather_cnn_<JOBID>.out
```

### 训练参数

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--epochs` | 30 | 训练轮数 |
| `--batch_size` | 8 | 批次大小 |
| `--lr` | 1e-3 | 初始学习率 |
| `--weight_decay` | 1e-4 | L2 正则化 |
| `--scheduler` | cosine | 学习率调度（cosine / plateau / none）|
| `--num_workers` | 2 | DataLoader 工作进程数（过多会 OOM）|
| `--base_channels` | 64 | CNN 基础通道数 |

### 训练输出

```
runs/<model_name>/
├── checkpoints/
│   ├── best.pt          # 最低验证损失的模型
│   └── latest.pt        # 最新一轮的模型
├── logs/
│   └── training_log.csv # 每轮的训练/验证指标
├── figures/
│   └── training_curves.png
└── config.json          # 训练超参数记录
```

### Checkpoint 内容

```python
{
    "epoch": int,
    "model": OrderedDict,        # model.state_dict()
    "optimizer": dict,            # optimizer.state_dict()
    "best_val_loss": float,
    "args": dict,                 # 完整的 argparse 参数
    "norm_stats": {               # 归一化统计量（推理时需要）
        "input_mean": Tensor,     # (42, 1, 1)
        "input_std": Tensor,
        "target_mean": Tensor,    # (6,)
        "target_std": Tensor,
    }
}
```

### 控制变量（Part 3 对比实验）

为公平比较，所有模型使用相同的：
- 训练/验证/测试划分（2018-19 / 2020 / 2021）
- 优化器（AdamW, lr=1e-3, weight_decay=1e-4）
- 学习率调度（CosineAnnealing, T_max=30）
- 梯度裁剪（max_norm=1.0）
- 训练轮数（30 epochs）
- 输出头设计（线性层 → 6 个回归值）
- 每个模型跑 3 次不同随机种子

---

## 分析 Pipeline

### Part 2: Saliency 分析

梯度 saliency 可视化——显示输入中哪些地理区域对预测贡献最大。

**原理**：

```python
x.requires_grad_(True)
output = model(x)          # (1, 6)
output[0, k].backward()    # 对第 k 个目标变量反向传播
saliency = x.grad.abs().mean(dim=1)  # 通道维取均值 → (450, 449) 热力图
```

对多个样本取平均，得到稳定的 saliency map。

**运行**：

```bash
sbatch scripts/saliency.slurm runs/cnn_baseline/checkpoints/best.pt
```

**输出**：
- `saliency_maps.png` — 6 个目标变量 + 总体 saliency 的热力图
- `saliency_<VAR>.png` — 每个变量的高分辨率单独热力图
- `saliency_data.pt` — 原始 saliency 数据（可后续分析）

**预期分析方向**：
- 温度预测：是否关注西/西北方向（中纬度盛行西风带，天气系统从上游来）
- 降水预测：是否关注海岸线、南方（水汽来源）或山地区域
- 风速预测：是否关注地形区域或海陆交界
- 不同目标变量的关注区域差异如何

### Part 3: 模型架构对比

训练完所有模型后，比较以下维度：

| 对比维度 | 说明 |
|----------|------|
| 逐变量 RMSE | 各变量的预测精度（哪些架构对哪些变量帮助更大）|
| 条件 APCP RMSE | 强降水场景下的预报精度 |
| AUC | 降水分类能力 |
| 参数量 | 模型复杂度 |
| 训练时间 / epoch | 计算效率 |
| 收敛速度 | 到达最优 val loss 的 epoch 数 |
| Saliency 对比 | 不同模型"看的区域"是否不同 |
| Train/Val 曲线 | 对比收敛行为和过拟合程度 |

---

## 实时预测 Pipeline

### 目标

建立从公开气象数据源到实时预测的端到端 pipeline，无需访问 HRRR 内部数据。

### 架构设计

```
┌─────────────────┐     ┌──────────────────┐     ┌─────────────────┐
│ NOAA GFS API    │────▶│ Regrid & Format  │────▶│ Trained Model   │
│ (0.25° global)  │     │ GFS → HRRR grid  │     │ (checkpoint)    │
└─────────────────┘     └──────────────────┘     └─────────────────┘
                                                          │
                                                          ▼
                                                 ┌─────────────────┐
                                                 │ 24h Forecast    │
                                                 │   at target     │
                                                 └─────────────────┘
```

### 实现步骤

1. **获取 GFS 数据** (`inference/predict.py` → `fetch_gfs_data()`)
   - 数据源：NOAA NOMADS — `https://nomads.ncep.noaa.gov/dods/gfs_0p25/`
   - 获取最新的 GFS 0.25° 全球分析数据
   - 提取新英格兰区域子集

2. **重新网格化**
   - 将 GFS 0.25° 经纬度网格插值到 HRRR 3km Lambert Conformal Conic 投影
   - 目标网格：450 × 449 像素
   - 投影参数见 `data_preparation/data_spec.py`

3. **变量映射**
   - GFS 变量名 → HRRR 42 通道格式（顺序和命名不同）
   - 缺失变量（如 VIL、RHPW）用近似值或零填充

4. **模型推理**
   - 加载 checkpoint（含 norm_stats）
   - 应用与训练相同的 Z-score 归一化
   - 前向传播 → denormalize 输出

5. **展示结果**
   - CLI 输出 / Web UI / API 接口

```bash
python -m inference.predict --checkpoint runs/cnn_baseline/checkpoints/best.pt
```

---

## HPC 操作指南

### 环境

- **集群**：SLURM-managed HPC
- **GPU**：通过 SLURM `--gres=gpu:1 -p gpu` 请求
- **项目目录 / 数据集目录**：见 `scripts/sync.ps1` 中的配置

### 常用命令

```bash
# 连接 HPC
ssh <hpc-login>

# 查看队列
squeue -u $USER

# 取消任务
scancel <JOBID>

# 实时查看输出
tail -f runs/<JOB_OUTPUT>.out

# 交互式 GPU 节点（调试用）
srun -p gpu --gres=gpu:1 --mem=32G -c 4 --pty bash

# 同步代码（从本地 PowerShell）
powershell -File scripts/sync.ps1
```

### SLURM 资源配置

| 参数 | 训练任务 | Saliency 分析 |
|------|----------|---------------|
| CPUs | 4 | 4 |
| Memory | 48GB | 32GB |
| GPU | 1 | 1 |
| Time limit | 24h | 4h |
| Partition | gpu | gpu |

---

## 故障排除

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| OOM Killed | DataLoader worker 占用过多内存 | `--num_workers 2`，不要超过 4 |
| NaN loss (少量) | 目标值含 NaN | 正常，代码自动跳过（0.4%/epoch）|
| NaN RMSE 指标 | 旧版 compute_metrics 未逐变量过滤 NaN | 已修复，逐变量 `isfinite` 过滤 |
| DOS line breaks | Windows → Linux 换行符 | `sed -i 's/\r$//' scripts/*.slurm` |
| 慢速数据加载 | NFS 网络文件系统 I/O 瓶颈 | 正常现象，每 epoch 约 17 分钟 |
| SSH timeout | HPC 节点变更 | 更新 `~/.ssh/config` 中的 HostName |
| 输入含 ~50% NaN 文件 | HRRR 原始数据缺失 | 有效样本约一半，已自动过滤，不影响训练 |
