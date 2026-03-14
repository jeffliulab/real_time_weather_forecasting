# Pipeline Documentation

## 项目概述 / Project Overview

本项目构建一个基于深度学习的区域天气预报系统。系统使用 HRRR（High-Resolution Rapid Refresh）再分析数据，输入新英格兰地区的空间天气快照，预测目标位置 24 小时后的天气状况。

This project builds a deep-learning-based regional weather forecasting system using HRRR reanalysis data. Given a spatial weather snapshot of the New England region, the system predicts 6 weather variables 24 hours ahead at the target location.

---

## 数据 Pipeline / Data Pipeline

### 1. 原始数据 → PyTorch 数据集

原始数据存储在 HPC 上的 Zarr 格式文件中（由 `data_preparation/generate_dataset.py` 处理）。

```
原始 HRRR Zarr 数据 (xarray/zarr)
    ↓  generate_dataset.py
PyTorch .pt 文件 (每小时一个快照)
    ├── dataset/YYYYMMDDHH.pt    # shape: (450, 449, 42) float32
    ├── dataset/targets.pt        # shape: (N, 6) — 各时间步的预测目标
    └── dataset/metadata.pt       # 变量名、网格信息等
```

**数据规模**：约 67GB（2018-2020 年，每小时一个样本，约 26,000 个样本）

**42 个输入通道**（定义在 `data_preparation/data_spec.py`）：
- 温度、相对湿度、风速（u, v）等大气变量
- 覆盖多个气压层（surface, 2m, 10m, 850mb, 700mb, 500mb, 250mb 等）

**6 个预测目标**：
| 变量 | 说明 | 单位 |
|------|------|------|
| TMP@2m | 2米温度 | K |
| RH@2m | 2米相对湿度 | % |
| UGRD@10m | 10米东向风 | m/s |
| VGRD@10m | 10米北向风 | m/s |
| GUST@sfc | 地面阵风 | m/s |
| APCP_1hr | 1小时累积降水 | mm |

### 2. 数据集划分

| 集合 | 年份 | 用途 |
|------|------|------|
| 训练集 | 2018, 2019 | 模型训练 |
| 验证集 | 2020 | 超参数调整、early stopping |
| 测试集 | 2021 | 最终评估（由 `evaluate.py` 使用） |

### 3. 数据加载 (`data_preparation/dataset.py`)

```
WeatherDataset
    ├── 按需加载 .pt 文件（避免一次性加载所有数据到内存）
    ├── 支持单帧 / 多帧模式
    │   ├── single:   (C, H, W)           — 1 个时间步
    │   ├── channel:  (C*k, H, W)         — k 帧沿通道维度拼接
    │   └── temporal: (k, C, H, W)        — k 帧保持时间维度
    ├── NaN 过滤（跳过含 NaN 的样本）
    └── 归一化（均值=0, 标准差=1）
```

---

## 训练 Pipeline / Training Pipeline

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

### 运行训练

```bash
# 本地同步代码到 HPC
powershell -File scripts/sync.ps1

# SSH 到 HPC（别名配置见 ~/.ssh/config）
ssh <hpc-login>

# 提交训练任务
cd $PROJECT_ROOT  # remote project directory
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
| `--lr` | 1e-3 | 学习率 |
| `--scheduler` | cosine | 学习率调度（cosine/plateau/none）|
| `--num_workers` | 2 | DataLoader 工作进程数 |

### 输出

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
    "norm_stats": {               # 归一化统计量
        "input_mean": Tensor,     # (42, 1, 1)
        "input_std": Tensor,
        "target_mean": Tensor,    # (6,)
        "target_std": Tensor,
    }
}
```

---

## 分析 Pipeline / Analysis Pipeline

### Part 2: Saliency 分析

梯度 saliency 可视化——显示输入中哪些地理区域对预测贡献最大。

```bash
# 提交 saliency 分析任务
sbatch scripts/saliency.slurm runs/cnn_baseline/checkpoints/best.pt
```

**原理**：对模型输出关于输入做反向传播，取梯度的绝对值在通道维度上取均值，得到每个空间位置的"重要性"。

**输出**：
- `saliency_maps.png` — 6 个目标变量 + 总体 saliency 的热力图
- `saliency_<VAR>.png` — 每个变量的高分辨率单独热力图
- `saliency_data.pt` — 原始 saliency 数据

### Part 3: 模型架构对比

训练完所有模型后，比较各模型在验证集上的表现：

| 对比维度 | 说明 |
|----------|------|
| RMSE per variable | 各变量的预测精度 |
| AUC | 降水分类能力 |
| 参数量 | 模型复杂度 |
| 训练时间 | 计算效率 |
| 时序信息 | 多帧 vs 单帧的影响 |

---

## 实时预测 Pipeline / Real-time Inference Pipeline (Part 4)

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
                                                 │ for Jumbo Statue│
                                                 └─────────────────┘
```

### 实现步骤

1. **获取 GFS 数据** (`inference/predict.py` → `fetch_gfs_data()`)
   - 数据源：NOAA NOMADS — `https://nomads.ncep.noaa.gov/dods/gfs_0p25/`
   - 获取最新的 GFS 0.25° 全球分析数据
   - 提取新英格兰区域

2. **重新网格化**
   - 将 GFS 0.25° 经纬度网格插值到 HRRR 3km Lambert Conformal Conic 投影
   - 目标网格：450 × 449 像素
   - 投影参数见 `data_preparation/data_spec.py` 中的 `PROJECTION`

3. **变量映射**
   - 将 GFS 变量名映射到 HRRR 42 通道格式
   - 注意：GFS 和 HRRR 的变量命名和层次定义可能不同
   - 缺失变量需要用近似值或零填充

4. **模型推理**
   - 加载训练好的 checkpoint
   - 应用与训练时相同的归一化
   - 运行前向传播，反归一化输出

5. **展示结果**
   - CLI 输出 / Web UI / API 接口

### 使用方式（已部分实现）

```bash
python -m inference.predict --checkpoint runs/cnn_baseline/checkpoints/best.pt
```

---

## HPC 操作指南 / HPC Operations

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

## 故障排除 / Troubleshooting

| 问题 | 原因 | 解决方案 |
|------|------|----------|
| OOM Killed | DataLoader worker 占用过多内存 | 减少 `--num_workers` 为 2 |
| NaN loss | 数值不稳定 | 代码已有 NaN 跳过逻辑 |
| DOS line breaks | Windows 换行符 | `sed -i 's/\r$//' scripts/*.slurm` |
| 慢速数据加载 | NFS 网络文件系统 I/O 瓶颈 | 正常现象，每 epoch 约 15-20 分钟 |
| SSH timeout | HPC 节点变更 | 更新 `~/.ssh/config` 中的 HostName |
