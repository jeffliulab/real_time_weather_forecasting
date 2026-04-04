<div align="center">

[![en](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![zh](https://img.shields.io/badge/lang-中文-red.svg)](README_zh.md)

<h1>基于深度学习的区域天气预报</h1>

<p>
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Gradio-5.x-orange?logo=gradio&logoColor=white" alt="Gradio">
  <img src="https://img.shields.io/badge/状态-已完成-brightgreen" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p>
  利用 <strong>42 通道 HRRR 3 km 再分析数据</strong>，对塔夫茨大学进行 <strong>24 小时天气预测</strong>。
</p>

<p>
  <a href="https://huggingface.co/spaces/jeffliulab/weather_predict"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20在线演示-天气预报-blue" alt="Live Demo"></a>
  <a href="https://huggingface.co/jeffliulab/weather-forecasting-v1"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20模型-权重下载-yellow" alt="Model Weights"></a>
</p>

</div>

---

## 项目亮点

- **6 种模型架构** 训练并对比：CNN Baseline、ResNet-18、ConvNeXt-Tiny、多帧 CNN、3D CNN、Vision Transformer
- **实时推理** 通过 Herbie 从 NOAA AWS S3 获取最新 HRRR 数据
- **在线 Demo** 部署在 HuggingFace Spaces，提供卫星/地图/温度三联图可视化
- **完整流水线** 覆盖数据准备、训练、评估、显著性分析到部署

---

## 目录

- [项目亮点](#项目亮点)
- [在线演示](#在线演示)
- [任务定义](#任务定义)
- [数据](#数据)
- [模型架构](#模型架构)
- [实验结果](#实验结果)
- [显著性分析](#显著性分析)
- [项目结构](#项目结构)
- [快速开始](#快速开始)
- [致谢](#致谢)

---

## 在线演示

Demo 已部署在 **[huggingface.co/spaces/jeffliulab/weather_predict](https://huggingface.co/spaces/jeffliulab/weather_predict)**。

点击 **Run Forecast** 即可：
1. 从 NOAA 获取最新的 42 通道 HRRR 分析数据（实时，约 30 秒）
2. 使用训练好的 CNN 进行推理
3. 展示 24 小时预报结果 + 三联地图（卫星、参考地图、温度场）

---

## 任务定义

| 项目 | 详情 |
|------|------|
| **输入** | `(B, 42, 450, 449)` 空间快照 — 42 个大气通道，Lambert Conformal 投影，3 km 分辨率 |
| **输出** | 6 个连续值 + 1 个降雨二分类标签 |
| **预报时效** | 24 小时 |
| **目标位置** | 塔夫茨大学 Jumbo 雕像（42.41°N, 71.12°W） |

### 预测变量

| # | 变量 | 单位 | 描述 |
|---|------|------|------|
| 1 | TMP@2m | K | 2 米温度 |
| 2 | RH@2m | % | 2 米相对湿度 |
| 3 | UGRD@10m | m/s | 10 米纬向风 |
| 4 | VGRD@10m | m/s | 10 米经向风 |
| 5 | GUST@surface | m/s | 地面阵风 |
| 6 | APCP_1hr@surface | mm | 1 小时累积降水 |
| 7 | APCP > 2mm | 二分类 | 有雨 / 无雨 |

### 评估指标

| 指标 | 适用变量 | 定义 |
|------|---------|------|
| **RMSE** | TMP, RH, UGRD, VGRD, GUST | 物理单位的均方根误差 |
| **条件 RMSE** | APCP | 仅在真实 APCP > 2 mm 时计算 |
| **AUC** | 降雨分类 | ROC 曲线下面积 |

<details>
<summary><strong>全部 42 个输入通道</strong>（点击展开）</summary>

**地面/近地面（7 个目标变量）**

| # | 变量 | 层级 | 描述 |
|---|------|------|------|
| 0 | TMP | 地面以上 2m | 温度 |
| 1 | RH | 地面以上 2m | 相对湿度 |
| 2 | UGRD | 地面以上 10m | 纬向风 |
| 3 | VGRD | 地面以上 10m | 经向风 |
| 4 | GUST | 地面 | 阵风 |
| 5 | DSWRF | 地面 | 向下短波辐射 |
| 6 | APCP_1hr | 地面 | 1 小时累积降水 |

**大气变量（35 个通道）**

| 通道 | 变量 | 层级 |
|------|------|------|
| 7 | CAPE | 地面 |
| 8–12 | DPT（露点温度） | 1000, 500, 700, 850, 925 mb |
| 13–17 | HGT（位势高度） | 1000, 500, 700, 850 mb, 地面 |
| 18–22 | TMP（温度） | 1000, 500, 700, 850, 925 mb |
| 23–28 | UGRD（纬向风） | 1000, 250, 500, 700, 850, 925 mb |
| 29–34 | VGRD（经向风） | 1000, 250, 500, 700, 850, 925 mb |
| 35–38 | 云量 | TCDC, HCDC, MCDC, LCDC |
| 39–41 | 水汽 | PWAT, RHPW, VIL |

</details>

---

## 数据

| 属性 | 值 |
|------|---|
| **来源** | NOAA HRRR（高分辨率快速更新）再分析数据 |
| **覆盖区域** | 美国东北部/新英格兰（约 1350 km × 1350 km） |
| **网格** | Lambert Conformal 投影，3 km 分辨率，450 × 449 像素 |
| **通道数** | 42 个大气变量 |
| **时间范围** | 2018 年 7 月 – 2021 年 7 月，逐小时 |

| 划分 | 年份 | 样本数 |
|------|------|--------|
| 训练集 | 2018–2019 | ~17,500 |
| 验证集 | 2020 | ~8,700 |
| 测试集 | 2021 | ~8,700 |

**预处理：** 基于 1,000 个随机训练样本的逐通道 z-score 标准化。四级 NaN 过滤（数据集→批次→损失→指标）。

---

## 模型架构

共实现并对比了 6 种架构：

| 模型 | 类名 | 参数量 | 输入 | 核心设计 |
|------|------|--------|------|---------|
| `cnn_baseline` | BaselineCNN | 11.3M | 单帧 | 6 个残差块，逐步下采样 |
| `resnet18` | ResNet18Baseline | 11.2M | 单帧 | 修改的 torchvision ResNet-18 |
| `convnext_tiny` | ConvNeXtBaseline | 7.7M | 单帧 | 修改的 torchvision ConvNeXt-Tiny |
| `cnn_multi_frame` | MultiFrameCNN | 11.4M | 4 帧 | 通道堆叠 4×42→168 + 时间混合层 |
| `cnn_3d` | CNN3D | — | 4 帧 | 3D 卷积，时间维度逐步压缩 |
| `vit` | WeatherViT | 2.3M | 单帧 | 15×15 patch → 900 token，6 层 Transformer |

<details>
<summary><strong>架构示意图</strong>（点击展开）</summary>

**BaselineCNN**
```
输入 (B,42,450,449) → Stem(42→64, 7×7, s=2) → 6×ResBlock → GAP → FC Head → (B,6)
                       225×225 → 113 → 57 → 29 → 15 → 8×8
```

**WeatherViT**
```
输入 (B,42,450,449) → pad→450×450 → PatchEmbed(15×15, 900 patches)
  → [CLS]+位置编码 → 6×TransformerBlock(8 heads, dim=256) → CLS → FC → (B,6)
```

**CNN3D**
```
输入 (B,4,42,450,449) → 3D Stem → 6×Res3D（时间维度在第2-3层压缩）→ Pool3D → FC → (B,6)
```

</details>

---

## 实验结果

### 测试集 (2021) — 模型对比

| 模型 | TMP (K) | RH (%) | UGRD (m/s) | VGRD (m/s) | GUST (m/s) | APCP>2mm (mm) | AUC |
|------|---------|--------|------------|------------|------------|--------------|-----|
| **ViT** | 4.06 | 16.45 | 2.59 | **2.21** | **3.57** | **4.50** | **0.776** |
| **ResNet-18** | **3.54** | **15.68** | 2.70 | 2.34 | 3.60 | 4.53 | 0.768 |
| CNN Baseline | 4.00 | 15.89 | **2.56** | 2.23 | 3.58 | 4.56 | 0.738 |
| ConvNeXt-Tiny | 3.66 | 15.85 | 2.54 | 2.17 | 3.65 | 4.55 | 0.692 |
| 多帧 CNN | 4.55 | 18.41 | 2.62 | 2.45 | 3.62 | 4.76 | 0.652 |
| 3D CNN | 4.76 | 17.44 | 2.61 | 2.32 | 3.58 | 4.75 | 0.668 |
| *持续性基准* | *4.86* | *23.01* | *3.73* | *2.89* | *4.87* | *4.62* | *0.506* |

**主要发现：**
- **ViT** 在降雨检测 AUC（0.776）和降水 RMSE 上表现最佳
- **ResNet-18** 在温度（3.54 K）和湿度（15.68%）精度上领先
- **多帧模型表现不佳** — 在当前数据和架构下，时间堆叠未带来收益
- 所有训练模型均显著优于持续性基准

### 训练配置

| 设置 | 值 |
|------|---|
| 优化器 | AdamW (lr=1e-3, weight_decay=1e-4) |
| 调度器 | CosineAnnealingLR |
| 损失函数 | MSELoss（等权重） |
| 梯度裁剪 | max_norm=1.0 |
| GPU | NVIDIA L40S / A100 / H200 |

---

## 显著性分析

基于梯度的显著性图揭示了哪些空间区域对预测影响最大：

- **西风主导**：西部区域贡献比东部高 1.12 倍（与盛行西风一致）
- **南方水汽**：南部区域贡献比北部高 1.18 倍（副热带水汽输送）
- **距离衰减**：近距离区域（0–75 km）平均影响力为远距离区域的 1.41 倍
- **变量差异**：湿度/风速利用广域空间信息；降水仅依赖局部特征

---

## 项目结构

```
real_time_weather_forecasting/
├── README.md                         # 项目文档（英文）
├── README_zh.md                      # 项目文档（中文）
├── models/                           # 模型架构（6 种模型）
│   ├── __init__.py                   #   模型注册工厂
│   ├── cnn_baseline.py               #   BaselineCNN
│   ├── resnet_baseline.py            #   ResNet-18
│   ├── convnext_baseline.py          #   ConvNeXt-Tiny
│   ├── cnn_multi_frame.py            #   多帧 CNN
│   ├── cnn_3d.py                     #   3D CNN
│   └── vit.py                        #   Vision Transformer
├── training/                         # 训练流水线
│   ├── train.py                      #   训练入口
│   ├── saliency.py                   #   显著性分析
│   └── data_preparation/             #   数据加载与预处理
├── evaluation/                       # 评估框架
│   ├── evaluate.py                   #   单模型评估
│   ├── evaluate_all.py               #   多模型对比
│   └── */model.py                    #   各模型评估封装
├── inference/                        # 命令行推理
│   └── predict.py
├── space/                            # HuggingFace Space 部署
│   ├── app.py                        #   Gradio 网页界面
│   ├── hrrr_fetch.py                 #   实时 HRRR 数据获取
│   ├── model_utils.py                #   模型加载与推理
│   ├── visualization.py              #   三联地图渲染
│   ├── var_mapping.py                #   42 通道 HRRR GRIB2 映射
│   ├── checkpoints/                  #   精简模型权重（~45MB/个）
│   └── models/                       #   模型架构代码（副本）
├── runs/                             # 训练产出（日志、图表、配置）
├── scripts/                          # SLURM 作业、HF 上传、部署脚本
├── docs/                             # 文档
└── tests/                            # 测试工具
```

---

## 快速开始

### 本地运行在线 Demo

```bash
cd space
pip install -r requirements.txt
python app.py
# 打开 http://127.0.0.1:7860
```

### 训练模型（HPC 集群）

```bash
# 同步代码到集群
powershell -File scripts/sync.ps1

# 提交训练任务
sbatch scripts/train.slurm cnn_baseline    # 或: resnet18, vit, cnn_3d, ...

# 查看训练日志
tail -f runs/cnn_baseline/logs/training_log.csv
```

### 评估

```bash
python evaluation/evaluate_all.py
```

### 部署到 HuggingFace Space

```bash
python scripts/deploy_space.py --space_id jeffliulab/weather_predict
```

---

## 相关链接

| 资源 | 地址 |
|------|------|
| 在线 Demo | [huggingface.co/spaces/jeffliulab/weather_predict](https://huggingface.co/spaces/jeffliulab/weather_predict) |
| 模型权重 | [huggingface.co/jeffliulab/weather-forecasting-v1](https://huggingface.co/jeffliulab/weather-forecasting-v1) |
| GitHub | [github.com/jeffliulab/real_time_weather_forecasting](https://github.com/jeffliulab/real_time_weather_forecasting) |

---

## 致谢

- **数据**：NOAA HRRR 再分析数据，通过 [Herbie](https://herbie.readthedocs.io/) 获取
- **算力**：Tufts Research Technology HPC（NVIDIA L40S / A100 / H200）
- **课程**：Tufts CS 137 — Deep Neural Networks, Spring 2026
