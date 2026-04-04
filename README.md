<div align="center">

[![en](https://img.shields.io/badge/lang-English-blue.svg)](README.md)
[![zh](https://img.shields.io/badge/lang-中文-red.svg)](README_zh.md)

<h1>Regional Weather Forecasting with Deep Learning</h1>

<p>
  <img src="https://img.shields.io/badge/python-3.10+-blue?logo=python&logoColor=white" alt="Python">
  <img src="https://img.shields.io/badge/PyTorch-2.1+-ee4c2c?logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/Gradio-5.x-orange?logo=gradio&logoColor=white" alt="Gradio">
  <img src="https://img.shields.io/badge/Status-Complete-brightgreen" alt="Status">
  <img src="https://img.shields.io/badge/License-MIT-green" alt="License">
</p>

<p>
  <strong>24-hour weather prediction</strong> at Tufts University using deep learning on <strong>42-channel HRRR 3 km reanalysis data</strong>.
</p>

<p>
  <a href="https://huggingface.co/spaces/jeffliulab/weather_predict"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Live%20Demo-Weather%20Forecast-blue" alt="Live Demo"></a>
  <a href="https://huggingface.co/jeffliulab/weather-forecasting-v1"><img src="https://img.shields.io/badge/%F0%9F%A4%97%20Model-Weights-yellow" alt="Model Weights"></a>
</p>

</div>

---

## Highlights

- **6 model architectures** trained and compared: CNN Baseline, ResNet-18, ConvNeXt-Tiny, Multi-frame CNN, 3D CNN, Vision Transformer
- **Real-time inference** fetching live HRRR data from NOAA AWS S3 via Herbie
- **Live demo** deployed on HuggingFace Spaces with satellite/street/temperature map visualization
- **Complete pipeline** from data preparation to training, evaluation, saliency analysis, and deployment

---

## Table of Contents

- [Highlights](#highlights)
- [Live Demo](#live-demo)
- [Task Definition](#task-definition)
- [Data](#data)
- [Architectures](#architectures)
- [Results](#results)
- [Saliency Analysis](#saliency-analysis)
- [Project Structure](#project-structure)
- [Quick Start](#quick-start)
- [Acknowledgments](#acknowledgments)

---

## Live Demo

The demo is deployed at **[huggingface.co/spaces/jeffliulab/weather_predict](https://huggingface.co/spaces/jeffliulab/weather_predict)**.

Click **Run Forecast** to:
1. Fetch the latest 42-channel HRRR analysis from NOAA (real-time, ~30s)
2. Run inference with a trained CNN
3. Display the 24-hour forecast + three-panel map (satellite, reference, temperature)

---

## Task Definition

| Component | Detail |
|-----------|--------|
| **Input** | `(B, 42, 450, 449)` spatial snapshot — 42 atmospheric channels, 3 km Lambert Conformal grid |
| **Output** | 6 continuous values + 1 binary rain label |
| **Lead time** | 24 hours |
| **Target location** | Jumbo Statue, Tufts University (42.41°N, 71.12°W) |

### Prediction Targets

| # | Variable | Unit | Description |
|---|----------|------|-------------|
| 1 | TMP@2m | K | 2-meter temperature |
| 2 | RH@2m | % | 2-meter relative humidity |
| 3 | UGRD@10m | m/s | 10-meter u-component wind |
| 4 | VGRD@10m | m/s | 10-meter v-component wind |
| 5 | GUST@surface | m/s | Surface gust speed |
| 6 | APCP_1hr@surface | mm | 1-hour accumulated precipitation |
| 7 | APCP > 2mm | binary | Rain / no-rain classification |

### Metrics

| Metric | Applies to | Definition |
|--------|-----------|------------|
| **RMSE** | TMP, RH, UGRD, VGRD, GUST | Root mean square error in physical units |
| **Conditional RMSE** | APCP | RMSE computed only when true APCP > 2 mm |
| **AUC** | Rain / no-rain | Area under the ROC curve |

<details>
<summary><strong>All 42 Input Channels</strong> (click to expand)</summary>

**Surface / Near-Surface (7 target variables)**

| # | Variable | Level | Description |
|---|----------|-------|-------------|
| 0 | TMP | 2m above ground | Temperature |
| 1 | RH | 2m above ground | Relative humidity |
| 2 | UGRD | 10m above ground | U-component wind |
| 3 | VGRD | 10m above ground | V-component wind |
| 4 | GUST | surface | Surface gust speed |
| 5 | DSWRF | surface | Downward shortwave radiation flux |
| 6 | APCP_1hr | surface | 1-hour accumulated precipitation |

**Atmospheric Variables (35 channels)**

| Channels | Variable | Levels |
|----------|----------|--------|
| 7 | CAPE | surface |
| 8–12 | DPT | 1000, 500, 700, 850, 925 mb |
| 13–17 | HGT | 1000, 500, 700, 850 mb, surface |
| 18–22 | TMP | 1000, 500, 700, 850, 925 mb |
| 23–28 | UGRD | 1000, 250, 500, 700, 850, 925 mb |
| 29–34 | VGRD | 1000, 250, 500, 700, 850, 925 mb |
| 35–38 | Cloud cover | TCDC, HCDC, MCDC, LCDC |
| 39–41 | Moisture | PWAT, RHPW, VIL |

</details>

---

## Data

| Property | Value |
|----------|-------|
| **Source** | HRRR (High-Resolution Rapid Refresh) reanalysis from NOAA |
| **Region** | US Northeast / New England (~1350 km × 1350 km) |
| **Grid** | Lambert Conformal, 3 km resolution, 450 × 449 pixels |
| **Channels** | 42 atmospheric variables |
| **Temporal** | Hourly, Jul 2018 – Jul 2021 |

| Split | Years | Samples |
|-------|-------|---------|
| Train | 2018–2019 | ~17,500 |
| Validation | 2020 | ~8,700 |
| Test | 2021 | ~8,700 |

**Preprocessing:** Per-channel z-score normalization from 1,000 random training samples. Four-stage NaN filtering at dataset, batch, loss, and metric levels.

---

## Architectures

Six architectures were implemented and compared:

| Model | Class | Params | Input | Key Design |
|-------|-------|--------|-------|-----------|
| `cnn_baseline` | BaselineCNN | 11.3M | 1 frame | 6 ResBlocks, progressive downsample |
| `resnet18` | ResNet18Baseline | 11.2M | 1 frame | Modified torchvision ResNet-18 |
| `convnext_tiny` | ConvNeXtBaseline | 7.7M | 1 frame | Modified torchvision ConvNeXt-Tiny |
| `cnn_multi_frame` | MultiFrameCNN | 11.4M | 4 frames | Channel-stack 4×42→168, temporal mixing stem |
| `cnn_3d` | CNN3D | — | 4 frames | 3D Conv with temporal collapse via stride-2 |
| `vit` | WeatherViT | 2.3M | 1 frame | 15×15 patches → 900 tokens, 6-layer Transformer |

<details>
<summary><strong>Architecture Diagrams</strong> (click to expand)</summary>

**BaselineCNN**
```
Input (B,42,450,449) → Stem(42→64, 7×7, s=2) → 6×ResBlock → GAP → FC Head → (B,6)
                        225×225 → 113 → 57 → 29 → 15 → 8×8
```

**WeatherViT**
```
Input (B,42,450,449) → pad→450×450 → PatchEmbed(15×15, 900 patches)
  → [CLS]+PosEmbed → 6×TransformerBlock(8 heads, dim=256) → CLS → FC → (B,6)
```

**CNN3D**
```
Input (B,4,42,450,449) → 3D Stem → 6×Res3D (temporal collapse in layers 2-3) → Pool3D → FC → (B,6)
```

</details>

---

## Results

### Test Set (2021) — Model Comparison

| Model | TMP (K) | RH (%) | UGRD (m/s) | VGRD (m/s) | GUST (m/s) | APCP>2mm (mm) | AUC |
|-------|---------|--------|------------|------------|------------|--------------|-----|
| **ViT** | 4.06 | 16.45 | 2.59 | **2.21** | **3.57** | **4.50** | **0.776** |
| **ResNet-18** | **3.54** | **15.68** | 2.70 | 2.34 | 3.60 | 4.53 | 0.768 |
| CNN Baseline | 4.00 | 15.89 | **2.56** | 2.23 | 3.58 | 4.56 | 0.738 |
| ConvNeXt-Tiny | 3.66 | 15.85 | 2.54 | 2.17 | 3.65 | 4.55 | 0.692 |
| CNN Multi-frame | 4.55 | 18.41 | 2.62 | 2.45 | 3.62 | 4.76 | 0.652 |
| CNN 3D | 4.76 | 17.44 | 2.61 | 2.32 | 3.58 | 4.75 | 0.668 |
| *Persistence* | *4.86* | *23.01* | *3.73* | *2.89* | *4.87* | *4.62* | *0.506* |

**Key findings:**
- **ViT** achieves the best rain detection AUC (0.776) and precipitation RMSE
- **ResNet-18** leads in temperature (3.54 K) and humidity (15.68%) accuracy
- **Multi-frame models underperform** — temporal stacking does not help with this data/architecture
- All trained models significantly beat the persistence baseline

### Training Configuration

| Setting | Value |
|---------|-------|
| Optimizer | AdamW (lr=1e-3, weight_decay=1e-4) |
| Scheduler | CosineAnnealingLR |
| Loss | MSELoss (equal weight) |
| Gradient clipping | max_norm=1.0 |
| GPU | NVIDIA L40S / A100 / H200 |

---

## Saliency Analysis

Gradient-based saliency maps reveal which spatial regions influence predictions most:

- **Westerly dominance**: Western regions contribute 1.12× more than eastern (consistent with prevailing winds)
- **Southern moisture**: Southern regions contribute 1.18× more than northern (subtropical moisture transport)
- **Distance decay**: Nearby regions (0–75 km) have 1.41× average influence vs. distant regions
- **Variable-specific patterns**: Humidity/wind use broad spatial context; precipitation relies on local features only

---

## Project Structure

```
real_time_weather_forecasting/
├── README.md                         # This file
├── models/                           # Model architectures (6 models)
│   ├── __init__.py                   #   Model registry & factory
│   ├── cnn_baseline.py               #   BaselineCNN
│   ├── resnet_baseline.py            #   ResNet-18
│   ├── convnext_baseline.py          #   ConvNeXt-Tiny
│   ├── cnn_multi_frame.py            #   Multi-frame CNN
│   ├── cnn_3d.py                     #   3D CNN
│   └── vit.py                        #   Vision Transformer
├── training/                         # Training pipeline
│   ├── train.py                      #   Training entry point
│   ├── saliency.py                   #   Saliency analysis
│   └── data_preparation/             #   Dataset loading & preprocessing
├── evaluation/                       # Evaluation framework
│   ├── evaluate.py                   #   Single model evaluation
│   ├── evaluate_all.py               #   Multi-model comparison
│   └── */model.py                    #   Per-model evaluation wrappers
├── inference/                        # CLI inference pipeline
│   └── predict.py
├── space/                            # HuggingFace Space deployment
│   ├── app.py                        #   Gradio web UI
│   ├── hrrr_fetch.py                 #   Real-time HRRR data fetching
│   ├── model_utils.py                #   Model loading & inference
│   ├── visualization.py              #   Three-panel map rendering
│   ├── var_mapping.py                #   42-channel HRRR GRIB2 mapping
│   ├── checkpoints/                  #   Stripped model weights (~45MB each)
│   └── models/                       #   Model architecture code (copy)
├── runs/                             # Training outputs (logs, figures, configs)
├── scripts/                          # SLURM jobs, HF upload, deploy
├── docs/                             # Documentation
└── tests/                            # Smoke tests & debugging tools
```

---

## Quick Start

### Run the Live Demo Locally

```bash
cd space
pip install -r requirements.txt
python app.py
# Open http://127.0.0.1:7860
```

### Train a Model (on HPC)

```bash
# Sync code to cluster
powershell -File scripts/sync.ps1

# Submit training job
sbatch scripts/train.slurm cnn_baseline    # or: resnet18, vit, cnn_3d, ...

# Monitor
tail -f runs/cnn_baseline/logs/training_log.csv
```

### Evaluate

```bash
python evaluation/evaluate_all.py
```

### Deploy to HuggingFace Space

```bash
python scripts/deploy_space.py --space_id jeffliulab/weather_predict
```

---

## Links

| Resource | URL |
|----------|-----|
| Live Demo | [huggingface.co/spaces/jeffliulab/weather_predict](https://huggingface.co/spaces/jeffliulab/weather_predict) |
| Model Weights | [huggingface.co/jeffliulab/weather-forecasting-v1](https://huggingface.co/jeffliulab/weather-forecasting-v1) |
| GitHub | [github.com/jeffliulab/real_time_weather_forecasting](https://github.com/jeffliulab/real_time_weather_forecasting) |

---

## Acknowledgments

- **Data**: HRRR reanalysis from NOAA, fetched via [Herbie](https://herbie.readthedocs.io/)
- **Compute**: Tufts Research Technology HPC (NVIDIA L40S / A100 / H200)
- **Course**: Tufts CS 137 — Deep Neural Networks, Spring 2026
