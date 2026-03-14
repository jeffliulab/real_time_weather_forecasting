# Regional Weather Forecasting with Deep Learning

A deep learning system that forecasts weather conditions 24 hours ahead at a target location in New England, using spatial weather snapshots from HRRR reanalysis data.

## Overview

| Component | Description |
|-----------|-------------|
| **Input** | Spatial weather map `(450, 449, 42)` — 42 atmospheric variables over a 3km Lambert Conformal grid |
| **Output** | 6 continuous variables (temperature, humidity, u-wind, v-wind, gust, precipitation) + binary rain label |
| **Lead time** | 24 hours ahead |

## Project Structure

```
weather-forecasting/
├── train.py                    # Training entry point
├── saliency.py                 # Saliency analysis (Part 2)
│
├── data/                       # Raw data files (gitignored)
│   └── dataset/                # .pt snapshots, targets.pt, metadata.pt
│
├── data_preparation/           # Data loading, processing & preparation (git tracked)
│   ├── dataset.py              # WeatherDataset, DataLoader utilities
│   ├── data_spec.py            # Variable names, levels, grid projection
│   └── generate_dataset.py     # Reference: how the .pt dataset was generated
│
├── models/                     # Model architectures
│   ├── __init__.py             # Model registry & factory
│   ├── cnn_baseline.py         # Part 1: Single-frame 2D CNN
│   ├── cnn_multi_frame.py      # Part 3: Multi-frame 2D CNN
│   ├── cnn_3d.py               # Part 3: 3D CNN (spatiotemporal)
│   └── vit.py                  # Part 3: Vision Transformer
│
├── inference/                  # Part 4: Real-time prediction pipeline
│   └── predict.py              # Fetch live data → model → forecast
│
├── evaluation/                 # Official evaluation framework
│   ├── evaluate.py             # Evaluation script
│   └── baseline_cnn/model.py   # Evaluation wrapper for baseline CNN
│
├── scripts/                    # Operational scripts
│   ├── train.slurm             # SLURM job script for training
│   ├── saliency.slurm          # SLURM job script for saliency
│   └── sync.ps1                # Sync local code → HPC
│
├── tests/                      # Tests & validation
│   └── smoke_test.py           # Quick forward-pass check
│
├── docs/                       # Reference documentation
│   ├── 说明                     # HPC setup instructions
│   └── ssh说明                  # SSH configuration notes
│
├── runs/                       # Training outputs (gitignored)
│   └── <model_name>/
│       ├── checkpoints/        # best.pt, latest.pt
│       ├── logs/               # training_log.csv
│       └── figures/            # training_curves.png
│
└── PIPELINE.md                 # Detailed pipeline documentation
```

## Quick Start

```bash
# 1. Sync code to HPC
powershell -File scripts/sync.ps1

# 2. Submit training job (from HPC)
cd $PROJECT_ROOT  # remote project directory
sbatch scripts/train.slurm                        # baseline CNN
sbatch --job-name=vit scripts/train.slurm vit     # Vision Transformer

# 3. Monitor training
squeue -u $USER
tail -f runs/weather_cnn_*.out

# 4. Run saliency analysis
sbatch scripts/saliency.slurm runs/cnn_baseline/checkpoints/best.pt
```

## Models

| Model | Params | Temporal | Architecture |
|-------|--------|----------|-------------|
| `cnn_baseline` | ~1.8M | 1 frame | 2D CNN with residual blocks |
| `cnn_multi_frame` | ~2.5M | 4 frames | Channel-stacked multi-frame 2D CNN |
| `cnn_3d` | ~1.2M | 4 frames | 3D convolutions for spatiotemporal learning |
| `vit` | ~3.5M | 1 frame | Vision Transformer with patch embeddings |

## Evaluation Metrics

- **RMSE** per variable (denormalized to physical units)
- **Conditional RMSE** for precipitation (only when actual precip > 2mm)
- **AUC** for binary rain/no-rain classification
