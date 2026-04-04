#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-24:00:00
#SBATCH --mem=48G
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH -o runs/%x_%j.out
#SBATCH -e runs/%x_%j.err
#SBATCH -J weather_cnn

# Usage (from project root):
#   sbatch scripts/train.slurm                                          # baseline CNN
#   sbatch --job-name=resnet18 scripts/train.slurm resnet18 --patience 10        # ResNet-18
#   sbatch --job-name=convnext scripts/train.slurm convnext_tiny --patience 10   # ConvNeXt-Tiny
#   sbatch --job-name=multi_frame scripts/train.slurm cnn_multi_frame   # multi-frame
#   sbatch --job-name=cnn_3d scripts/train.slurm cnn_3d                 # 3D CNN
#   sbatch --job-name=vit scripts/train.slurm vit                       # ViT

module load class/default
module load cs137/2026spring

cd /cluster/tufts/c26sp1cs0137/pliu07/assignment2

MODEL=${1:-cnn_baseline}
EXTRA_ARGS="${@:2}"

echo "=========================================="
echo "Model: $MODEL"
echo "Extra args: $EXTRA_ARGS"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

python training/train.py --model $MODEL $EXTRA_ARGS
