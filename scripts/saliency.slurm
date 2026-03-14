#!/bin/bash
#SBATCH -n 1
#SBATCH -t 0-02:00:00
#SBATCH --mem=16G
#SBATCH --gres=gpu:1
#SBATCH -o runs/saliency_%j.out
#SBATCH -e runs/saliency_%j.err
#SBATCH -J saliency

module load class/default
module load cs137/2026spring

# Usage (from project root):
#   sbatch scripts/saliency.slurm
#   sbatch scripts/saliency.slurm runs/cnn_baseline/checkpoints/best.pt

cd /cluster/tufts/c26sp1cs0137/pliu07/assignment2

CHECKPOINT=${1:-runs/cnn_baseline/checkpoints/best.pt}

echo "Checkpoint: $CHECKPOINT"
echo "Node: $(hostname)"
echo "Date: $(date)"

python saliency.py --checkpoint $CHECKPOINT --n_samples 300
