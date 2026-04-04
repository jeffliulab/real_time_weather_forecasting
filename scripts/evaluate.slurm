#!/bin/bash
#SBATCH -c 4
#SBATCH -t 0-8:00:00
#SBATCH --mem=64G
#SBATCH -p batch
#SBATCH -o runs/evaluate_%j.out
#SBATCH -e runs/evaluate_%j.err
#SBATCH -J eval_all

# Usage: sbatch scripts/evaluate.slurm

module load class/default
module load cs137/2026spring

cd /cluster/tufts/c26sp1cs0137/pliu07/assignment2

echo "=========================================="
echo "Evaluating all Part 1 models on test set"
echo "Node: $(hostname)"
echo "GPU: $(nvidia-smi --query-gpu=name --format=csv,noheader 2>/dev/null || echo 'N/A')"
echo "Date: $(date)"
echo "=========================================="

PYTHONUNBUFFERED=1 python evaluation/evaluate_all.py
