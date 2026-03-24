#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=train_snmf
#SBATCH --output=logs/train_%j.out
#SBATCH --error=logs/train_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

# --- Environment Setup ---
source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate snmf_env

# --- Space & Cache Management ---
export HF_HOME="/home/morg/students/rashkovits/hf_cache"
export TORCH_HOME="/home/morg/students/rashkovits/hf_cache/torch"
export TMPDIR="/home/morg/students/rashkovits/hf_cache"

# --- Project Setup ---
cd /home/morg/students/rashkovits/snmf-mlp-decomposition
export PYTHONPATH=$PYTHONPATH:$(pwd)

mkdir -p logs outputs/snmf_train_results $HF_HOME

# --- Parallelism Optimization ---
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Execute Training ---
echo "--------------------------------------------------------"
echo "Starting SNMF Training on Node: $SLURMD_NODENAME"
echo "--------------------------------------------------------"

python train_snmf.py \
    --model-path "models/gemma2-2.03B_best_unlearn_model" \
    --data-path "data/data_subsampled.json" \
    --output-dir "outputs/snmf_train_results" \
    --layers "0,3,5,13" \
    --rank 25 \
    --mode "mlp_intermediate" \
    --batch-size 1 \
    --device "cpu" \
    --sparsity 0.01 \
    --max-iter 5000 \
    --seed 42

echo "--------------------------------------------------------"
echo "SNMF Training Finished"