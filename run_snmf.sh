#!/bin/bash

# --- Slurm Configuration ---
#SBATCH --job-name=snmf_shir
#SBATCH --output=logs/snmf_%j.out
#SBATCH --error=logs/snmf_%j.err
#SBATCH --time=24:00:00
#SBATCH --partition=studentkillable
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=16
#SBATCH --mem=80G

# --- Environment Setup ---
# Load conda profile and activate environment
source /home/morg/students/rashkovits/miniconda3/etc/profile.d/conda.sh
conda activate snmf_env

# --- Space & Cache Management ---
# Directing HF and Torch to your large storage area
export HF_HOME="/home/morg/students/rashkovits/hf_cache"
export TORCH_HOME="/home/morg/students/rashkovits/hf_cache/torch"
export TMPDIR="/home/morg/students/rashkovits/hf_cache"

# --- Project Setup ---
# Set working directory to project root
cd /home/morg/students/rashkovits/snmf-mlp-decomposition
export PYTHONPATH=$PYTHONPATH:$(pwd)

# Ensure necessary directories exist
mkdir -p logs outputs/snmf_results $HF_HOME

# --- Parallelism Optimization ---
# Ensure PyTorch uses all 16 allocated CPUs for the SNMF math
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK
export MKL_NUM_THREADS=$SLURM_CPUS_PER_TASK

# --- Sanity Check ---
echo "Active environment: $CONDA_DEFAULT_ENV"
python -c "import torch; print('SUCCESS: PyTorch loaded from ' + torch.__file__)"

echo "--------------------------------------------------------"
echo "Starting SNMF Analysis on Node: $SLURMD_NODENAME"
echo "--------------------------------------------------------"

# --- Execute Analysis ---
# Note: --device "cpu" is used for the factorization stage
# while the model uses the GPU for activation generation.
python run_snmf.py \
    --model-path "models/gemma2-2.03B_best_unlearn_model" \
    --data-path "data/data.json" \
    --output-dir "outputs/snmf_results" \
    --layers "0,3,5,13" \
    --rank 25 \
    --mode "mlp_intermediate" \
    --batch-size 1 \
    --device "cpu" \
    --sparsity 0.01 \
    --max-iter 5000 \
    --seed 42

echo "--------------------------------------------------------"
echo "SNMF Analysis Finished"