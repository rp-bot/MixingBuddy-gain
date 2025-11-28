#!/bin/bash
#SBATCH -J debug_alignment             # Job name
#SBATCH -N1 --ntasks-per-node=1        # 1 node / 1 task
#SBATCH --gres=gpu:H100:1              # 1 H100 GPU (uses CUDA if available)
#SBATCH --mem-per-gpu=64GB             # Memory per GPU
#SBATCH -t00:15:00                     # Walltime (15 mins is plenty)
#SBATCH -o outputs/logs/debug-alignment-%j.out  # Log file

# Change to submission directory
cd "$SLURM_SUBMIT_DIR"

# Load environment
module load anaconda3
module load cuda
conda activate ml-env

# Isolate caches per job to avoid quota issues
JOB_CACHE_DIR=/tmp/job_cache_${SLURM_JOB_ID}
mkdir -p "$JOB_CACHE_DIR"
export HF_HOME="$JOB_CACHE_DIR/hf"
export TRANSFORMERS_CACHE="$HF_HOME"
export HF_DATASETS_CACHE="$HF_HOME"
export PIP_CACHE_DIR="$JOB_CACHE_DIR/pip"
mkdir -p "$HF_HOME" "$PIP_CACHE_DIR"

echo "Cache directories set to $JOB_CACHE_DIR"
echo "Job started at:"
date
echo "Running on node:"
hostname
echo "---------------------------------------"
echo "GPU status:"
srun nvidia-smi
echo "---------------------------------------"

echo "Running debug script..."
python debug_alignment.py

echo "Job finished at:"
date
