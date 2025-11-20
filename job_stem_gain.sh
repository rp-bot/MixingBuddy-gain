#!/bin/bash
#SBATCH -J stem_gain_job                 # Job name
#SBATCH -N1 --ntasks-per-node=1          # 1 node and 1 core
#SBATCH --gres=gpu:H100:1              # Request 1 H100 GPU
#SBATCH --mem-per-gpu=224GB            # Recommended memory for H100
#SBATCH -t10:00:00                       # 10 hours of walltime
#SBATCH -o outputs/logs/stem_gain_report-%j.out   # Output file name (%j = job ID)

# --- Optional Email Notifications ---
#SBATCH --mail-type=END,BEGIN,FAIL      # Send email on job END or FAIL
#SBATCH --mail-user=pvadhulas3@gatech.edu     # Email address (set if you want notifications)

# --- Job Commands ---

# Change to the directory where the job was submitted
cd "$SLURM_SUBMIT_DIR"

# Add ~/.local/bin to PATH so pip-installed scripts are accessible
export PATH="$HOME/.local/bin:$PATH"

# Load modules (adapt to your cluster)
module load anaconda3
module load cuda

conda activate ml-env

# Redirect caches to /tmp to avoid home directory quota limits
JOB_CACHE_DIR=/tmp/job_cache_${SLURM_JOB_ID}
mkdir -p "$JOB_CACHE_DIR"

# HuggingFace cache
export HF_HOME="$JOB_CACHE_DIR/hf"
export TRANSFORMERS_CACHE="$JOB_CACHE_DIR/hf"
export HF_DATASETS_CACHE="$JOB_CACHE_DIR/hf"
mkdir -p "$HF_HOME"

# pip cache
export PIP_CACHE_DIR="$JOB_CACHE_DIR/pip"
mkdir -p "$PIP_CACHE_DIR"

echo "Cache directories set to: $JOB_CACHE_DIR"
echo "  - HuggingFace: $HF_HOME"
echo "  - pip: $PIP_CACHE_DIR"

# Print basic run info
echo "Job started at:"
date
echo "Running on node:"
hostname
echo "---------------------------------------"
echo "NVIDIA GPU Status (nvidia-smi):"
srun nvidia-smi
echo "---------------------------------------"

export CUDA_LAUNCH_BLOCKING=1

echo "---------------------------------------"
echo "Running train_stem_gain_model.py..."
echo "---------------------------------------"

# You can override Hydra config values here, e.g. change output_dir or seed:
# python scripts/train_stem_gain_model.py training.training_args.output_dir=./outputs/checkpoints/stem_gain_model_custom

python scripts/train_stem_gain_model.py

echo "---------------------------------------"
echo "Job finished at:"
date


