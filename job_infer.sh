#!/bin/bash
#SBATCH -J h100_infer_job           # Job name
#SBATCH -N1 --ntasks-per-node=1         # 1 node and 1 core
#SBATCH --gres=gpu:H100:1               # Request 1 H100 GPU
#SBATCH --mem-per-gpu=224GB             # Recommended memory for H100
#SBATCH -t00:20:00                      # 4 hours of walltime (adjust as needed)
#SBATCH -o outputs/logs/h100_infer-%j.out            # Output file name (%j = job ID)

# --- Optional Email Notifications ---
#SBATCH --mail-type=END,BEGIN,FAIL      # Send email on job events
#SBATCH --mail-user=pvadhulas3@gatech.edu

# --- Job Commands ---

# Change to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Add ~/.local/bin to PATH so pip-installed scripts are accessible
export PATH="$HOME/.local/bin:$PATH"

# Load the CUDA and Anaconda modules
module load anaconda3
module load cuda

conda activate ml-env

# Redirect caches to /tmp to avoid home directory quota limits
# Create a unique cache directory for this job to avoid conflicts
JOB_CACHE_DIR=/tmp/job_cache_${SLURM_JOB_ID}
mkdir -p $JOB_CACHE_DIR

# Redirect HuggingFace cache
export HF_HOME=$JOB_CACHE_DIR/hf
export TRANSFORMERS_CACHE=$JOB_CACHE_DIR/hf
export HF_DATASETS_CACHE=$JOB_CACHE_DIR/hf
mkdir -p $HF_HOME

# Redirect pip cache (uses significant space)
export PIP_CACHE_DIR=$JOB_CACHE_DIR/pip
mkdir -p $PIP_CACHE_DIR

echo "Cache directories set to: $JOB_CACHE_DIR"
echo "  - HuggingFace: $HF_HOME"
echo "  - pip: $PIP_CACHE_DIR"

# Print the date, the node's hostname, and GPU status
echo "Job started at:"
date
echo "Running on node:"
hostname
echo "---------------------------------------"
echo "NVIDIA GPU Status (nvidia-smi):"
srun nvidia-smi
echo "---------------------------------------"

# Run inference/generation script
echo "---------------------------------------"
echo "Running 08_generate_samples.py..."
echo "---------------------------------------"
python scripts/08_generate_samples.py
echo "---------------------------------------"

echo "Job finished at:"
date


