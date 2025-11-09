#!/bin/bash
#SBATCH -J h100_job                 # Job name
#SBATCH -N1 --ntasks-per-node=1        # 1 node and 1 core
#SBATCH --gres=gpu:H100:1              # Request 1 H100 GPU
#SBATCH --mem-per-gpu=224GB            # Recommended memory for H100
#SBATCH -t10:00:00                      # 10 hours of walltime
#SBATCH -o outputs/logs/h100_report-%j.out          # Output file name (%j = job ID)

# --- Optional Email Notifications ---
#SBATCH --mail-type=END,BEGIN,FAIL           # Send email on job END or FAIL
#SBATCH --mail-user=pvadhulas3@gatech.edu # Email address (change this to yours!)

# --- Job Commands ---

# Change to the directory where the job was submitted
cd $SLURM_SUBMIT_DIR

# Add ~/.local/bin to PATH so pip-installed scripts are accessible
export PATH="$HOME/.local/bin:$PATH"

# Load the CUDA module to make GPU tools available
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

# # Test imports and CUDA availability
# echo "---------------------------------------"
# echo "Testing imports and CUDA availability..."
# echo "---------------------------------------"
# python scripts/06_train_model.py
# echo "---------------------------------------"
export CUDA_LAUNCH_BLOCKING=1
# run the dpo training
echo "---------------------------------------"
echo "Running 08_train_dpo.py..."
echo "---------------------------------------"
python scripts/08_train_dpo.py
echo "---------------------------------------"

# --- Your code goes here ---
#
# srun ./your_gpu_executable
#
# srun python your_gpu_script.py
#

echo "Job finished at:"
date