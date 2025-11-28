#!/bin/bash
#SBATCH -J align_qformer            # Job name
#SBATCH -N1 --ntasks-per-node=1     # 1 node / 1 task
#SBATCH --gres=gpu:H100:1           # Request 1 H100 GPU
#SBATCH --mem-per-gpu=224GB         # Recommended memory for H100
#SBATCH -t12:00:00                  # Walltime (12h)
#SBATCH -o outputs/logs/alignment-%j.out  # Log file

# Optional email notifications (uncomment + set email)
#SBATCH --mail-type=END,BEGIN,FAIL
#SBATCH --mail-user=pvadhulas3@gatech.edu

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
echo "  - HuggingFace: $HF_HOME"
echo "  - pip: $PIP_CACHE_DIR"

echo "Job started at:"
date
echo "Running on node:"
hostname
echo "---------------------------------------"
echo "GPU status:"
srun nvidia-smi
echo "---------------------------------------"

export CUDA_LAUNCH_BLOCKING=1

# Run alignment training
echo "Launching alignment pretraining..."
python scripts/train_alignment.py 

echo "Job finished at:"
date
