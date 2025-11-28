#!/bin/bash
#SBATCH -J eval_align_qformer          # Job name
#SBATCH -N1 --ntasks-per-node=1        # 1 node / 1 task
#SBATCH --gres=gpu:H100:1              # 1 H100 GPU (uses CUDA if available)
#SBATCH --mem-per-gpu=64GB             # Memory per GPU (eval needs less than training)
#SBATCH -t01:00:00                     # Walltime (1h should be plenty for eval)
#SBATCH -o outputs/logs/eval-alignment-%j.out  # Log file

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

# Directory containing qformer_final.pt or qformer_best.pt from alignment training
CHECKPOINT_DIR="outputs/checkpoints/mixing_buddy_milestone_0/alignment_pretraining_contrastive_only"

echo "Launching alignment retrieval evaluation..."
python scripts/eval_alignment_retrieval.py \
  --checkpoint-dir "$CHECKPOINT_DIR" \
  --split val \
  --num-samples 128

echo "Job finished at:"
date


