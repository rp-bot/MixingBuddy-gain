#!/bin/bash

# --- Job Configuration ---
#SBATCH -J aug_no_error_dataset      # Job name
#SBATCH -N1                               # Request 1 entire node
#SBATCH --ntasks-per-node=64              # Request all 64 cores
#SBATCH --mem=0                           # Request all available memory on the node
#SBATCH -t02:00:00                        # Max walltime for CPU jobs (18 hours)
#SBATCH -o outputs/logs/aug_no_error_dataset-%j.out   # Stdout/stderr log

# --- Email Notifications ---
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=pvadhulas3@gatech.edu

set -euo pipefail

echo "Job started at: $(date)"
echo "Running on node: $(hostname)"
echo "Allocated $SLURM_NTASKS cores on this node."
echo "---------------------------------------"

# Change to submission directory
cd "$SLURM_SUBMIT_DIR"

# Load modules (adjust as needed for the environment)
module load anaconda3

# Activate environment
source activate ml-env 2>/dev/null || conda activate ml-env

echo "---------------------------------------"
echo "Starting augmentation script"

echo "Setting OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE"
export OMP_NUM_THREADS=$SLURM_CPUS_ON_NODE

srun -n 1 python augment_no_error_samples.py

EXIT_CODE=$?

echo "---------------------------------------"
if [ $EXIT_CODE -eq 0 ]; then
    echo "Augmentation completed successfully."
else
    echo "Augmentation failed with exit code $EXIT_CODE"
fi

echo "Job finished at: $(date)"
