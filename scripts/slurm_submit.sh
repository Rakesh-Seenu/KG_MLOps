#!/bin/bash
#SBATCH --job-name=biobridge_gnn       # Job name for tracking
#SBATCH --partition=gpu                # Partition/Queue name (e.g., gpu, compute)
#SBATCH --nodes=1                      # Number of nodes requested
#SBATCH --ntasks-per-node=1            # Number of tasks (typically 1 per node for DDP)
#SBATCH --gres=gpu:a100:4              # Requesting 4 A100 GPUs per node
#SBATCH --cpus-per-task=8              # Number of CPU cores per task (matches num_workers)
#SBATCH --mem=128G                     # System RAM requested 
#SBATCH --time=24:00:00                # Max walltime (HH:MM:SS)
#SBATCH --output=logs/slurm-%j.out     # Standard output/error log file (%j = Job ID)
#SBATCH --mail-type=END,FAIL           # Email when job finishes or fails
#SBATCH --mail-user=your_email@university.edu

# ==============================================================================
# Step 4: Large Scale Training on GPU Cluster (SLURM script)
# ==============================================================================

echo "=========================================================="
echo "Starting BioBridge Job on Node: $SLURMD_NODENAME"
echo "Date: $(date)"
echo "GPUs allocated:"
nvidia-smi
echo "=========================================================="

# 1. Load the conda environment
# Notice we assume the environment was created via setup_env.sh
source ~/.bashrc
conda activate kg_mlops

# 2. Export environment variables required by PyTorch Lightning for Multi-Node
export MASTER_PORT=29500
export MASTER_ADDR=$(scontrol show hostnames ${SLURM_JOB_NODELIST} | head -n 1)

# Debugging flags for NCCL (the NVIDIA backend for multi-gpu communication)
# export NCCL_DEBUG=INFO 
# export PYTHONFAULTHANDLER=1

# 3. Create logs directory if it doesn't exist
mkdir -p logs

# 4. Run the training script via srun
# srun acts as the launcher for DDP processes across the cluster
srun python scripts/train_hpc.py

echo "Job finished at $(date)"
