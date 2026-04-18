#!/bin/bash

# ==============================================================================
# Script: setup_env.sh
# Purpose: Create the Conda environment for the BioBridge PyTorch Lightning pipeline
# ==============================================================================

# 1. Load the cluster's base Anaconda or Miniconda module (Optional but common on HPC)
# module load anaconda3/2023.09  <-- Uncomment and adjust to your cluster's specific module name if needed

# 2. Define environment variables
ENV_NAME="kg_mlops"
PYTHON_VERSION="3.10"
CUDA_VERSION="11.8" # Highly recommended to match standard cluster GPU drivers

echo "Creating Conda environment: $ENV_NAME"

# 3. Create the conda environment
conda create -n $ENV_NAME python=$PYTHON_VERSION -y

# 4. Activate the environment
eval "$(conda shell.bash hook)" # Ensures conda activate works in scripts
conda activate $ENV_NAME

# 5. Install PyTorch with specific CUDA bindings
echo "Installing PyTorch..."
conda install pytorch torchvision torchaudio pytorch-cuda=$CUDA_VERSION -c pytorch -c nvidia -y

# 6. Install PyTorch Geometric and its dependencies
echo "Installing PyTorch Geometric..."
conda install pyg -c pyg -y

# 7. Install PyTorch Lightning and other crucial mlops tools
echo "Installing ML framework and MLOps tools..."
pip install pytorch-lightning torchmetrics
pip install wandb pandas numpy scikit-learn jupyter

echo "========================================================"
echo "Environment setup complete!"
echo "To activate: conda activate $ENV_NAME"
echo "To login to Weights & Biases: wandb login"
echo "========================================================"
