#!/bin/bash

# Exit on error
set -e

# Environment name
ENV_NAME="nanovlm"

# Check if conda is installed
if ! command -v conda &> /dev/null; then
    echo "❌ Conda is not installed. Please install conda first."
    exit 1
fi

# Create new conda environment
echo "🔧 Creating new conda environment: $ENV_NAME"
conda create -y -n $ENV_NAME python=3.10

# Activate the environment
echo "🔧 Activating environment: $ENV_NAME"
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install required packages
echo "📦 Installing required packages..."
pip install -U pip
pip install torch torchvision torchaudio
pip install datasets
pip install Pillow
pip install transformers
pip install accelerate
pip install wandb
pip install tqdm
pip install numpy
pip install matplotlib

# Verify installation
echo "✅ Environment setup complete!"
echo "To activate the environment, run: conda activate $ENV_NAME" 