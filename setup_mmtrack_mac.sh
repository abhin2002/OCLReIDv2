#!/bin/bash
# MMTracking Environment Setup Script for macOS (Apple Silicon)
# This script creates a conda environment for running ByteTrack with MMTracking on Mac

set -e

ENV_NAME="mmtrack"

echo "========================================"
echo "MMTracking Environment Setup for macOS"
echo "========================================"

# Check if environment already exists
if conda info --envs | grep -q "^$ENV_NAME "; then
    echo "Environment '$ENV_NAME' already exists."
    read -p "Do you want to remove and recreate it? (y/n) " -n 1 -r
    echo
    if [[ $REPLY =~ ^[Yy]$ ]]; then
        conda env remove -n $ENV_NAME -y
    else
        echo "Exiting. Activate existing environment with: conda activate $ENV_NAME"
        exit 0
    fi
fi

# Create conda environment
echo "[1/7] Creating conda environment with Python 3.8..."
conda create -n $ENV_NAME python=3.8 -y

# Activate environment
echo "[2/7] Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install PyTorch for Mac (CPU/MPS)
echo "[3/7] Installing PyTorch for macOS..."
pip install torch==1.13.1 torchvision==0.14.1

# Install MMCV (CPU version for Mac)
echo "[4/7] Installing MMCV..."
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cpu/torch1.13/index.html

# Install MMDetection and MMClassification
echo "[5/7] Installing MMDetection and MMClassification..."
pip install mmdet==2.28.0 mmcls==0.25.0

# Install MMTracking and dependencies
echo "[6/7] Installing MMTracking and dependencies..."
pip install scipy pandas seaborn motmetrics lapx attributee dotty-dict tqdm pytz requests
pip install -e .

# Download checkpoint
echo "[7/7] Downloading ByteTrack checkpoint..."
mkdir -p checkpoints
if [ ! -f "checkpoints/bytetrack_yolox_x_mot17.pth" ]; then
    curl -L -o checkpoints/bytetrack_yolox_x_mot17.pth \
        "https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth"
else
    echo "Checkpoint already exists, skipping download."
fi

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run ByteTrack on a video:"
echo "  python demo/demo_mot_vis.py \\"
echo "    configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \\"
echo "    --input <your_video.mp4> \\"
echo "    --output outputs/result.mp4 \\"
echo "    --checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \\"
echo "    --device cpu \\"
echo "    --fps 30"
echo ""
echo "Note: Use '--device cpu' on Mac (MPS support may vary)."
