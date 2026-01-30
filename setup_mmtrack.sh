#!/bin/bash
# MMTracking Environment Setup Script
# This script creates a conda environment for running ByteTrack with MMTracking

set -e

ENV_NAME="mmtrack"

echo "========================================"
echo "MMTracking Environment Setup"
echo "========================================"

# Create conda environment
echo "[1/6] Creating conda environment..."
conda create -n $ENV_NAME python=3.8 -y

# Activate environment
echo "[2/6] Activating environment..."
source $(conda info --base)/etc/profile.d/conda.sh
conda activate $ENV_NAME

# Install PyTorch with CUDA 11.3
echo "[3/6] Installing PyTorch..."
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# Install MMCV
echo "[4/6] Installing MMCV..."
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# Install MMDetection and MMClassification
echo "[5/6] Installing MMDetection and MMClassification..."
pip install mmdet==2.24.1 mmcls==0.25.0

# Install MMTracking and dependencies
echo "[6/6] Installing MMTracking and dependencies..."
pip install scipy==1.7.3 pandas==1.3.5 seaborn motmetrics lapx attributee dotty-dict tqdm pytz requests
pip install mmtrack

echo ""
echo "========================================"
echo "Setup Complete!"
echo "========================================"
echo ""
echo "To activate the environment:"
echo "  conda activate $ENV_NAME"
echo ""
echo "To run ByteTrack:"
echo "  PYTHONNOUSERSITE=1 python demo/demo_mot_vis.py \\"
echo "    configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \\"
echo "    --input <your_video.mp4> \\"
echo "    --output <output.mp4> \\"
echo "    --checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \\"
echo "    --device cuda:0 \\"
echo "    --fps 30"
echo ""
echo "Note: Use PYTHONNOUSERSITE=1 to avoid conflicts with user-level packages."
