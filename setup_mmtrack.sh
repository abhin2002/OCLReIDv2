#!/bin/bash
# MMTracking + MMPose Environment Setup Script
# This script creates a conda environment for running ByteTrack with MMTracking and MMPose

set -e

ENV_NAME="mmtrack"

echo "========================================"
echo "MMTracking + MMPose Environment Setup"
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
echo "[4/7] Installing MMCV..."
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# Install MMDetection and MMClassification
echo "[5/7] Installing MMDetection and MMClassification..."
pip install mmdet==2.28.0 mmcls==0.25.0

# Install MMTracking and dependencies
echo "[6/7] Installing MMTracking and dependencies..."
pip install scipy pandas seaborn motmetrics lapx attributee dotty-dict tqdm pytz requests
pip install mmtrack==0.14.0

# Install MMPose and dependencies
echo "[7/7] Installing MMPose and dependencies..."
pip install cython
pip install xtcocotools --no-build-isolation
pip install mmpose==0.29.0

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
echo ""
echo "To run MMPose:"
echo "  python demo/demo_pose.py \\"
echo "    --config configs/pose/hrnet_w48_coco_256x192.py \\"
echo "    --checkpoint checkpoints/hrnet_w48_coco_256x192.pth \\"
echo "    --img <your_image.jpg> \\"
echo "    --output <output.jpg> \\"
echo "    --device cuda:0"
