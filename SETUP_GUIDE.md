# MMTracking + MMPose Environment Setup Guide

This guide explains how to set up the MMTracking environment for running ByteTrack with pose estimation using MMPose.

## Environment Specifications

| Component | Version |
|-----------|----------|
| Python | 3.8 |
| PyTorch | 1.13.1 (CPU) / 1.10.1+cu113 (CUDA) |
| CUDA | 11.3 (optional) |
| MMCV | 1.7.0 |
| MMDetection | 2.28.0 |
| MMClassification | 0.25.0 |
| MMTracking | 0.14.0 |
| MMPose | 0.29.0 |

## Installation Options

### Option 1: Using the Setup Script (Recommended)

```bash
./setup_mmtrack.sh
```

### Option 2: Using Conda Environment File

```bash
conda env create -f environment.yml
conda activate mmtrack
```

### Option 3: Manual Installation (CUDA)

```bash
# 1. Create conda environment
conda create -n mmtrack python=3.8 -y
conda activate mmtrack

# 2. Install PyTorch with CUDA 11.3
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 3. Install MMCV
pip install mmcv-full==1.7.0 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# 4. Install MMDetection and MMClassification
pip install mmdet==2.28.0 mmcls==0.25.0

# 5. Install MMTracking and dependencies
pip install scipy pandas seaborn motmetrics lapx attributee dotty-dict tqdm pytz requests
pip install mmtrack==0.14.0

# 6. Install MMPose and dependencies
pip install cython xtcocotools --no-build-isolation
pip install mmpose==0.29.0
```

### Option 4: Manual Installation (macOS / CPU)

```bash
# 1. Create conda environment
conda create -n mmtrack python=3.8 -y
conda activate mmtrack

# 2. Install PyTorch (CPU)
pip install torch==1.13.1 torchvision==0.14.1

# 3. Install MMCV
pip install mmcv-full==1.7.0

# 4. Install MMDetection and MMClassification
pip install mmdet==2.28.0 mmcls==0.25.0

# 5. Install MMTracking and dependencies
pip install scipy pandas seaborn motmetrics lapx attributee dotty-dict tqdm pytz requests
pip install -e .  # Install mmtrack from source

# 6. Install MMPose and dependencies
pip install cython
pip install xtcocotools --no-build-isolation
pip install mmpose==0.29.0
```

## Download Pretrained Models

### ByteTrack Model (Object Tracking)
```bash
mkdir -p checkpoints
# Using wget
wget https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth -O checkpoints/bytetrack_yolox_x_mot17.pth

# Or using curl (macOS)
curl -L -o checkpoints/bytetrack_yolox_x_mot17.pth https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth
```

### HRNet Model (Pose Estimation)
```bash
# Using wget
wget https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth -O checkpoints/hrnet_w48_coco_256x192.pth

# Or using curl (macOS)
curl -L -o checkpoints/hrnet_w48_coco_256x192.pth https://download.openmmlab.com/mmpose/top_down/hrnet/hrnet_w48_coco_256x192-b9e0b3ab_20200708.pth
```

## Running ByteTrack (Object Tracking)

### Basic Usage

```bash
python demo/demo_mot_vis.py \
    configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
    --input <your_video.mp4> \
    --output <output.mp4> \
    --checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \
    --device cuda:0 \
    --fps 30
```

### Example with Demo Video

```bash
# GPU
python demo/demo_mot_vis.py \
    configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
    --input demo/demo.mp4 \
    --output outputs/bytetrack_output.mp4 \
    --checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \
    --device cuda:0 \
    --fps 30

# CPU (macOS)
python demo/demo_mot_vis.py \
    configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
    --input demo/demo.mp4 \
    --output outputs/bytetrack_output.mp4 \
    --checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \
    --device cpu \
    --fps 30
```

## Running MMPose (Pose Estimation)

### Basic Usage

```bash
python demo/demo_pose.py \
    --config configs/pose/hrnet_w48_coco_256x192.py \
    --checkpoint checkpoints/hrnet_w48_coco_256x192.pth \
    --img <your_image.jpg> \
    --output <output.jpg> \
    --device cpu
```

### Example with Test Image

```bash
python demo/demo_pose.py \
    --config configs/pose/hrnet_w48_coco_256x192.py \
    --checkpoint checkpoints/hrnet_w48_coco_256x192.pth \
    --img demo/test_images/test_frame.jpg \
    --output outputs/pose_result.jpg \
    --device cpu
```

## Important Notes

### Available ByteTrack Configs

| Config | Description |
|--------|-------------|
| `bytetrack_yolox_x_crowdhuman_mot17-private-half.py` | YOLOX-X trained on CrowdHuman + MOT17 (half) |
| `bytetrack_yolox_x_crowdhuman_mot17-private.py` | YOLOX-X trained on CrowdHuman + MOT17 (full) |
| `bytetrack_yolox_x_crowdhuman_mot20-private.py` | YOLOX-X trained on CrowdHuman + MOT20 |

### Command Line Options

| Option | Description |
|--------|-------------|
| `--input` | Input video file or folder containing images |
| `--output` | Output video file (mp4) or folder |
| `--checkpoint` | Path to model checkpoint |
| `--device` | Device for inference (e.g., `cuda:0`, `cpu`) |
| `--fps` | FPS for output video |
| `--score-thr` | Score threshold for filtering detections (default: 0.0) |
| `--show` | Show results on the fly |

### MMPose Command Line Options

| Option | Description |
|--------|-------------|
| `--config` | Config file for pose estimation model |
| `--checkpoint` | Path to pose model checkpoint |
| `--img` | Input image file |
| `--output` | Output image file |
| `--device` | Device for inference (e.g., `cuda:0`, `cpu`) |
| `--kpt-thr` | Keypoint score threshold (default: 0.3) |
| `--show` | Show results on the fly |

### Available Pose Estimation Configs

| Config | Description |
|--------|-------------|
| `hrnet_w48_coco_256x192.py` | HRNet-W48 trained on COCO (256x192 input) |

## Exported Files

| File | Description |
|------|-------------|
| `environment.yml` | Full conda environment export |
| `requirements_frozen.txt` | Pip freeze output with exact versions |
| `setup_mmtrack.sh` | Automated setup script |
| `SETUP_GUIDE.md` | This guide |
| `demo/demo_pose.py` | MMPose demo script |
| `configs/pose/hrnet_w48_coco_256x192.py` | HRNet pose config |

## Troubleshooting

### NumPy/SciPy Warning

You may see a warning about NumPy version compatibility with SciPy. This can be safely ignored as it doesn't affect functionality.

### Module Not Found Errors

If you encounter `ModuleNotFoundError`, ensure you're using `PYTHONNOUSERSITE=1` to isolate the conda environment from user packages.

### CUDA Out of Memory

If you run out of GPU memory, try:
- Using a smaller model config
- Reducing input resolution
- Using `--device cpu` for CPU inference (slower)

### xtcocotools Build Error (macOS)

If you encounter build errors with xtcocotools, install Cython first:
```bash
pip install cython
pip install xtcocotools --no-build-isolation
```

## References

- [MMTracking Documentation](https://mmtracking.readthedocs.io/)
- [MMPose Documentation](https://mmpose.readthedocs.io/)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [HRNet Paper](https://arxiv.org/abs/1902.09212)
- [OpenMMLab GitHub](https://github.com/open-mmlab)
