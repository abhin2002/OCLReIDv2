# MMTracking Environment Setup Guide

This guide explains how to set up the MMTracking environment for running ByteTrack.

## Environment Specifications

| Component | Version |
|-----------|---------|
| Python | 3.8 |
| PyTorch | 1.10.1+cu113 |
| CUDA | 11.3 |
| MMCV | 1.5.3 |
| MMDetection | 2.24.1 |
| MMClassification | 0.25.0 |
| MMTracking | 0.14.0 |

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

### Option 3: Manual Installation

```bash
# 1. Create conda environment
conda create -n mmtrack python=3.8 -y
conda activate mmtrack

# 2. Install PyTorch with CUDA 11.3
pip install torch==1.10.1+cu113 torchvision==0.11.2+cu113 -f https://download.pytorch.org/whl/cu113/torch_stable.html

# 3. Install MMCV
pip install mmcv-full==1.5.3 -f https://download.openmmlab.com/mmcv/dist/cu113/torch1.10.0/index.html

# 4. Install MMDetection and MMClassification
pip install mmdet==2.24.1 mmcls==0.25.0

# 5. Install MMTracking and dependencies
pip install scipy==1.7.3 pandas==1.3.5 seaborn motmetrics lapx attributee dotty-dict tqdm pytz requests
pip install mmtrack
```

## Download Pretrained Model

```bash
mkdir -p checkpoints
wget https://download.openmmlab.com/mmtracking/mot/bytetrack/bytetrack_yolox_x/bytetrack_yolox_x_crowdhuman_mot17-private-half_20211218_205500-1985c9f0.pth -O checkpoints/bytetrack_yolox_x_mot17.pth
```

## Running ByteTrack

### Basic Usage

```bash
PYTHONNOUSERSITE=1 python demo/demo_mot_vis.py \
    configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
    --input <your_video.mp4> \
    --output <output.mp4> \
    --checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \
    --device cuda:0 \
    --fps 30
```

### Example with Demo Video

```bash
PYTHONNOUSERSITE=1 python demo/demo_mot_vis.py \
    configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
    --input demo/demo.mp4 \
    --output outputs/bytetrack_output.mp4 \
    --checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \
    --device cuda:0 \
    --fps 30
```

## Important Notes

### PYTHONNOUSERSITE Flag

Always use `PYTHONNOUSERSITE=1` when running scripts to avoid conflicts with user-level Python packages installed in `~/.local/lib/python3.8/site-packages/`.

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

## Exported Files

| File | Description |
|------|-------------|
| `environment.yml` | Full conda environment export |
| `requirements_frozen.txt` | Pip freeze output with exact versions |
| `setup_mmtrack.sh` | Automated setup script |
| `SETUP_GUIDE.md` | This guide |

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

## References

- [MMTracking Documentation](https://mmtracking.readthedocs.io/)
- [ByteTrack Paper](https://arxiv.org/abs/2110.06864)
- [OpenMMLab GitHub](https://github.com/open-mmlab/mmtracking)
