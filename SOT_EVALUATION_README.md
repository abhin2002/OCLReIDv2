# Single Object Tracking (SOT) Evaluation

This directory contains scripts for evaluating single object tracking performance on DanceTrack dataset using ByteTrack as the tracker.

## Overview

The evaluation converts multi-object tracking (MOT) into single object tracking (SOT) by:
1. Selecting a specific target person by ID from ground truth
2. Tracking only that target across all frames
3. Computing standard SOT metrics (Success Rate, Precision)

## Scripts

### 1. `demo/sot_evaluation.py`
Evaluates tracking for a single target in a single sequence.

**Features:**
- Loads ground truth from MOT format (`gt/gt.txt`)
- Initializes target using GT bbox in first frame
- Tracks target across all frames
- Computes SOT metrics: Success Rate (AUC), Precision @ 20px, Mean IoU
- Outputs results to JSON file
- Optional video visualization

### 2. `demo/sot_batch_evaluation.py`
Batch evaluation across multiple sequences and targets.

**Features:**
- Evaluates all sequences in a test directory
- Can evaluate multiple targets per sequence
- Aggregates results with statistics (mean, std, min, max)
- Generates comprehensive JSON report

## Installation

Ensure you have the required dependencies:
```bash
pip install mmcv-full mmdet mmtrack mmpose opencv-python numpy
```

## Usage

### Single Sequence Evaluation

```bash
python demo/sot_evaluation.py \
  configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
  --track-checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \
  --sequence-dir test1/dancetrack0003 \
  --target-id 1 \
  --output-json results/dancetrack0003_target1.json \
  --device cpu
```

**Arguments:**
- `track_config`: Path to tracking model config file
- `--track-checkpoint`: Path to tracking model checkpoint
- `--sequence-dir`: Path to sequence directory (contains `img1/` and `gt/gt.txt`)
- `--target-id`: Target person ID from ground truth annotations
- `--output-json`: Output JSON file for results (default: `sot_results.json`)
- `--output-video`: Optional output video with visualization
- `--show`: Display tracking results in real-time
- `--device`: Device to run on (`cpu` or `cuda:0`)
- `--score-thr`: Detection score threshold (default: 0.0)
- `--iou-threshold`: IoU threshold for target initialization (default: 0.3)
- `--fps`: Video FPS for visualization (default: 20)

### Batch Evaluation

Evaluate all sequences in test directory:

```bash
python demo/sot_batch_evaluation.py \
  configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
  --track-checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \
  --test-dir test1/ \
  --output-dir sot_results/ \
  --device cpu
```

Evaluate specific sequences and targets:

```bash
python demo/sot_batch_evaluation.py \
  configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
  --track-checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \
  --test-dir test1/ \
  --sequences dancetrack0003 dancetrack0009 \
  --target-ids 1 2 3 \
  --max-targets-per-seq 3 \
  --output-dir sot_results/ \
  --device cuda:0
```

**Arguments:**
- `track_config`: Path to tracking model config file
- `--track-checkpoint`: Path to tracking model checkpoint
- `--test-dir`: Test directory containing sequence folders
- `--sequences`: Specific sequences to evaluate (default: all)
- `--target-ids`: Specific target IDs to evaluate (default: all in GT)
- `--max-targets-per-seq`: Maximum targets per sequence (default: all)
- `--output-dir`: Output directory for results (default: `sot_results/`)
- `--device`: Device to run on (`cpu` or `cuda:0`)

## Dataset Structure

Expected directory structure:
```
test1/
├── dancetrack0003/
│   ├── img1/
│   │   ├── 00000001.jpg
│   │   ├── 00000002.jpg
│   │   └── ...
│   ├── gt/
│   │   └── gt.txt
│   └── seqinfo.ini
├── dancetrack0009/
│   └── ...
└── ...
```

Ground truth format (`gt/gt.txt`):
```
frame_id,person_id,x,y,w,h,conf,class,visibility
1,1,100,200,50,150,1,-1,-1
1,2,300,250,45,140,1,-1,-1
2,1,102,201,50,150,1,-1,-1
...
```

## Output Format

### Single Sequence Results (`sot_results.json`)

```json
{
  "sequence": "dancetrack0003",
  "target_id": 1,
  "metrics": {
    "auc": 0.6543,
    "precision_20px": 0.8234,
    "mean_iou": 0.7123,
    "mean_center_error": 12.45,
    "failures": 5,
    "total_frames": 1203,
    "tracked_frames": 1198,
    "success_rates": [1.0, 0.98, 0.95, ...],
    "precisions": [0.45, 0.52, 0.61, ...],
    "per_frame_results": [
      {
        "frame": 1,
        "iou": 0.95,
        "center_error": 2.3,
        "tracked": true
      },
      ...
    ]
  }
}
```

### Aggregate Results (`aggregate_results.json`)

```json
{
  "num_sequences": 20,
  "total_frames": 24060,
  "total_failures": 123,
  "auc": {
    "mean": 0.6543,
    "std": 0.0823,
    "min": 0.4521,
    "max": 0.8234
  },
  "precision_20px": {
    "mean": 0.7821,
    "std": 0.0654,
    "min": 0.6123,
    "max": 0.9012
  },
  "mean_iou": {
    "mean": 0.7012,
    "std": 0.0712,
    "min": 0.5234,
    "max": 0.8456
  },
  "per_sequence_results": [...]
}
```

## Evaluation Metrics

### Success Rate (AUC)
- Measures overlap between predicted and ground truth bounding boxes
- Computed as percentage of frames with IoU > threshold
- Thresholds range from 0 to 1.0 in steps of 0.05
- **AUC (Area Under Curve)**: Average success rate across all thresholds
- Higher is better (range: 0-1)

### Precision
- Measures center location error in pixels
- Computed as percentage of frames with center error < threshold
- Thresholds range from 0 to 50 pixels
- **Precision @ 20px**: Success rate at 20-pixel threshold
- Higher is better (range: 0-1)

### Additional Metrics
- **Mean IoU**: Average IoU across all frames
- **Mean Center Error**: Average distance between predicted and GT centers (pixels)
- **Failures**: Number of frames where target is lost (IoU = 0)
- **Tracked Frames**: Number of frames where target is successfully tracked

## Visualization

To generate output video with visualization:

```bash
python demo/sot_evaluation.py \
  configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
  --track-checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \
  --sequence-dir test1/dancetrack0003 \
  --target-id 1 \
  --output-json results/dancetrack0003_target1.json \
  --output-video results/dancetrack0003_target1.mp4 \
  --device cpu
```

Visualization shows:
- **Green box**: Ground truth bounding box
- **Blue box**: Predicted bounding box with IoU score
- **"LOST" text**: Tracking failure (target not detected)
- **Frame number**: Current frame ID

## Example Workflow

1. **Prepare dataset:**
   ```bash
   # Ensure test1/ directory has sequences with img1/ and gt/gt.txt
   ls test1/
   ```

2. **Evaluate single target:**
   ```bash
   python demo/sot_evaluation.py \
     configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
     --track-checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \
     --sequence-dir test1/dancetrack0003 \
     --target-id 1 \
     --output-json results/seq03_t1.json \
     --device cpu
   ```

3. **Batch evaluate all sequences:**
   ```bash
   python demo/sot_batch_evaluation.py \
     configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
     --track-checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \
     --test-dir test1/ \
     --output-dir sot_results/ \
     --max-targets-per-seq 5 \
     --device cuda:0
   ```

4. **Analyze results:**
   ```bash
   # View aggregate statistics
   cat sot_results/aggregate_results.json
   
   # View per-sequence results
   cat sot_results/dancetrack0003_target1_results.json
   ```

## Notes

- The tracker uses ByteTrack for multi-object tracking, then filters to track only the target
- Target initialization uses IoU matching between GT bbox and detected persons in first frame
- If target is not detected in a frame, it's marked as a tracking failure
- The evaluation follows standard SOT benchmarks (OTB/VOT style metrics)
- For best results, use GPU (`--device cuda:0`) for faster processing

## Troubleshooting

**Issue: "No ground truth found for target ID X"**
- Check that `gt/gt.txt` exists in sequence directory
- Verify target ID exists in ground truth file
- Use `--target-ids` to see available IDs

**Issue: "Target not initialized"**
- Lower `--iou-threshold` (default: 0.3) to allow looser matching
- Check that target appears in first frame of sequence
- Verify detection score threshold (`--score-thr`) is not too high

**Issue: High failure rate**
- Target may be occluded or leave frame
- Try different tracking model or parameters
- Check if GT annotations are complete

## Citation

If you use this evaluation code, please cite:
```bibtex
@inproceedings{bytetrack2021,
  title={ByteTrack: Multi-Object Tracking by Associating Every Detection Box},
  author={Zhang, Yifu and Sun, Peize and Jiang, Yi and Yu, Dongdong and Weng, Fucheng and Yuan, Zehuan and Luo, Ping and Liu, Wenyu and Wang, Xinggang},
  booktitle={ECCV},
  year={2022}
}