# Interactive Target Selection Feature

## Overview

The `demo_mot_pose.py` script now supports **interactive target selection** by drawing a bounding box directly on the video's first frame, eliminating the need to manually specify bbox coordinates via command line.

## Features

### Interactive Selection Mode
- **Automatic activation**: When you run the script without the `--gt-bbox` argument
- **Visual interface**: Draw a bounding box using your mouse on the first frame
- **Real-time feedback**: See the box as you draw it
- **Easy controls**: Simple keyboard shortcuts for confirmation, reset, or cancellation

### How to Use

#### Method 1: Interactive Selection (Recommended)

Run the script **without** the `--gt-bbox` argument:

```bash
python demo/demo_mot_pose.py \
  configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
  --track-checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \
  --pose-config configs/pose/hrnet_w48_coco_256x192.py \
  --pose-checkpoint checkpoints/hrnet_w48_coco_256x192.pth \
  --input your_video.mp4 \
  --output outputs/result.mp4 \
  --device cpu
```

**Interactive Selection Steps:**
1. The script loads and displays the first frame
2. **Click and drag** with your mouse to draw a bounding box around the target person
3. **Press ENTER** to confirm your selection
4. **Press 'r'** to reset and draw again if needed
5. **Press ESC** to cancel and exit

#### Method 2: Manual Bbox (Legacy)

You can still specify the bbox manually using the `--gt-bbox` argument:

```bash
python demo/demo_mot_pose.py \
  configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
  --track-checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \
  --pose-config configs/pose/hrnet_w48_coco_256x192.py \
  --pose-checkpoint checkpoints/hrnet_w48_coco_256x192.pth \
  --input your_video.mp4 \
  --output outputs/result.mp4 \
  --device cpu \
  --gt-bbox 206 93 305 407
```

## Interactive Selection Controls

| Key/Action | Function |
|------------|----------|
| **Click + Drag** | Draw bounding box |
| **ENTER** | Confirm selection and start processing |
| **r** | Reset - clear current box and draw again |
| **ESC** | Cancel - exit the script |

## Implementation Details

### Function: `select_bbox_interactive()`

Located in `demo/demo_mot_pose.py`, this function provides the interactive selection interface:

```python
def select_bbox_interactive(frame, window_name='Select Target Person'):
    """
    Interactive bounding box selection using mouse drawing.
    
    Args:
        frame: Input frame (BGR image)
        window_name: Name of the window for display
    
    Returns:
        bbox: [x1, y1, x2, y2] or None if cancelled
    """
```

**Features:**
- Mouse callback for drawing
- Real-time visual feedback
- Automatic bbox normalization (ensures x1 < x2, y1 < y2)
- Minimum size validation (prevents too-small boxes)
- Clear on-screen instructions

### Integration

The interactive selection is seamlessly integrated into the main processing pipeline:

1. **Check for manual bbox**: If `--gt-bbox` is provided, use it
2. **Otherwise**: Launch interactive selection on first frame
3. **Validate selection**: Ensure valid bbox before proceeding
4. **Continue processing**: Use selected bbox for target initialization

## Benefits

✅ **No coordinate guessing**: Visually select the target person  
✅ **Faster workflow**: No need to open video in another tool to find coordinates  
✅ **More accurate**: Direct visual selection reduces errors  
✅ **User-friendly**: Intuitive mouse-based interface  
✅ **Backward compatible**: Old `--gt-bbox` method still works  

## Troubleshooting

### Issue: Window doesn't appear
- **Solution**: Ensure you have a display available (not running in headless mode)
- For remote servers, use X11 forwarding or VNC

### Issue: Bbox too small error
- **Solution**: Draw a larger bounding box (minimum 10x10 pixels)

### Issue: Want to use headless mode
- **Solution**: Use the `--gt-bbox` argument to specify coordinates manually

## Examples

### Example 1: Track a person in a dance video
```bash
python demo/demo_mot_pose.py \
  configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
  --track-checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \
  --pose-config configs/pose/hrnet_w48_coco_256x192.py \
  --pose-checkpoint checkpoints/hrnet_w48_coco_256x192.pth \
  --input dance_video.mp4 \
  --output outputs/dance_tracked.mp4 \
  --device cuda:0
```

### Example 2: Process with custom IOU threshold
```bash
python demo/demo_mot_pose.py \
  configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py \
  --track-checkpoint checkpoints/bytetrack_yolox_x_mot17.pth \
  --pose-config configs/pose/hrnet_w48_coco_256x192.py \
  --pose-checkpoint checkpoints/hrnet_w48_coco_256x192.pth \
  --input sports_video.mp4 \
  --output outputs/sports_tracked.mp4 \
  --device cpu \
  --iou-threshold 0.3
```

## Technical Notes

- The interactive selection uses OpenCV's `cv2.setMouseCallback()` for mouse event handling
- The bbox is stored in `[x1, y1, x2, y2]` format (top-left and bottom-right corners)
- The selection window is destroyed after confirmation to free resources
- The selected bbox is used for target person initialization via IOU matching in frame 0

## See Also

- `demo/demo_mot_pose.py` - Main script with interactive selection
- `run-command-interactive.txt` - Example commands
- Original documentation for other features and parameters