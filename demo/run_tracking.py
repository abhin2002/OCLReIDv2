# Copyright (c) OpenMMLab. All rights reserved.
"""
Run ByteTrack MOT Demo

This script runs Multi-Object Tracking using ByteTrack.
Modify the configuration parameters below as needed.
"""

import os
import os.path as osp
import tempfile

import mmcv

from mmtrack.apis import inference_mot, init_model

# ============== CONFIGURATION ==============
# Modify these parameters as needed

CONFIG_FILE = '../configs/mot/bytetrack/bytetrack_yolox_x_crowdhuman_mot17-private-half.py'
INPUT_VIDEO = '../demo_video.mp4'
OUTPUT_VIDEO = '../outputs/bytetrack_demo_video.mp4'
CHECKPOINT = '../checkpoints/bytetrack_yolox_x_mot17.pth'
DEVICE = 'cuda:0'
FPS = 30
SCORE_THR = 0.0  # Threshold of score to filter bboxes
SHOW = False  # Whether to show results on the fly
BACKEND = 'cv2'  # 'cv2' or 'plt'

# ===========================================


def run_mot():
    """Run Multi-Object Tracking on input video."""
    
    # Load images/video
    if osp.isdir(INPUT_VIDEO):
        imgs = sorted(
            filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')),
                   os.listdir(INPUT_VIDEO)),
            key=lambda x: int(x.split('.')[0]))
        IN_VIDEO = False
    else:
        imgs = mmcv.VideoReader(INPUT_VIDEO)
        IN_VIDEO = True

    # Define output
    if OUTPUT_VIDEO is not None:
        if OUTPUT_VIDEO.endswith('.mp4'):
            OUT_VIDEO = True
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = OUTPUT_VIDEO.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            OUT_VIDEO = False
            out_path = OUTPUT_VIDEO
            os.makedirs(out_path, exist_ok=True)
    else:
        OUT_VIDEO = False
        out_path = None

    fps = FPS
    if SHOW or OUT_VIDEO:
        if fps is None and IN_VIDEO:
            fps = imgs.fps
        if not fps:
            raise ValueError('Please set the FPS for the output video.')
        fps = int(fps)

    # Build the model from config file and checkpoint file
    print(f'Loading model from {CONFIG_FILE}...')
    print(f'Using checkpoint: {CHECKPOINT}')
    print(f'Device: {DEVICE}')
    model = init_model(CONFIG_FILE, CHECKPOINT, device=DEVICE)

    print(f'Processing input: {INPUT_VIDEO}')
    print(f'Output will be saved to: {OUTPUT_VIDEO}')
    
    prog_bar = mmcv.ProgressBar(len(imgs))
    
    # Test and show/save the images
    for i, img in enumerate(imgs):
        if isinstance(img, str):
            img = osp.join(INPUT_VIDEO, img)
        
        result = inference_mot(model, img, frame_id=i)
        
        if OUTPUT_VIDEO is not None:
            if IN_VIDEO or OUT_VIDEO:
                out_file = osp.join(out_path, f'{i:06d}.jpg')
            else:
                out_file = osp.join(out_path, img.rsplit(os.sep, 1)[-1])
        else:
            out_file = None
        
        model.show_result(
            img,
            result,
            score_thr=SCORE_THR,
            show=SHOW,
            wait_time=int(1000. / fps) if fps else 0,
            out_file=out_file,
            backend=BACKEND)
        
        prog_bar.update()

    # Create output video
    if OUTPUT_VIDEO and OUT_VIDEO:
        print(f'\nMaking the output video at {OUTPUT_VIDEO} with a FPS of {fps}')
        mmcv.frames2video(out_path, OUTPUT_VIDEO, fps=fps, fourcc='mp4v')
        out_dir.cleanup()

    print('\nDone!')


if __name__ == '__main__':
    run_mot()
