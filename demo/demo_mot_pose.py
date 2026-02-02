# Copyright (c) OpenMMLab. All rights reserved.
"""
MMTracking + MMPose Pipeline Demo
Combines ByteTrack object tracking with HRNet pose estimation.
"""
import os
import os.path as osp
import tempfile
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np

from mmtrack.apis import inference_mot, init_model as init_track_model
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


# COCO dataset info for pose estimation
COCO_DATASET_INFO = dict(
    dataset_name='coco',
    paper_info=dict(
        author='Lin, Tsung-Yi and Maire, Michael',
        title='Microsoft COCO: Common Objects in Context',
        year='2014'
    ),
    keypoint_info={
        0: dict(name='nose', id=0, color=[51, 153, 255], type='upper', swap=''),
        1: dict(name='left_eye', id=1, color=[51, 153, 255], type='upper', swap='right_eye'),
        2: dict(name='right_eye', id=2, color=[51, 153, 255], type='upper', swap='left_eye'),
        3: dict(name='left_ear', id=3, color=[51, 153, 255], type='upper', swap='right_ear'),
        4: dict(name='right_ear', id=4, color=[51, 153, 255], type='upper', swap='left_ear'),
        5: dict(name='left_shoulder', id=5, color=[0, 255, 0], type='upper', swap='right_shoulder'),
        6: dict(name='right_shoulder', id=6, color=[255, 128, 0], type='upper', swap='left_shoulder'),
        7: dict(name='left_elbow', id=7, color=[0, 255, 0], type='upper', swap='right_elbow'),
        8: dict(name='right_elbow', id=8, color=[255, 128, 0], type='upper', swap='left_elbow'),
        9: dict(name='left_wrist', id=9, color=[0, 255, 0], type='upper', swap='right_wrist'),
        10: dict(name='right_wrist', id=10, color=[255, 128, 0], type='upper', swap='left_wrist'),
        11: dict(name='left_hip', id=11, color=[0, 255, 0], type='lower', swap='right_hip'),
        12: dict(name='right_hip', id=12, color=[255, 128, 0], type='lower', swap='left_hip'),
        13: dict(name='left_knee', id=13, color=[0, 255, 0], type='lower', swap='right_knee'),
        14: dict(name='right_knee', id=14, color=[255, 128, 0], type='lower', swap='left_knee'),
        15: dict(name='left_ankle', id=15, color=[0, 255, 0], type='lower', swap='right_ankle'),
        16: dict(name='right_ankle', id=16, color=[255, 128, 0], type='lower', swap='left_ankle')
    },
    skeleton_info={
        0: dict(link=('left_ankle', 'left_knee'), id=0, color=[0, 255, 0]),
        1: dict(link=('left_knee', 'left_hip'), id=1, color=[0, 255, 0]),
        2: dict(link=('right_ankle', 'right_knee'), id=2, color=[255, 128, 0]),
        3: dict(link=('right_knee', 'right_hip'), id=3, color=[255, 128, 0]),
        4: dict(link=('left_hip', 'right_hip'), id=4, color=[51, 153, 255]),
        5: dict(link=('left_shoulder', 'left_hip'), id=5, color=[51, 153, 255]),
        6: dict(link=('right_shoulder', 'right_hip'), id=6, color=[51, 153, 255]),
        7: dict(link=('left_shoulder', 'right_shoulder'), id=7, color=[51, 153, 255]),
        8: dict(link=('left_shoulder', 'left_elbow'), id=8, color=[0, 255, 0]),
        9: dict(link=('right_shoulder', 'right_elbow'), id=9, color=[255, 128, 0]),
        10: dict(link=('left_elbow', 'left_wrist'), id=10, color=[0, 255, 0]),
        11: dict(link=('right_elbow', 'right_wrist'), id=11, color=[255, 128, 0]),
        12: dict(link=('left_eye', 'right_eye'), id=12, color=[51, 153, 255]),
        13: dict(link=('nose', 'left_eye'), id=13, color=[51, 153, 255]),
        14: dict(link=('nose', 'right_eye'), id=14, color=[51, 153, 255]),
        15: dict(link=('left_eye', 'left_ear'), id=15, color=[51, 153, 255]),
        16: dict(link=('right_eye', 'right_ear'), id=16, color=[51, 153, 255]),
        17: dict(link=('left_ear', 'left_shoulder'), id=17, color=[51, 153, 255]),
        18: dict(link=('right_ear', 'right_shoulder'), id=18, color=[51, 153, 255])
    },
    joint_weights=[1., 1., 1., 1., 1., 1., 1., 1.2, 1.2, 1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5],
    sigmas=[0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
            0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089]
)


def get_track_id_color(track_id):
    """Generate a unique color for each track ID."""
    np.random.seed(track_id)
    color = tuple(int(c) for c in np.random.randint(0, 255, 3))
    return color


def draw_tracking_results(img, track_bboxes, track_ids, font_scale=0.6, thickness=2):
    """Draw tracking bounding boxes with IDs on image."""
    for bbox, track_id in zip(track_bboxes, track_ids):
        x1, y1, x2, y2 = map(int, bbox[:4])
        color = get_track_id_color(track_id)
        
        # Draw bounding box
        cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
        # Draw track ID label
        label = f'ID:{track_id}'
        (label_w, label_h), baseline = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
        cv2.rectangle(img, (x1, y1 - label_h - 10), (x1 + label_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), cv2.FONT_HERSHEY_SIMPLEX, font_scale, (255, 255, 255), thickness)
    
    return img


def draw_pose_results(img, pose_results, kpt_thr=0.3, radius=4, thickness=2):
    """Draw pose keypoints and skeleton on image."""
    # COCO skeleton connections (pairs of keypoint indices)
    skeleton = [
        [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # legs
        [5, 11], [6, 12], [5, 6],  # torso
        [5, 7], [6, 8], [7, 9], [8, 10],  # arms
        [1, 2], [0, 1], [0, 2], [1, 3], [2, 4], [3, 5], [4, 6]  # face
    ]
    
    # Colors for different body parts
    pose_link_color = [
        [0, 255, 0], [0, 255, 0], [255, 128, 0], [255, 128, 0], [51, 153, 255],
        [51, 153, 255], [51, 153, 255], [51, 153, 255],
        [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0],
        [51, 153, 255], [51, 153, 255], [51, 153, 255], [51, 153, 255],
        [51, 153, 255], [51, 153, 255], [51, 153, 255]
    ]
    
    for pose_result in pose_results:
        keypoints = pose_result['keypoints']
        track_id = pose_result.get('track_id', 0)
        base_color = get_track_id_color(track_id)
        
        # Draw keypoints
        for kid, kpt in enumerate(keypoints):
            x, y, score = int(kpt[0]), int(kpt[1]), kpt[2]
            if score > kpt_thr:
                cv2.circle(img, (x, y), radius, base_color, -1)
        
        # Draw skeleton
        for sk_idx, sk in enumerate(skeleton):
            kpt1_idx, kpt2_idx = sk
            kpt1 = keypoints[kpt1_idx]
            kpt2 = keypoints[kpt2_idx]
            
            if kpt1[2] > kpt_thr and kpt2[2] > kpt_thr:
                x1, y1 = int(kpt1[0]), int(kpt1[1])
                x2, y2 = int(kpt2[0]), int(kpt2[1])
                link_color = pose_link_color[sk_idx] if sk_idx < len(pose_link_color) else base_color
                cv2.line(img, (x1, y1), (x2, y2), link_color, thickness)
    
    return img


def main():
    parser = ArgumentParser(description='MMTracking + MMPose Pipeline Demo')
    
    # Tracking arguments
    parser.add_argument('track_config', help='tracking config file')
    parser.add_argument('--track-checkpoint', help='tracking checkpoint file')
    
    # Pose arguments
    parser.add_argument('--pose-config', 
                        default='configs/pose/hrnet_w48_coco_256x192.py',
                        help='pose config file')
    parser.add_argument('--pose-checkpoint',
                        default='checkpoints/hrnet_w48_coco_256x192.pth',
                        help='pose checkpoint file')
    
    # I/O arguments
    parser.add_argument('--input', required=True, help='input video file')
    parser.add_argument('--output', help='output video file (mp4 format)')
    
    # Detection/tracking thresholds
    parser.add_argument('--score-thr', type=float, default=0.5,
                        help='bounding box score threshold for tracking')
    parser.add_argument('--kpt-thr', type=float, default=0.3,
                        help='keypoint score threshold for pose')
    
    # Device
    parser.add_argument('--device', default='cpu', help='device (cpu or cuda:0)')
    
    # Visualization
    parser.add_argument('--show', action='store_true', help='show results')
    parser.add_argument('--fps', type=int, help='output video FPS')
    parser.add_argument('--thickness', type=int, default=2, help='line thickness')
    parser.add_argument('--radius', type=int, default=4, help='keypoint radius')
    
    args = parser.parse_args()
    
    assert args.output or args.show, 'Please specify --output or --show'
    
    # Initialize models
    print('Loading tracking model...')
    track_model = init_track_model(args.track_config, args.track_checkpoint, device=args.device)
    
    print('Loading pose model...')
    pose_model = init_pose_model(args.pose_config, args.pose_checkpoint, device=args.device)
    
    # Get dataset info for pose
    dataset_info = DatasetInfo(COCO_DATASET_INFO)
    
    # Load video
    print(f'Loading video: {args.input}')
    video = mmcv.VideoReader(args.input)
    fps = args.fps if args.fps else video.fps
    fps = int(fps)
    
    # Setup output
    if args.output:
        if args.output.endswith('.mp4'):
            out_dir = tempfile.TemporaryDirectory()
            out_path = out_dir.name
            _out = args.output.rsplit(os.sep, 1)
            if len(_out) > 1:
                os.makedirs(_out[0], exist_ok=True)
        else:
            out_path = args.output
            os.makedirs(out_path, exist_ok=True)
    
    print(f'Processing {len(video)} frames at {fps} FPS...')
    prog_bar = mmcv.ProgressBar(len(video))
    
    for frame_id, frame in enumerate(video):
        # Step 1: Run tracking
        track_result = inference_mot(track_model, frame, frame_id=frame_id)
        
        # Extract bboxes and track IDs from tracking result
        # track_result format: dict with 'track_bboxes' key
        track_bboxes = track_result.get('track_bboxes', [])
        
        if len(track_bboxes) > 0 and len(track_bboxes[0]) > 0:
            # track_bboxes[0] contains [x1, y1, x2, y2, score, track_id]
            bboxes = track_bboxes[0]
            
            # Filter by score threshold
            valid_mask = bboxes[:, 4] >= args.score_thr
            bboxes = bboxes[valid_mask]
            
            if len(bboxes) > 0:
                # Extract track IDs (last column)
                track_ids = bboxes[:, 5].astype(int)
                
                # Prepare person results for pose estimation
                # Format: [{'bbox': [x1, y1, x2, y2, score]}]
                person_results = []
                for i, bbox in enumerate(bboxes):
                    person_results.append({
                        'bbox': bbox[:5],  # x1, y1, x2, y2, score
                        'track_id': int(bbox[5])
                    })
                
                # Step 2: Run pose estimation on tracked persons
                pose_results, _ = inference_top_down_pose_model(
                    pose_model,
                    frame,
                    person_results,
                    bbox_thr=None,
                    format='xyxy',
                    dataset='TopDownCocoDataset',
                    dataset_info=dataset_info,
                    return_heatmap=False
                )
                
                # Add track_id to pose results
                for i, pose_result in enumerate(pose_results):
                    if i < len(person_results):
                        pose_result['track_id'] = person_results[i]['track_id']
                
                # Step 3: Visualize results
                vis_frame = frame.copy()
                
                # Draw tracking boxes with IDs
                vis_frame = draw_tracking_results(
                    vis_frame, bboxes[:, :4], track_ids,
                    thickness=args.thickness
                )
                
                # Draw pose keypoints
                vis_frame = draw_pose_results(
                    vis_frame, pose_results,
                    kpt_thr=args.kpt_thr,
                    radius=args.radius,
                    thickness=args.thickness
                )
            else:
                vis_frame = frame.copy()
        else:
            vis_frame = frame.copy()
        
        # Add frame info
        cv2.putText(vis_frame, f'Frame: {frame_id}', (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
        # Show or save
        if args.show:
            cv2.imshow('MOT + Pose', vis_frame)
            if cv2.waitKey(int(1000. / fps)) & 0xFF == ord('q'):
                break
        
        if args.output:
            out_file = osp.join(out_path, f'{frame_id:06d}.jpg')
            mmcv.imwrite(vis_frame, out_file)
        
        prog_bar.update()
    
    # Create output video
    if args.output and args.output.endswith('.mp4'):
        print(f'\nCreating output video at {args.output} with FPS={fps}')
        mmcv.frames2video(out_path, args.output, fps=fps, fourcc='mp4v')
        out_dir.cleanup()
    
    if args.show:
        cv2.destroyAllWindows()
    
    print('\nDone!')


if __name__ == '__main__':
    main()
