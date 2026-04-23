# Copyright (c) OpenMMLab. All rights reserved.
"""
MMTracking + MMPose + HOE Pipeline Demo
Combines ByteTrack object tracking with HRNet pose estimation and
Human Orientation Estimation (HOE).
"""
import os
import os.path as osp
import tempfile
import math
from argparse import ArgumentParser

import cv2
import mmcv
import numpy as np
import torch

from mmtrack.apis import inference_mot, init_model as init_track_model
from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo

# Import HOE estimator for orientation estimation
from mmtrack.models.orientation.hoe_estimator import init_hoe_model

# Import OCLREIDIdentifier for person re-identification
from mmtrack.models.identifier.oclreid_identifier import OCLREIDIdentifier
from mmtrack.models.identifier.params.hyper_params import HyperParams


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
        
        # # Draw bounding box
        # cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
        
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


def draw_pose_results_14kpt(img, keypoints_14_list, track_ids, kpt_thr=0.3, radius=4, thickness=2):
    """Draw 14-keypoint pose results on image.
    
    14 keypoints: Nose, LShoulder, RShoulder, LElbow, RElbow, LWrist, RWrist,
                  LHip, RHip, LKnee, RKnee, LAnkle, RAnkle, Neck
    """
    # Skeleton for 14-keypoint format (pairs of keypoint indices)
    skeleton_14 = [
        [11, 9], [9, 7], [12, 10], [10, 8], [7, 8],  # legs: LAnkle-LKnee, LKnee-LHip, RAnkle-RKnee, RKnee-RHip, LHip-RHip
        [1, 7], [2, 8], [1, 2],  # torso: LShoulder-LHip, RShoulder-RHip, LShoulder-RShoulder
        [1, 3], [2, 4], [3, 5], [4, 6],  # arms: LShoulder-LElbow, RShoulder-RElbow, LElbow-LWrist, RElbow-RWrist
        [0, 13], [13, 1], [13, 2]  # neck connections: Nose-Neck, Neck-LShoulder, Neck-RShoulder
    ]
    
    # Colors for skeleton links
    pose_link_color = [
        [0, 255, 0], [0, 255, 0], [255, 128, 0], [255, 128, 0], [51, 153, 255],
        [51, 153, 255], [51, 153, 255], [51, 153, 255],
        [0, 255, 0], [255, 128, 0], [0, 255, 0], [255, 128, 0],
        [51, 153, 255], [51, 153, 255], [51, 153, 255]
    ]
    
    for keypoints, track_id in zip(keypoints_14_list, track_ids):
        base_color = get_track_id_color(track_id)
        
        # Draw keypoints
        for kid, kpt in enumerate(keypoints):
            x, y, score = int(kpt[0]), int(kpt[1]), kpt[2]
            if score > kpt_thr:
                cv2.circle(img, (x, y), radius, base_color, -1)
        
        # Draw skeleton
        for sk_idx, sk in enumerate(skeleton_14):
            kpt1_idx, kpt2_idx = sk
            kpt1 = keypoints[kpt1_idx]
            kpt2 = keypoints[kpt2_idx]
            
            if kpt1[2] > kpt_thr and kpt2[2] > kpt_thr:
                x1, y1 = int(kpt1[0]), int(kpt1[1])
                x2, y2 = int(kpt2[0]), int(kpt2[1])
                link_color = pose_link_color[sk_idx] if sk_idx < len(pose_link_color) else base_color
                cv2.line(img, (x1, y1), (x2, y2), link_color, thickness)
    
    return img


def get_iou(pred_box, gt_box):
    """
    Calculate Intersection over Union (IOU) between two bounding boxes.
    
    Args:
        pred_box: predicted bounding box [x1, y1, x2, y2]
        gt_box: ground truth bounding box [x1, y1, x2, y2]
    
    Returns:
        float: IOU score between 0 and 1
    """
    # Get intersection coordinates
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax - ixmin + 1., 0.)
    ih = np.maximum(iymax - iymin + 1., 0.)

    # Calculate intersection area
    inters = iw * ih

    # Calculate union area
    uni = ((pred_box[2] - pred_box[0] + 1.) * (pred_box[3] - pred_box[1] + 1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)

    # Calculate IOU
    iou = inters / uni if uni > 0 else 0
    return iou


def init_target_person(bboxes, track_ids, gt_bbox, iou_threshold=0.4):
    """
    Initialize target person by matching ground truth bbox with tracked persons.
    
    Args:
        bboxes: Array of [x1, y1, x2, y2] bounding boxes
        track_ids: Array of track IDs corresponding to bboxes
        gt_bbox: Ground truth bounding box [x1, y1, x2, y2]
        iou_threshold: Minimum IOU threshold to consider a match
    
    Returns:
        tuple: (target_id, target_bbox, max_iou) or (None, None, 0) if no match
    """
    if gt_bbox is None or len(bboxes) == 0:
        return None, None, 0
    
    max_iou = iou_threshold
    target_id = None
    target_bbox = None
    
    print(f"\n[Target Init] GT bbox: {gt_bbox}")
    for i, (bbox, track_id) in enumerate(zip(bboxes, track_ids)):
        bbox_coords = bbox[:4]
        iou = get_iou(bbox_coords, gt_bbox)
        print(f"  Track {int(track_id)}: bbox={bbox_coords.astype(int)}, IOU={iou:.3f}")
        
        if iou > max_iou:
            max_iou = iou
            target_id = int(track_id)
            target_bbox = bbox_coords.astype(int)
    
    if target_id is not None:
        print(f"  -> Target selected: ID={target_id}, IOU={max_iou:.3f}")
    else:
        print(f"  -> No target matched (max IOU < {iou_threshold})")
    
    return target_id, target_bbox, max_iou


def draw_orientation_results(img, bboxes, track_ids, orientations, binary_orientations, 
                              font_scale=0.5, thickness=2, arrow_length=40):
    """
    Draw orientation indicators on the image.
    
    Args:
        img: Input image (BGR)
        bboxes: Array of [x1, y1, x2, y2, ...] bounding boxes
        track_ids: List of track IDs
        orientations: List of orientation angles in degrees (0-355)
        binary_orientations: List of binary orientation (0=Front, 1=Back)
        font_scale: Font scale for text
        thickness: Line thickness
        arrow_length: Length of orientation arrow
    
    Returns:
        Image with orientation indicators
    """
    for i, (bbox, track_id, ori, bin_ori) in enumerate(zip(bboxes, track_ids, 
                                                            orientations, binary_orientations)):
        x1, y1, x2, y2 = map(int, bbox[:4])
        color = get_track_id_color(track_id)
        
        # Calculate center of bbox
        cx = (x1 + x2) // 2
        cy = (y1 + y2) // 2
        
        # Draw orientation arrow from center
        # Convert degrees to radians (0° = facing right, 90° = facing down)
        angle_rad = math.radians(ori - 90)  # Adjust so 0° points up
        arrow_end_x = int(cx + arrow_length * math.cos(angle_rad))
        arrow_end_y = int(cy + arrow_length * math.sin(angle_rad))
        
        # Draw arrow
        cv2.arrowedLine(img, (cx, cy), (arrow_end_x, arrow_end_y), 
                        color, thickness + 1, tipLength=0.3)
        
        # Draw orientation text below or above bbox, ensure it's visible on frame
        ori_label = f'{int(round(ori))}deg ({"F" if bin_ori == 0 else "B"})'
        (label_w, label_h), baseline = cv2.getTextSize(ori_label, cv2.FONT_HERSHEY_SIMPLEX,
                                   font_scale, thickness)
        pad = 4
        h, w = img.shape[:2]

        # Try placing below bbox
        lx1 = x1
        ly1 = y2 + pad
        lx2 = lx1 + label_w + 2 * pad
        ly2 = ly1 + label_h + baseline + 2 * pad

        # If label goes outside bottom, place above bbox
        if ly2 > h:
            ly2 = y1 - pad
            ly1 = ly2 - (label_h + baseline + 2 * pad)

        # Clamp horizontally
        if lx2 > w:
            lx2 = w - 1
            lx1 = max(w - (label_w + 2 * pad), 0)
        if lx1 < 0:
            lx1 = 0

        # Clamp vertically
        if ly1 < 0:
            ly1 = 0
        if ly2 > h:
            ly2 = h - 1

        # Draw background rectangle and text (white text on track color)
        cv2.rectangle(img, (int(lx1), int(ly1)), (int(lx2), int(ly2)), color, -1)
        text_org = (int(lx1 + pad), int(ly2 - baseline - pad))
        cv2.putText(img, ori_label, text_org, cv2.FONT_HERSHEY_SIMPLEX,
                font_scale, (255, 255, 255), thickness, cv2.LINE_AA)
    
    return img



# Target 14-keypoint format indices and names
KEYPOINT_14_NAMES = [
    'Nose', 'LShoulder', 'RShoulder', 'LElbow', 'RElbow',
    'LWrist', 'RWrist', 'LHip', 'RHip', 'LKnee', 'RKnee',
    'LAnkle', 'RAnkle', 'Neck'
]


def convert_coco17_to_14keypoints(coco_keypoints):
    """
    Convert COCO 17-keypoint format to 14-keypoint format.
    
    COCO 17 keypoints:
        0: nose, 1: left_eye, 2: right_eye, 3: left_ear, 4: right_ear,
        5: left_shoulder, 6: right_shoulder, 7: left_elbow, 8: right_elbow,
        9: left_wrist, 10: right_wrist, 11: left_hip, 12: right_hip,
        13: left_knee, 14: right_knee, 15: left_ankle, 16: right_ankle
    
    Target 14 keypoints:
        0: Nose, 1: LShoulder, 2: RShoulder, 3: LElbow, 4: RElbow,
        5: LWrist, 6: RWrist, 7: LHip, 8: RHip, 9: LKnee, 10: RKnee,
        11: LAnkle, 12: RAnkle, 13: Neck (computed from shoulders)
    
    Args:
        coco_keypoints: numpy array of shape (17, 3) with [x, y, confidence]
    
    Returns:
        numpy array of shape (14, 3) with [x, y, confidence]
    """
    # Mapping from target index to COCO index
    # Neck (index 13) is computed separately
    coco_to_14_mapping = {
        0: 0,   # Nose -> nose
        1: 5,   # LShoulder -> left_shoulder
        2: 6,   # RShoulder -> right_shoulder
        3: 7,   # LElbow -> left_elbow
        4: 8,   # RElbow -> right_elbow
        5: 9,   # LWrist -> left_wrist
        6: 10,  # RWrist -> right_wrist
        7: 11,  # LHip -> left_hip
        8: 12,  # RHip -> right_hip
        9: 13,  # LKnee -> left_knee
        10: 14, # RKnee -> right_knee
        11: 15, # LAnkle -> left_ankle
        12: 16, # RAnkle -> right_ankle
    }
    
    keypoints_14 = np.zeros((14, 3), dtype=np.float32)
    
    # Map direct correspondences
    for target_idx, coco_idx in coco_to_14_mapping.items():
        keypoints_14[target_idx] = coco_keypoints[coco_idx]
    
    # Compute Neck as midpoint of left_shoulder (5) and right_shoulder (6)
    left_shoulder = coco_keypoints[5]
    right_shoulder = coco_keypoints[6]
    
    neck_x = (left_shoulder[0] + right_shoulder[0]) / 2
    neck_y = (left_shoulder[1] + right_shoulder[1]) / 2
    # Confidence is the minimum of the two shoulders
    neck_conf = min(left_shoulder[2], right_shoulder[2])
    
    keypoints_14[13] = [neck_x, neck_y, neck_conf]
    
    return keypoints_14


def convert_pose_results_to_14keypoints(pose_results):
    """
    Convert a list of pose results from COCO 17-keypoint to 14-keypoint format.
    
    Args:
        pose_results: List of dicts, each containing 'keypoints' of shape (17, 3)
    
    Returns:
        List of numpy arrays of shape (14, 3) - one per detected person
    """
    track_kpts = []
    for result in pose_results:
        coco_kpts = result['keypoints']
        kpts_14 = convert_coco17_to_14keypoints(coco_kpts)
        track_kpts.append(kpts_14)
    return track_kpts


def prepare_image_metadata(frame, scale_factor=1.0):
    """
    Prepare image metadata for OCLREIDIdentifier.
    
    Args:
        frame: Input image/frame (H, W, 3)
        scale_factor: Scale factor for the image
    
    Returns:
        List containing metadata dict
    """
    h, w, c = frame.shape
    img_metas = [{
        'img_shape': (h, w, c),
        'ori_shape': (h, w, c),
        'pad_shape': (h, w, c),
        'scale_factor': scale_factor,
        'flip': False,
        'filename': None
    }]
    return img_metas


def prepare_tracks_for_reid(bboxes, track_ids, keypoints_14_list):
    """
    Prepare tracks dict in the format expected by OCLREIDIdentifier.
    
    Args:
        bboxes: Array of [x1, y1, x2, y2, score] from tracking
        track_ids: Array of track IDs
        keypoints_14_list: List of (14, 3) keypoint arrays
    
    Returns:
        Dict with format {id: [x1,y1,x2,y2,score,kpts,ori], ...}
    """
    tracks = {}
    for i, (bbox, track_id) in enumerate(zip(bboxes, track_ids)):
        track_id_int = int(track_id)
        x1, y1, x2, y2, score = bbox[:5]
        
        # Get keypoints for this tracklet
        kpts = keypoints_14_list[i] if i < len(keypoints_14_list) else np.zeros((14, 3))
        
        # For orientation, compute a simple estimate from keypoints if available
        # Left/right shoulder positions indicate orientation
        # For now, use 0 as default orientation
        ori = 0
        
        tracks[track_id_int] = [x1, y1, x2, y2, score, kpts, ori]
    
    return tracks


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
    
    # HOE (Orientation Estimation) arguments
    parser.add_argument('--hoe-checkpoint',
                        default='mmtrack/models/orientation/checkpoints/keypoints_net.pth',
                        help='HOE model checkpoint file')
    parser.add_argument('--enable-hoe', action='store_true', default=True,
                        help='Enable orientation estimation')
    parser.add_argument('--no-hoe', action='store_false', dest='enable_hoe',
                        help='Disable orientation estimation')
    
    # I/O arguments
    parser.add_argument('--input', required=True, help='input video file')
    parser.add_argument('--output', help='output video file (mp4 format)')
    
    # Detection/tracking thresholds
    parser.add_argument('--score-thr', type=float, default=0.0,
                        help='bounding box score threshold for tracking')
    parser.add_argument('--kpt-thr', type=float, default=0.3,
                        help='keypoint score threshold for pose')
    
    # Target person initialization
    parser.add_argument('--gt-bbox', type=int, nargs=4, metavar=('X1', 'Y1', 'X2', 'Y2'),
                        help='ground truth bbox for target person [x1 y1 x2 y2] (only frame 0)')
    parser.add_argument('--iou-threshold', type=float, default=0.4,
                        help='IOU threshold for target person matching')
    
    # Device
    parser.add_argument('--device', default='cpu', help='device (cpu or cuda:0)')
    
    # Visualization
    parser.add_argument('--show', action='store_true', help='show results', default=True)
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
    
    # Load HOE model for orientation estimation
    hoe_model = None
    if args.enable_hoe:
        print('Loading HOE (orientation estimation) model...')
        hoe_model = init_hoe_model(checkpoint_path=args.hoe_checkpoint, device=args.device)
    
    # Initialize OCLREIDIdentifier for person re-identification
    print('Initializing OCLREIDIdentifier for person re-identification...')
    
    # Create ReID parameters
    reid_params = {
        'height': 256,  # Image patch height for ReID
        'width': 192,   # Image patch width for ReID
        'norm_mean': [0.485, 0.456, 0.406],  # ImageNet normalization mean
        'norm_std': [0.229, 0.224, 0.225],   # ImageNet normalization std
        'agent': 'PartOCLWeightedClassifier',  # Classifier type
        'reid_pos_confidence_thresh': 0.6,
        'reid_neg_confidence_thresh': 0.30,
        'reid_positive_count': 5,
        'initial_training_num_samples': 5,
        'min_target_confidence': -1,
        'id_switch_detection_thresh': 0.35,
    }
    
    # Create HyperParams object
    hyper_params = HyperParams(reid_params)
    # Add all params to the hyper_params object
    for key, value in reid_params.items():
        setattr(hyper_params, key, value)
    
    # Initialize the ReID identifier
    reid_identifier = OCLREIDIdentifier(hyper_params)
    
    # Create a minimal RPF model wrapper to provide necessary attributes to the identifier
    class RPFModelWrapper:
        def __init__(self, reid_model):
            self.reid = reid_model
            self.visdom = None
            self.debug = False
            self.save = False
    
    # Try to get reid model from tracking model, otherwise use a placeholder
    reid_model = None
    try:
        # Try to extract reid model from tracking model if available
        if hasattr(track_model, 'module'):
            if hasattr(track_model.module, 'reid'):
                reid_model = track_model.module.reid
        elif hasattr(track_model, 'reid'):
            reid_model = track_model.reid
    except:
        pass
    
    # If no reid model found, use pose_model as a feature extractor
    if reid_model is None:
        reid_model = pose_model
    
    rpf_model = RPFModelWrapper(reid_model)
    
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
    
    # Parse ground truth bbox if provided
    gt_bbox = None
    if args.gt_bbox:
        gt_bbox = np.array(args.gt_bbox, dtype=np.float32)
    
    # Target person tracking state
    target_id = None
    target_bbox = None
    
    for frame_id, frame in enumerate(video):
        # Step 1: Run tracking
        track_result = inference_mot(track_model, frame, frame_id=frame_id)
        
        # Extract bboxes and track IDs from tracking result
        # track_result format: dict with 'track_bboxes' key
        track_bboxes = track_result.get('track_bboxes', [])
        
        if len(track_bboxes) > 0 and len(track_bboxes[0]) > 0:
            # track_bboxes[0] contains [track_id, x1, y1, x2, y2, score]
            bboxes = track_bboxes[0]
            
            # Filter by score threshold (score is at index 5)
            valid_mask = bboxes[:, 5] >= args.score_thr
            bboxes = bboxes[valid_mask]
            
            if len(bboxes) > 0:
                # Extract track IDs (first column)
                track_ids = bboxes[:, 0].astype(int)
                
                # Prepare person results for pose estimation
                # Format: [{'bbox': [x1, y1, x2, y2, score]}]
                person_results = []
                for i, bbox in enumerate(bboxes):
                    person_results.append({
                        'bbox': np.array([bbox[1], bbox[2], bbox[3], bbox[4], bbox[5]]),  # x1, y1, x2, y2, score
                        'track_id': int(bbox[0])
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
                
                # Convert to 14-keypoint format
                keypoints_14_list = convert_pose_results_to_14keypoints(pose_results)
                # keypoints_14_list is a list of (14, 3) arrays, one per tracked person
                # Can be used for downstream tasks like ReID, action recognition, etc.
                
                # Step 3: Orientation estimation (HOE)
                orientations = []
                binary_orientations = []
                if hoe_model is not None and len(keypoints_14_list) > 0:
                    # Get bboxes for normalization (x1, y1, x2, y2 format)
                    bbox_list = [bbox[1:5] for bbox in bboxes]  # Skip track_id
                    
                    # Estimate orientation
                    orientations, binary_orientations, confidences = hoe_model.estimate_orientation(
                        keypoints_14_list, bboxes=bbox_list
                    )
                    
                    # Print orientation info for debugging
                    # for tid, ori, bin_ori in zip(track_ids, orientations, binary_orientations):
                    #     print(f'  Track {tid}: {ori}° ({"Front" if bin_ori == 0 else "Back"})')
                
                # Step 3.5: Target person initialization (Frame 0 only)
                if frame_id == 0 and target_id is None and gt_bbox is not None:
                    target_id, target_bbox, max_iou = init_target_person(
                        bboxes[:, 1:5],  # bbox coordinates [x1, y1, x2, y2]
                        track_ids,
                        gt_bbox,
                        iou_threshold=args.iou_threshold
                    )
                
                # ============================================================================
                # STEP 4: TARGET IDENTIFICATION (OCL-ReID)
                # ============================================================================
                ident_result = None
                if target_id is not None:
                    try:
                        # Prepare image metadata
                        img_metas = prepare_image_metadata(frame)
                        
                        # Prepare tracks in the format expected by OCLREIDIdentifier
                        tracks = prepare_tracks_for_reid(bboxes[:, 1:5], track_ids, keypoints_14_list)
                        
                        # Initialize the identifier with current target_id if not yet initialized
                        if reid_identifier.target_id == -1:
                            reid_identifier.init_identifier(target_id, rpf_model)
                        
                        # Convert frame to tensor format if needed
                        frame_tensor = torch.from_numpy(frame).permute(2, 0, 1).unsqueeze(0).float()
                        if args.device != 'cpu':
                            frame_tensor = frame_tensor.to(args.device)
                        
                        # Run OCLREIDIdentifier
                        ident_result = reid_identifier.identify(
                            img=frame_tensor,
                            img_metas=img_metas,
                            model=rpf_model,
                            tracks=tracks,
                            frame_id=frame_id,
                            rescale=False,
                            gt_bbox=gt_bbox if frame_id == 0 else None
                        )
                        
                        if ident_result is not None:
                            print(f"  [ReID Frame {frame_id}] State: {ident_result.get('state', 'N/A')}, "
                                  f"Target ID: {ident_result.get('target_id', -1)}, "
                                  f"Target Conf: {ident_result.get('target_conf', -1):.3f}")
                    except Exception as e:
                        print(f"  [ReID Warning] Error during identification: {str(e)}")
                        ident_result = None
                
                # ============================================================================
                # STEP 5: Visualize results
                vis_frame = frame.copy()
                
                # Draw tracking boxes with IDs (bbox coords are at indices 1-4)
                vis_frame = draw_tracking_results(
                    vis_frame, bboxes[:, 1:5], track_ids,
                    thickness=args.thickness
                )
                
                # Draw 14-keypoint pose results
                vis_frame = draw_pose_results_14kpt(
                    vis_frame, keypoints_14_list, track_ids,
                    kpt_thr=args.kpt_thr,
                    radius=args.radius,
                    thickness=args.thickness
                )
                
                # Draw orientation indicators
                if len(orientations) > 0:
                    vis_frame = draw_orientation_results(
                        vis_frame, bboxes[:, 1:5], track_ids,
                        orientations, binary_orientations,
                        thickness=args.thickness
                    )
                
                # Draw target person bbox if initialized
                if target_bbox is not None:
                    x1, y1, x2, y2 = map(int, target_bbox)
                    cv2.rectangle(vis_frame, (x1, y1), (x2, y2), (0, 255, 0), 3)  # Green box
                    
                    # Prepare target label with confidence from ReID if available
                    target_label = f'TARGET (ID:{target_id})'
                    target_conf_text = ''
                    if ident_result is not None:
                        target_conf = ident_result.get('target_conf', -1)
                        state = ident_result.get('state', 'N/A')
                        if target_conf >= 0:
                            target_conf_text = f' Conf:{target_conf:.3f} [{state}]'
                    
                    cv2.putText(vis_frame, target_label + target_conf_text, (x1, y1 - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
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
