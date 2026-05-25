#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
Single Object Tracking (SOT) Evaluation Script
Tracks a single target person and evaluates using standard SOT metrics.
"""
import os
import os.path as osp
import json
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


def load_mot_gt(gt_file, target_id):
    """
    Load ground truth bounding boxes for a specific target from MOT format file.
    
    MOT format: frame,id,x,y,w,h,conf,class,visibility
    
    Args:
        gt_file: Path to gt.txt file
        target_id: Target person ID to extract
    
    Returns:
        dict: {frame_id: [x1, y1, x2, y2]} for target person
    """
    gt_bboxes = {}
    
    if not osp.exists(gt_file):
        print(f"Warning: GT file not found: {gt_file}")
        return gt_bboxes
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) < 6:
                continue
            
            frame_id = int(parts[0])
            obj_id = int(parts[1])
            x = float(parts[2])
            y = float(parts[3])
            w = float(parts[4])
            h = float(parts[5])
            
            # Only keep target person
            if obj_id == target_id:
                # Convert from (x, y, w, h) to (x1, y1, x2, y2)
                x1, y1 = x, y
                x2, y2 = x + w, y + h
                gt_bboxes[frame_id] = [x1, y1, x2, y2]
    
    return gt_bboxes


def compute_iou(bbox1, bbox2):
    """
    Calculate Intersection over Union (IoU) between two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        float: IoU score between 0 and 1
    """
    x1 = max(bbox1[0], bbox2[0])
    y1 = max(bbox1[1], bbox2[1])
    x2 = min(bbox1[2], bbox2[2])
    y2 = min(bbox1[3], bbox2[3])
    
    inter_area = max(0, x2 - x1) * max(0, y2 - y1)
    
    bbox1_area = (bbox1[2] - bbox1[0]) * (bbox1[3] - bbox1[1])
    bbox2_area = (bbox2[2] - bbox2[0]) * (bbox2[3] - bbox2[1])
    union_area = bbox1_area + bbox2_area - inter_area
    
    iou = inter_area / union_area if union_area > 0 else 0
    return iou


def compute_center_error(bbox1, bbox2):
    """
    Compute center location error in pixels between two bounding boxes.
    
    Args:
        bbox1: [x1, y1, x2, y2]
        bbox2: [x1, y1, x2, y2]
    
    Returns:
        float: Euclidean distance between centers
    """
    c1_x = (bbox1[0] + bbox1[2]) / 2
    c1_y = (bbox1[1] + bbox1[3]) / 2
    c2_x = (bbox2[0] + bbox2[2]) / 2
    c2_y = (bbox2[1] + bbox2[3]) / 2
    
    distance = math.sqrt((c1_x - c2_x) ** 2 + (c1_y - c2_y) ** 2)
    return distance


def init_target_from_gt(bboxes, track_ids, gt_bbox, iou_threshold=0.3):
    """
    Initialize target person by matching ground truth bbox with tracked persons.
    
    Args:
        bboxes: Array of [x1, y1, x2, y2, score] bounding boxes
        track_ids: Array of track IDs corresponding to bboxes
        gt_bbox: Ground truth bounding box [x1, y1, x2, y2]
        iou_threshold: Minimum IoU threshold to consider a match
    
    Returns:
        tuple: (target_track_id, target_bbox, max_iou) or (None, None, 0) if no match
    """
    if gt_bbox is None or len(bboxes) == 0:
        return None, None, 0
    
    max_iou = 0
    target_track_id = None
    target_bbox = None
    
    for i, (bbox, track_id) in enumerate(zip(bboxes, track_ids)):
        bbox_coords = bbox[:4]
        iou = compute_iou(bbox_coords, gt_bbox)
        
        if iou > max_iou:
            max_iou = iou
            target_track_id = int(track_id)
            target_bbox = bbox_coords
    
    if max_iou >= iou_threshold:
        print(f"[Init] Target initialized: Track ID={target_track_id}, IoU={max_iou:.3f}")
        return target_track_id, target_bbox, max_iou
    else:
        print(f"[Init] No match found (max IoU={max_iou:.3f} < {iou_threshold})")
        return None, None, 0


class SOTEvaluator:
    """Evaluates single object tracking performance using standard SOT metrics."""
    
    def __init__(self):
        # IoU thresholds for success plot (0 to 1.0 in steps of 0.05)
        self.iou_thresholds = np.arange(0, 1.05, 0.05)
        # Distance thresholds for precision plot (0 to 50 pixels)
        self.distance_thresholds = np.arange(0, 51, 1)
    
    def evaluate_sequence(self, pred_bboxes, gt_bboxes):
        """
        Evaluate tracking on a sequence.
        
        Args:
            pred_bboxes: dict {frame_id: [x1,y1,x2,y2] or None}
            gt_bboxes: dict {frame_id: [x1,y1,x2,y2]}
        
        Returns:
            dict with evaluation metrics
        """
        # Align frames
        all_frames = sorted(set(pred_bboxes.keys()) | set(gt_bboxes.keys()))
        
        ious = []
        center_errors = []
        per_frame_results = []
        
        for frame_id in all_frames:
            pred = pred_bboxes.get(frame_id)
            gt = gt_bboxes.get(frame_id)
            
            if gt is None:
                # No GT for this frame, skip
                continue
            
            if pred is None:
                # Tracking failure
                iou = 0.0
                center_error = float('inf')
                tracked = False
            else:
                iou = compute_iou(pred, gt)
                center_error = compute_center_error(pred, gt)
                tracked = True
            
            ious.append(iou)
            center_errors.append(center_error)
            per_frame_results.append({
                'frame': frame_id,
                'iou': float(iou),
                'center_error': float(center_error) if center_error != float('inf') else None,
                'tracked': tracked
            })
        
        # Compute success rate (percentage of frames with IoU > threshold)
        success_rates = []
        for thr in self.iou_thresholds:
            success_rate = np.mean([iou > thr for iou in ious])
            success_rates.append(float(success_rate))
        
        # Compute precision (percentage of frames with center error < threshold)
        precisions = []
        valid_errors = [e for e in center_errors if e != float('inf')]
        for thr in self.distance_thresholds:
            if valid_errors:
                precision = np.mean([e < thr for e in valid_errors])
            else:
                precision = 0.0
            precisions.append(float(precision))
        
        # Compute aggregate metrics
        auc = float(np.mean(success_rates))  # Area Under Curve
        precision_20px = precisions[20] if len(precisions) > 20 else 0.0
        failures = sum([1 for iou in ious if iou == 0])
        
        results = {
            'auc': auc,
            'precision_20px': precision_20px,
            'mean_iou': float(np.mean(ious)),
            'mean_center_error': float(np.mean(valid_errors)) if valid_errors else None,
            'failures': failures,
            'total_frames': len(all_frames),
            'tracked_frames': len(valid_errors),
            'success_rates': success_rates,
            'precisions': precisions,
            'per_frame_results': per_frame_results
        }
        
        return results


def draw_bbox(img, bbox, color, thickness=2, label=None):
    """Draw bounding box on image."""
    if bbox is None:
        return img
    
    x1, y1, x2, y2 = map(int, bbox)
    cv2.rectangle(img, (x1, y1), (x2, y2), color, thickness)
    
    if label:
        (label_w, label_h), baseline = cv2.getTextSize(
            label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, thickness)
        cv2.rectangle(img, (x1, y1 - label_h - 10), 
                     (x1 + label_w, y1), color, -1)
        cv2.putText(img, label, (x1, y1 - 5), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), thickness)
    
    return img


def main():
    parser = ArgumentParser(description='Single Object Tracking Evaluation')
    
    # Tracking model arguments
    parser.add_argument('track_config', help='tracking config file')
    parser.add_argument('--track-checkpoint', help='tracking checkpoint file')
    
    # Dataset arguments
    parser.add_argument('--sequence-dir', required=True,
                       help='path to sequence directory (e.g., test1/dancetrack0003)')
    parser.add_argument('--target-id', type=int, required=True,
                       help='target person ID from ground truth')
    parser.add_argument('--gt-file', default='gt/gt.txt',
                       help='ground truth file path relative to sequence dir')
    
    # Output arguments
    parser.add_argument('--output-json', default='sot_results.json',
                       help='output JSON file for evaluation results')
    parser.add_argument('--output-video', help='output video file (optional)')
    parser.add_argument('--show', action='store_true', 
                       help='show tracking results')
    
    # Tracking parameters
    parser.add_argument('--score-thr', type=float, default=0.0,
                       help='bounding box score threshold')
    parser.add_argument('--iou-threshold', type=float, default=0.3,
                       help='IoU threshold for target initialization')
    
    # Device
    parser.add_argument('--device', default='cpu', 
                       help='device (cpu or cuda:0)')
    parser.add_argument('--fps', type=int, default=20,
                       help='video FPS for visualization')
    
    args = parser.parse_args()
    
    # Setup paths
    sequence_dir = args.sequence_dir
    gt_file_path = osp.join(sequence_dir, args.gt_file)
    img_dir = osp.join(sequence_dir, 'img1')
    
    # Validate paths
    if not osp.exists(sequence_dir):
        raise FileNotFoundError(f"Sequence directory not found: {sequence_dir}")
    if not osp.exists(img_dir):
        raise FileNotFoundError(f"Image directory not found: {img_dir}")
    
    # Load ground truth
    print(f"Loading ground truth for target ID {args.target_id}...")
    gt_bboxes = load_mot_gt(gt_file_path, args.target_id)
    
    if not gt_bboxes:
        raise ValueError(f"No ground truth found for target ID {args.target_id}")
    
    print(f"Loaded {len(gt_bboxes)} GT frames for target {args.target_id}")
    
    # Initialize tracking model
    print('Loading tracking model...')
    track_model = init_track_model(
        args.track_config, args.track_checkpoint, device=args.device)
    
    # Get image files
    img_files = sorted([f for f in os.listdir(img_dir) if f.endswith('.jpg')])
    print(f"Found {len(img_files)} frames")
    
    # Initialize tracking state
    target_track_id = None
    pred_bboxes = {}
    
    # Setup visualization
    if args.output_video:
        os.makedirs(osp.dirname(args.output_video) or '.', exist_ok=True)
    
    prog_bar = mmcv.ProgressBar(len(img_files))
    
    # Process each frame
    for frame_idx, img_file in enumerate(img_files):
        frame_id = int(osp.splitext(img_file)[0])  # Extract frame number
        img_path = osp.join(img_dir, img_file)
        frame = cv2.imread(img_path)
        
        if frame is None:
            print(f"Warning: Could not read frame {img_path}")
            continue
        
        # Run tracking
        track_result = inference_mot(track_model, frame, frame_id=frame_idx)
        track_bboxes = track_result.get('track_bboxes', [])
        
        pred_bbox = None
        
        if len(track_bboxes) > 0 and len(track_bboxes[0]) > 0:
            bboxes = track_bboxes[0]
            
            # Filter by score threshold
            valid_mask = bboxes[:, 5] >= args.score_thr
            bboxes = bboxes[valid_mask]
            
            if len(bboxes) > 0:
                track_ids = bboxes[:, 0].astype(int)
                bbox_coords = bboxes[:, 1:5]  # x1, y1, x2, y2
                
                # Initialize target in first frame with GT
                if target_track_id is None and frame_id in gt_bboxes:
                    gt_bbox = gt_bboxes[frame_id]
                    target_track_id, _, _ = init_target_from_gt(
                        bbox_coords, track_ids, gt_bbox, 
                        iou_threshold=args.iou_threshold)
                
                # Track target
                if target_track_id is not None:
                    target_idx = np.where(track_ids == target_track_id)[0]
                    if len(target_idx) > 0:
                        pred_bbox = bbox_coords[target_idx[0]]
        
        # Store prediction
        pred_bboxes[frame_id] = pred_bbox.tolist() if pred_bbox is not None else None
        
        # Visualization
        if args.show or args.output_video:
            vis_frame = frame.copy()
            
            # Draw GT bbox (green)
            if frame_id in gt_bboxes:
                gt_bbox = gt_bboxes[frame_id]
                vis_frame = draw_bbox(vis_frame, gt_bbox, (0, 255, 0), 
                                     thickness=2, label='GT')
            
            # Draw predicted bbox (blue)
            if pred_bbox is not None:
                iou = compute_iou(pred_bbox, gt_bboxes.get(frame_id, pred_bbox))
                label = f'Pred (IoU:{iou:.2f})'
                vis_frame = draw_bbox(vis_frame, pred_bbox, (255, 0, 0), 
                                     thickness=2, label=label)
            else:
                # Tracking failure
                cv2.putText(vis_frame, 'LOST', (50, 100),
                           cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 0, 255), 3)
            
            # Add frame info
            cv2.putText(vis_frame, f'Frame: {frame_id}', (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            if args.show:
                cv2.imshow('SOT Evaluation', vis_frame)
                if cv2.waitKey(int(1000 / args.fps)) & 0xFF == ord('q'):
                    break
            
            if args.output_video:
                # Save frame for video creation
                if frame_idx == 0:
                    h, w = vis_frame.shape[:2]
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                    video_writer = cv2.VideoWriter(
                        args.output_video, fourcc, args.fps, (w, h))
                video_writer.write(vis_frame)
        
        prog_bar.update()
    
    if args.show:
        cv2.destroyAllWindows()
    
    if args.output_video:
        video_writer.release()
        print(f"\nOutput video saved to: {args.output_video}")
    
    # Evaluate
    print("\n" + "="*60)
    print("EVALUATING TRACKING PERFORMANCE")
    print("="*60)
    
    evaluator = SOTEvaluator()
    results = evaluator.evaluate_sequence(pred_bboxes, gt_bboxes)
    
    # Print results
    print(f"\nSequence: {osp.basename(sequence_dir)}")
    print(f"Target ID: {args.target_id}")
    print(f"Total Frames: {results['total_frames']}")
    print(f"Tracked Frames: {results['tracked_frames']}")
    print(f"Failures: {results['failures']}")
    print(f"\nSuccess Rate (AUC): {results['auc']:.4f}")
    print(f"Precision @ 20px: {results['precision_20px']:.4f}")
    print(f"Mean IoU: {results['mean_iou']:.4f}")
    if results['mean_center_error'] is not None:
        print(f"Mean Center Error: {results['mean_center_error']:.2f} pixels")
    
    # Save results to JSON
    output_data = {
        'sequence': osp.basename(sequence_dir),
        'target_id': args.target_id,
        'metrics': results
    }
    
    # Create output directory if it doesn't exist
    output_dir = osp.dirname(args.output_json)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    
    with open(args.output_json, 'w') as f:
        json.dump(output_data, f, indent=2)
    
    print(f"\nResults saved to: {args.output_json}")
    print("="*60)


if __name__ == '__main__':
    main()

# Made with Bob
