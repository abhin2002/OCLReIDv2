"""
Human Orientation Estimation (HOE) Wrapper

This module provides a simple interface for estimating human body orientation
from 2D keypoints using the KeypointsNet model.

Input: 14-keypoint format (from pose estimation)
Output: Orientation in degrees (0-355) and binary orientation (front/back)
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F


# Define KeypointsNet directly here to avoid import issues
class MyLinearSimple(nn.Module):
    def __init__(self, linear_size, p_dropout=0.5):
        super().__init__()
        self.l_size = linear_size

        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(p_dropout)

        self.w1 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm1 = nn.BatchNorm1d(self.l_size)

        self.w2 = nn.Linear(self.l_size, self.l_size)
        self.batch_norm2 = nn.BatchNorm1d(self.l_size)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        y = self.w2(y)
        y = self.batch_norm2(y)
        y = self.relu(y)
        y = self.dropout(y)

        out = x + y
        return out


class KeypointsNet(nn.Module):
    """
    Neural network for estimating human body orientation from 2D keypoints.
    
    Input: 34 values (17 keypoints × 2 coordinates)
    Output: 72 bins (360°/5° = 72 orientation bins)
    """
    def __init__(self):
        super(KeypointsNet, self).__init__()
        self.output_size = 72
        self.input_size = 34
        self.linear_size = 512
        self.p_dropout = 0.2
        self.num_stage = 3
        
        # Preprocessing
        self.w1 = nn.Linear(self.input_size, self.linear_size)
        self.batch_norm1 = nn.BatchNorm1d(self.linear_size)

        # Internal residual blocks
        self.linear_stages = nn.ModuleList([
            MyLinearSimple(self.linear_size, self.p_dropout) 
            for _ in range(self.num_stage)
        ])

        # Post processing
        self.w2 = nn.Linear(self.linear_size, self.linear_size)
        self.w3 = nn.Linear(self.linear_size, self.linear_size)
        self.batch_norm3 = nn.BatchNorm1d(self.linear_size)

        # Final output layer
        self.w_fin = nn.Linear(self.linear_size, self.output_size)

        # Activation and regularization
        self.relu = nn.ReLU(inplace=True)
        self.dropout = nn.Dropout(self.p_dropout)

    def forward(self, x):
        y = self.w1(x)
        y = self.batch_norm1(y)
        y = self.relu(y)
        y = self.dropout(y)

        for stage in self.linear_stages:
            y = stage(y)

        y = self.w2(y)
        y = self.w3(y)
        y = self.batch_norm3(y)
        y = self.relu(y)
        y = self.dropout(y)
        y = self.w_fin(y)

        return y


class HOEEstimator:
    """
    Human Orientation Estimator using keypoints-based neural network.
    
    The network takes normalized 2D keypoints (17 COCO keypoints) and outputs
    72 bins representing orientation in 5-degree increments (360°/5° = 72 bins).
    
    Args:
        checkpoint_path: Path to the trained model checkpoint
        device: Device to run inference on ('cpu' or 'cuda:0')
    """
    
    def __init__(self, checkpoint_path=None, device='cpu'):
        self.device = device
        
        # Default checkpoint path
        if checkpoint_path is None:
            base_dir = os.path.dirname(os.path.abspath(__file__))
            checkpoint_path = os.path.join(base_dir, 'checkpoints', 'keypoints_net.pth')
        
        # Initialize model
        self.model = KeypointsNet()
        
        # Load weights
        if os.path.exists(checkpoint_path):
            state_dict = torch.load(checkpoint_path, map_location=device)
            # Handle DataParallel wrapped models
            if any(k.startswith('module.') for k in state_dict.keys()):
                state_dict = {k.replace('module.', ''): v for k, v in state_dict.items()}
            self.model.load_state_dict(state_dict, strict=False)
            print(f'HOE model loaded from {checkpoint_path}')
        else:
            print(f'Warning: HOE checkpoint not found at {checkpoint_path}')
        
        self.model.to(device)
        self.model.eval()
    
    def convert_14kpt_to_17kpt(self, keypoints_14, bbox=None):
        """
        Convert 14-keypoint format to the COCO 17-keypoint format expected by the model.
        
        Based on the original process_kpts function:
        - The model expects 13 keypoints (no neck) mapped to COCO 17 format
        - coco[0] = pose[0] (Nose)
        - coco[5-16] = pose[1-12] (body keypoints)
        - coco[1-4] (eyes/ears) are left as zeros
        
        14-keypoint input format:
            0: Nose, 1: LShoulder, 2: RShoulder, 3: LElbow, 4: RElbow,
            5: LWrist, 6: RWrist, 7: LHip, 8: RHip, 9: LKnee, 10: RKnee,
            11: LAnkle, 12: RAnkle, 13: Neck (ignored)
        
        Args:
            keypoints_14: numpy array (14, 3) with [x, y, confidence]
            bbox: [x1, y1, x2, y2] bounding box for clamping
        
        Returns:
            numpy array (17, 3) with [x, y, confidence]
        """
        keypoints_17 = np.zeros((17, 3), dtype=np.float32)
        
        # Mapping: 14-kpt index -> COCO 17-kpt index
        # This matches the original: coco[j] = track_kpts[j-4] for j in range(5, 17)
        mapping = {
            0: 0,    # Nose -> nose
            1: 5,    # LShoulder -> left_shoulder
            2: 6,    # RShoulder -> right_shoulder
            3: 7,    # LElbow -> left_elbow
            4: 8,    # RElbow -> right_elbow
            5: 9,    # LWrist -> left_wrist
            6: 10,   # RWrist -> right_wrist
            7: 11,   # LHip -> left_hip
            8: 12,   # RHip -> right_hip
            9: 13,   # LKnee -> left_knee
            10: 14,  # RKnee -> right_knee
            11: 15,  # LAnkle -> left_ankle
            12: 16,  # RAnkle -> right_ankle
            # 13: Neck - not used in original model
        }
        
        for kpt14_idx, coco_idx in mapping.items():
            keypoints_17[coco_idx] = keypoints_14[kpt14_idx]
        
        # Clamp keypoints to bounding box (as in original code)
        if bbox is not None:
            tl_x, tl_y, br_x, br_y = bbox[:4]
            for i in range(17):
                if keypoints_17[i, 2] > 0:
                    keypoints_17[i, 0] = np.clip(keypoints_17[i, 0], tl_x, br_x)
                    keypoints_17[i, 1] = np.clip(keypoints_17[i, 1], tl_y, br_y)
        
        return keypoints_17
    
    def normalize_keypoints(self, keypoints, bbox):
        """
        Normalize keypoints exactly as in the original process_kpts function.
        
        Original code:
            kpts[:,0] = (kpts[:,0]-tl_x)/(br_x-tl_x)*input_width
            kpts[:,1] = (kpts[:,1]-tl_y)/(br_y-tl_y)*input_height
            coco_kpts[j, :] = track_kpts[i, j-4, :2] * track_kpts[i, j-4, 2]
        
        Args:
            keypoints: numpy array (17, 3) with [x, y, confidence]
            bbox: [x1, y1, x2, y2] bounding box
        
        Returns:
            numpy array (34,) normalized keypoint coordinates
        """
        # Original model uses image_patch_size = [192, 256] (width, height)
        INPUT_WIDTH = 192.0
        INPUT_HEIGHT = 256.0
        
        tl_x, tl_y, br_x, br_y = bbox[:4]
        bbox_width = max(br_x - tl_x, 1)
        bbox_height = max(br_y - tl_y, 1)
        
        # Normalize keypoints as in original code
        normalized = np.zeros(34, dtype=np.float32)
        for i in range(17):
            conf = keypoints[i, 2]
            if conf > 0.1:
                # Normalize to [0, INPUT_WIDTH/HEIGHT] range
                norm_x = (keypoints[i, 0] - tl_x) / bbox_width * INPUT_WIDTH
                norm_y = (keypoints[i, 1] - tl_y) / bbox_height * INPUT_HEIGHT
                # Multiply by confidence (as in original: kpts[:2] * score)
                normalized[i * 2] = norm_x * conf
                normalized[i * 2 + 1] = norm_y * conf
            else:
                normalized[i * 2] = 0.0
                normalized[i * 2 + 1] = 0.0
        
        return normalized
    
    def estimate_orientation(self, keypoints_14_list, bboxes=None):
        """
        Estimate orientation for multiple persons.
        
        Args:
            keypoints_14_list: List of (14, 3) numpy arrays with [x, y, confidence]
            bboxes: List of [x1, y1, x2, y2] bounding boxes (required for proper normalization)
        
        Returns:
            orientations: List of int (0-355 degrees, in 5° increments)
            binary_orientations: List of int (0=Front, 1=Back)
            confidences: List of float (max softmax probability)
        """
        if len(keypoints_14_list) == 0:
            return [], [], []
        
        # Prepare batch input
        batch_inputs = []
        for i, kpts_14 in enumerate(keypoints_14_list):
            # Get bbox (required for proper normalization)
            if bboxes is not None and i < len(bboxes):
                bbox = bboxes[i]
            else:
                # Fallback: compute bbox from keypoints
                valid_mask = kpts_14[:, 2] > 0.1
                if valid_mask.sum() > 0:
                    valid_kpts = kpts_14[valid_mask]
                    bbox = [valid_kpts[:, 0].min(), valid_kpts[:, 1].min(),
                            valid_kpts[:, 0].max(), valid_kpts[:, 1].max()]
                else:
                    bbox = [0, 0, 192, 256]  # Default
            
            # Convert to 17-keypoint format (with bbox clamping)
            kpts_17 = self.convert_14kpt_to_17kpt(kpts_14, bbox)
            
            # Normalize keypoints (as in original process_kpts)
            normalized = self.normalize_keypoints(kpts_17, bbox)
            batch_inputs.append(normalized)
        
        # Convert to tensor
        batch_tensor = torch.tensor(np.array(batch_inputs), dtype=torch.float32).to(self.device)
        
        # Run inference
        with torch.no_grad():
            outputs = self.model(batch_tensor)
            probs = torch.softmax(outputs, dim=1)
            
            # Get orientation predictions
            bin_indices = outputs.argmax(dim=1).cpu().numpy()
            confidences = probs.max(dim=1).values.cpu().numpy()
        
        # Convert to degrees (72 bins × 5° = 360°)
        orientations = (bin_indices * 5).astype(int).tolist()
        
        # Binary orientation based on model convention:
        # The model outputs ~180° for front-facing, ~0°/360° for back-facing
        # So: 90°-270° = Front (0), 0°-90° or 270°-360° = Back (1)
        binary_orientations = []
        for ori in orientations:
            if 90 <= ori < 270:
                binary_orientations.append(0)  # Front
            else:
                binary_orientations.append(1)  # Back
        
        return orientations, binary_orientations, confidences.tolist()


def init_hoe_model(checkpoint_path=None, device='cpu'):
    """
    Initialize the HOE estimator model.
    
    Args:
        checkpoint_path: Path to model checkpoint (optional)
        device: 'cpu' or 'cuda:0'
    
    Returns:
        HOEEstimator instance
    """
    return HOEEstimator(checkpoint_path=checkpoint_path, device=device)
