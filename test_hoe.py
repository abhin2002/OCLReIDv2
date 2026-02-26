#!/usr/bin/env python
"""Test script for HOE orientation estimation."""

from mmtrack.models.orientation.hoe_estimator import init_hoe_model
import numpy as np

def main():
    print("Initializing HOE model...")
    hoe = init_hoe_model(device='cpu')
    
    # Define bbox (typical person bounding box)
    bbox = [50, 20, 150, 280]  # [x1, y1, x2, y2]
    
    # Test 1: Person facing front (nose in front of hips)
    print('\nTest 1: Person facing front')
    kpts_front = np.zeros((14, 3), dtype=np.float32)
    kpts_front[0] = [100, 50, 0.9]   # Nose
    kpts_front[1] = [80, 100, 0.9]   # LShoulder
    kpts_front[2] = [120, 100, 0.9]  # RShoulder  
    kpts_front[3] = [70, 130, 0.9]   # LElbow
    kpts_front[4] = [130, 130, 0.9]  # RElbow
    kpts_front[5] = [60, 160, 0.9]   # LWrist
    kpts_front[6] = [140, 160, 0.9]  # RWrist
    kpts_front[7] = [85, 160, 0.9]   # LHip
    kpts_front[8] = [115, 160, 0.9]  # RHip
    kpts_front[9] = [80, 210, 0.9]   # LKnee
    kpts_front[10] = [120, 210, 0.9] # RKnee
    kpts_front[11] = [75, 260, 0.9]  # LAnkle
    kpts_front[12] = [125, 260, 0.9] # RAnkle
    kpts_front[13] = [100, 90, 0.9]  # Neck
    
    ori, bin_ori, conf = hoe.estimate_orientation([kpts_front], bboxes=[bbox])
    print(f'  Orientation: {ori[0]} deg, Binary: {"Front" if bin_ori[0]==0 else "Back"}, Conf: {conf[0]:.3f}')
    
    # Test 2: Person facing back (swap left/right)
    print('\nTest 2: Person facing back')
    kpts_back = kpts_front.copy()
    kpts_back[1], kpts_back[2] = kpts_front[2].copy(), kpts_front[1].copy()
    kpts_back[3], kpts_back[4] = kpts_front[4].copy(), kpts_front[3].copy()
    kpts_back[5], kpts_back[6] = kpts_front[6].copy(), kpts_front[5].copy()
    kpts_back[7], kpts_back[8] = kpts_front[8].copy(), kpts_front[7].copy()
    kpts_back[9], kpts_back[10] = kpts_front[10].copy(), kpts_front[9].copy()
    kpts_back[11], kpts_back[12] = kpts_front[12].copy(), kpts_front[11].copy()
    kpts_back[0, 2] = 0.2  # Low confidence on nose from back
    
    ori, bin_ori, conf = hoe.estimate_orientation([kpts_back], bboxes=[bbox])
    print(f'  Orientation: {ori[0]} deg, Binary: {"Front" if bin_ori[0]==0 else "Back"}, Conf: {conf[0]:.3f}')
    
    # Test 3: Person facing left (side view)
    print('\nTest 3: Person facing left (side view)')
    kpts_left = np.zeros((14, 3), dtype=np.float32)
    kpts_left[0] = [80, 50, 0.9]
    kpts_left[1] = [100, 100, 0.9]  # LShoulder behind
    kpts_left[2] = [85, 100, 0.9]   # RShoulder in front
    kpts_left[3] = [110, 130, 0.9]
    kpts_left[4] = [80, 130, 0.9]
    kpts_left[5] = [115, 160, 0.9]
    kpts_left[6] = [75, 160, 0.9]
    kpts_left[7] = [100, 160, 0.9]
    kpts_left[8] = [90, 160, 0.9]
    kpts_left[9] = [105, 210, 0.9]
    kpts_left[10] = [85, 210, 0.9]
    kpts_left[11] = [100, 260, 0.9]
    kpts_left[12] = [90, 260, 0.9]
    kpts_left[13] = [90, 90, 0.9]
    
    ori, bin_ori, conf = hoe.estimate_orientation([kpts_left], bboxes=[bbox])
    print(f'  Orientation: {ori[0]} deg, Binary: {"Front" if bin_ori[0]==0 else "Back"}, Conf: {conf[0]:.3f}')
    
    # Test 4: Debug - print raw network output distribution
    print('\n--- Debug: Raw output distribution ---')
    import torch
    kpts_17 = hoe.convert_14kpt_to_17kpt(kpts_front, bbox)
    normalized = hoe.normalize_keypoints(kpts_17, bbox)
    print(f"Keypoints 17 shape: {kpts_17.shape}")
    print(f"Non-zero kpts in 17-format: {(kpts_17[:, 2] > 0.1).sum()}")
    print(f"Normalized input (indices 0-9, nose+eyes+ears): {normalized[:10]}")
    print(f"Normalized input (indices 10-17, shoulders): {normalized[10:18]}")
    print(f"Normalized input (indices 22-29, hips): {normalized[22:30]}")
    
    batch_tensor = torch.tensor(np.array([normalized]), dtype=torch.float32).to(hoe.device)
    with torch.no_grad():
        outputs = hoe.model(batch_tensor)
        probs = torch.softmax(outputs, dim=1)
        top5_bins = probs[0].argsort(descending=True)[:5].tolist()
        top5_probs = probs[0].sort(descending=True).values[:5].tolist()
        print(f"Top-5 bins (x5=degrees): {[b*5 for b in top5_bins]}")
        print(f"Top-5 probs: {[f'{p:.4f}' for p in top5_probs]}")

if __name__ == '__main__':
    main()
