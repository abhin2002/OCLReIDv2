# Copyright (c) OpenMMLab. All rights reserved.
"""Demo script for MMPose - Human Pose Estimation."""
import os
import os.path as osp
from argparse import ArgumentParser

import mmcv
from mmcv import Config

from mmpose.apis import (inference_top_down_pose_model, init_pose_model,
                         vis_pose_result)
from mmpose.datasets import DatasetInfo


def main():
    parser = ArgumentParser()
    parser.add_argument('--config', 
                        default='configs/pose/hrnet_w48_coco_256x192.py',
                        help='Config file for pose estimation')
    parser.add_argument('--checkpoint',
                        default='checkpoints/hrnet_w48_coco_256x192.pth',
                        help='Checkpoint file for pose estimation')
    parser.add_argument('--img', help='Image file')
    parser.add_argument('--output', help='Output image file')
    parser.add_argument(
        '--device', default='cpu', help='Device used for inference')
    parser.add_argument(
        '--kpt-thr',
        type=float,
        default=0.3,
        help='Keypoint score threshold')
    parser.add_argument(
        '--show',
        action='store_true',
        help='whether to show the results on the fly')
    args = parser.parse_args()

    # Build the pose model from a config file and a checkpoint file
    pose_model = init_pose_model(
        args.config, args.checkpoint, device=args.device)

    dataset = pose_model.cfg.data['test']['type']
    dataset_info = pose_model.cfg.data['test'].get('dataset_info', None)
    if dataset_info is None:
        # For COCO dataset
        dataset_info = DatasetInfo(
            dict(
                dataset_name='coco',
                paper_info=dict(
                    author='Lin, Tsung-Yi and Maire, Michael',
                    title='Microsoft COCO: Common Objects in Context',
                    year='2014'),
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
                joint_weights=[
                    1., 1., 1., 1., 1., 1., 1., 1.2, 1.2,
                    1.5, 1.5, 1., 1., 1.2, 1.2, 1.5, 1.5
                ],
                sigmas=[
                    0.026, 0.025, 0.025, 0.035, 0.035, 0.079, 0.079, 0.072, 0.072,
                    0.062, 0.062, 0.107, 0.107, 0.087, 0.087, 0.089, 0.089
                ]
            )
        )
    else:
        dataset_info = DatasetInfo(dataset_info)

    # Read the image
    img = mmcv.imread(args.img)
    
    # Create a person bounding box (full image as one person)
    h, w = img.shape[:2]
    person_results = [{'bbox': [0, 0, w, h, 1.0]}]
    
    # Run inference
    pose_results, returned_outputs = inference_top_down_pose_model(
        pose_model,
        args.img,
        person_results,
        bbox_thr=None,
        format='xyxy',
        dataset=dataset,
        dataset_info=dataset_info,
        return_heatmap=False,
        outputs=None)

    # Visualize the results
    vis_result = vis_pose_result(
        pose_model,
        args.img,
        pose_results,
        dataset=dataset,
        dataset_info=dataset_info,
        kpt_score_thr=args.kpt_thr,
        radius=4,
        thickness=1,
        show=args.show,
        out_file=args.output)

    print(f'Pose estimation complete!')
    if args.output:
        print(f'Result saved to {args.output}')
    
    # Print keypoint info
    if pose_results:
        keypoints = pose_results[0]['keypoints']
        print(f'\nDetected {len(keypoints)} keypoints:')
        kpt_names = ['nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
                     'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
                     'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
                     'left_knee', 'right_knee', 'left_ankle', 'right_ankle']
        for i, (kpt, name) in enumerate(zip(keypoints, kpt_names)):
            if kpt[2] > args.kpt_thr:
                print(f'  {name}: ({kpt[0]:.1f}, {kpt[1]:.1f}) conf={kpt[2]:.2f}')


if __name__ == '__main__':
    main()
