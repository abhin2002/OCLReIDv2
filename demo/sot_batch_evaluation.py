#!/usr/bin/env python
# Copyright (c) OpenMMLab. All rights reserved.
"""
Batch Single Object Tracking (SOT) Evaluation Script
Evaluates multiple sequences and aggregates results.
"""
import os
import os.path as osp
import json
import subprocess
from argparse import ArgumentParser
import numpy as np


def find_sequences(test_dir):
    """Find all sequence directories in test directory."""
    sequences = []
    for item in os.listdir(test_dir):
        seq_path = osp.join(test_dir, item)
        if osp.isdir(seq_path):
            img_dir = osp.join(seq_path, 'img1')
            gt_file = osp.join(seq_path, 'gt', 'gt.txt')
            if osp.exists(img_dir):
                sequences.append(item)
    return sorted(sequences)


def get_target_ids_from_gt(gt_file):
    """Extract all unique person IDs from ground truth file."""
    target_ids = set()
    if not osp.exists(gt_file):
        return []
    
    with open(gt_file, 'r') as f:
        for line in f:
            parts = line.strip().split(',')
            if len(parts) >= 2:
                obj_id = int(parts[1])
                target_ids.add(obj_id)
    
    return sorted(list(target_ids))


def run_sot_evaluation(track_config, track_checkpoint, sequence_dir, 
                       target_id, output_json, device='cpu'):
    """Run SOT evaluation for a single sequence and target."""
    cmd = [
        'python', 'demo/sot_evaluation.py',
        track_config,
        '--track-checkpoint', track_checkpoint,
        '--sequence-dir', sequence_dir,
        '--target-id', str(target_id),
        '--output-json', output_json,
        '--device', device
    ]
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, check=True)
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        return False, e.stderr


def aggregate_results(results_list):
    """Aggregate results from multiple evaluations."""
    if not results_list:
        return {}
    
    # Collect metrics
    aucs = []
    precisions_20px = []
    mean_ious = []
    failures = []
    total_frames = []
    
    for result in results_list:
        metrics = result.get('metrics', {})
        aucs.append(metrics.get('auc', 0))
        precisions_20px.append(metrics.get('precision_20px', 0))
        mean_ious.append(metrics.get('mean_iou', 0))
        failures.append(metrics.get('failures', 0))
        total_frames.append(metrics.get('total_frames', 0))
    
    # Compute aggregate statistics
    aggregate = {
        'num_sequences': len(results_list),
        'auc': {
            'mean': float(np.mean(aucs)),
            'std': float(np.std(aucs)),
            'min': float(np.min(aucs)),
            'max': float(np.max(aucs))
        },
        'precision_20px': {
            'mean': float(np.mean(precisions_20px)),
            'std': float(np.std(precisions_20px)),
            'min': float(np.min(precisions_20px)),
            'max': float(np.max(precisions_20px))
        },
        'mean_iou': {
            'mean': float(np.mean(mean_ious)),
            'std': float(np.std(mean_ious)),
            'min': float(np.min(mean_ious)),
            'max': float(np.max(mean_ious))
        },
        'total_failures': int(np.sum(failures)),
        'total_frames': int(np.sum(total_frames)),
        'per_sequence_results': results_list
    }
    
    return aggregate


def main():
    parser = ArgumentParser(description='Batch SOT Evaluation')
    
    # Model arguments
    parser.add_argument('track_config', help='tracking config file')
    parser.add_argument('--track-checkpoint', required=True,
                       help='tracking checkpoint file')
    
    # Dataset arguments
    parser.add_argument('--test-dir', required=True,
                       help='test directory containing sequences (e.g., test1/)')
    parser.add_argument('--sequences', nargs='+',
                       help='specific sequences to evaluate (default: all)')
    parser.add_argument('--target-ids', type=int, nargs='+',
                       help='specific target IDs to evaluate (default: all in GT)')
    parser.add_argument('--max-targets-per-seq', type=int, default=None,
                       help='maximum number of targets to evaluate per sequence')
    
    # Output arguments
    parser.add_argument('--output-dir', default='sot_results',
                       help='output directory for results')
    
    # Device
    parser.add_argument('--device', default='cpu',
                       help='device (cpu or cuda:0)')
    
    args = parser.parse_args()
    
    # Create output directory
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Find sequences
    if args.sequences:
        sequences = args.sequences
    else:
        sequences = find_sequences(args.test_dir)
    
    print(f"Found {len(sequences)} sequences to evaluate")
    print(f"Sequences: {sequences}")
    
    # Evaluate each sequence
    all_results = []
    
    for seq_name in sequences:
        seq_dir = osp.join(args.test_dir, seq_name)
        gt_file = osp.join(seq_dir, 'gt', 'gt.txt')
        
        print(f"\n{'='*60}")
        print(f"Processing sequence: {seq_name}")
        print(f"{'='*60}")
        
        # Get target IDs
        if args.target_ids:
            target_ids = args.target_ids
        else:
            target_ids = get_target_ids_from_gt(gt_file)
        
        if args.max_targets_per_seq:
            target_ids = target_ids[:args.max_targets_per_seq]
        
        print(f"Target IDs: {target_ids}")
        
        # Evaluate each target
        for target_id in target_ids:
            print(f"\n--- Evaluating Target ID: {target_id} ---")
            
            output_json = osp.join(
                args.output_dir, 
                f'{seq_name}_target{target_id}_results.json'
            )
            
            success, output = run_sot_evaluation(
                args.track_config,
                args.track_checkpoint,
                seq_dir,
                target_id,
                output_json,
                device=args.device
            )
            
            if success:
                print(f"✓ Evaluation completed for {seq_name} - Target {target_id}")
                
                # Load results
                with open(output_json, 'r') as f:
                    result = json.load(f)
                    all_results.append(result)
            else:
                print(f"✗ Evaluation failed for {seq_name} - Target {target_id}")
                print(f"Error: {output}")
    
    # Aggregate results
    print(f"\n{'='*60}")
    print("AGGREGATE RESULTS")
    print(f"{'='*60}")
    
    aggregate = aggregate_results(all_results)
    
    print(f"\nTotal Sequences Evaluated: {aggregate['num_sequences']}")
    print(f"Total Frames: {aggregate['total_frames']}")
    print(f"Total Failures: {aggregate['total_failures']}")
    print(f"\nSuccess Rate (AUC):")
    print(f"  Mean: {aggregate['auc']['mean']:.4f} ± {aggregate['auc']['std']:.4f}")
    print(f"  Range: [{aggregate['auc']['min']:.4f}, {aggregate['auc']['max']:.4f}]")
    print(f"\nPrecision @ 20px:")
    print(f"  Mean: {aggregate['precision_20px']['mean']:.4f} ± {aggregate['precision_20px']['std']:.4f}")
    print(f"  Range: [{aggregate['precision_20px']['min']:.4f}, {aggregate['precision_20px']['max']:.4f}]")
    print(f"\nMean IoU:")
    print(f"  Mean: {aggregate['mean_iou']['mean']:.4f} ± {aggregate['mean_iou']['std']:.4f}")
    print(f"  Range: [{aggregate['mean_iou']['min']:.4f}, {aggregate['mean_iou']['max']:.4f}]")
    
    # Save aggregate results
    aggregate_file = osp.join(args.output_dir, 'aggregate_results.json')
    with open(aggregate_file, 'w') as f:
        json.dump(aggregate, f, indent=2)
    
    print(f"\nAggregate results saved to: {aggregate_file}")
    print(f"{'='*60}")


if __name__ == '__main__':
    main()

# Made with Bob
