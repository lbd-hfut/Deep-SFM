#!/usr/bin/env python3
"""
Evaluation script for DeepSFM.
Computes depth and pose metrics on test datasets.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Any
import json

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.deepsfm import DeepSFM
from datasets import get_dataset
from datasets.transforms import create_val_transforms
from utils.metrics import compute_depth_metrics, compute_pose_metrics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Evaluate DeepSFM models")
    parser.add_argument(
        "--config",
        type=str,
        required=True,
        help="Path to configuration file"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="demon",
        choices=["demon", "kitti", "custom"],
        help="Dataset to evaluate on"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        required=True,
        help="Root directory of dataset"
    )
    parser.add_argument(
        "--depth-checkpoint",
        type=str,
        required=True,
        help="Path to depth subnet checkpoint"
    )
    parser.add_argument(
        "--pose-checkpoint",
        type=str,
        required=True,
        help="Path to pose subnet checkpoint"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="evaluation_results",
        help="Directory to save evaluation results"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=1,
        help="Batch size for evaluation"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=None,
        help="Number of samples to evaluate (default: all)"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of alternating iterations"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    parser.add_argument(
        "--save-predictions",
        action="store_true",
        help="Save predicted depth and poses"
    )
    
    return parser.parse_args()


def evaluate_depth(
    model: DeepSFM,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int = 1,
    num_samples: int = None,
    iterations: int = 5,
) -> Dict[str, List[float]]:
    """Evaluate depth estimation performance.
    
    Args:
        model: DeepSFM model
        dataset: Evaluation dataset
        device: Device to use
        batch_size: Batch size for evaluation
        num_samples: Number of samples to evaluate
        iterations: Number of alternating iterations
    
    Returns:
        Dictionary of metric names to lists of values per sample
    """
    model.eval()
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Initialize metrics storage
    all_metrics = {
        'abs_rel': [],
        'sq_rel': [],
        'rmse': [],
        'rmse_log': [],
        'a1': [],
        'a2': [],
        'a3': [],
    }
    
    samples_processed = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if num_samples and samples_processed >= num_samples:
                break
            
            # Move to device
            ref_img = batch['ref_image'].to(device)
            src_imgs = [img.to(device) for img in batch['src_images']]
            intrinsics = batch['intrinsics'].to(device)
            
            # Ground truth depth
            gt_depth = batch.get('ref_depth')
            if gt_depth is not None:
                gt_depth = gt_depth.to(device)
            
            # Run inference
            output = model(
                ref_img=ref_img,
                src_imgs=src_imgs,
                intrinsics=intrinsics,
                init_depth=None,
                init_poses=None,
                num_iterations=iterations,
                return_all_iterations=False,
            )
            
            pred_depth = output['depth']
            
            # Compute metrics if ground truth available
            if gt_depth is not None:
                # Create mask for valid depth
                valid_mask = (gt_depth > 0) & (gt_depth < 10.0)
                
                # Compute metrics
                metrics = compute_depth_metrics(pred_depth, gt_depth, valid_mask)
                
                # Store metrics
                for key in all_metrics.keys():
                    if key in metrics:
                        all_metrics[key].append(metrics[key])
            
            samples_processed += batch_size
            
            if batch_idx % 10 == 0:
                print(f"  Processed {samples_processed} samples")
    
    return all_metrics


def evaluate_pose(
    model: DeepSFM,
    dataset: torch.utils.data.Dataset,
    device: torch.device,
    batch_size: int = 1,
    num_samples: int = None,
    iterations: int = 5,
) -> Dict[str, List[float]]:
    """Evaluate pose estimation performance.
    
    Args:
        model: DeepSFM model
        dataset: Evaluation dataset
        device: Device to use
        batch_size: Batch size for evaluation
        num_samples: Number of samples to evaluate
        iterations: Number of alternating iterations
    
    Returns:
        Dictionary of metric names to lists of values per sample
    """
    model.eval()
    
    # Create data loader
    data_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    
    # Initialize metrics storage
    all_metrics = {
        'rotation_error': [],
        'translation_error': [],
        'translation_angular_error': [],
    }
    
    samples_processed = 0
    
    with torch.no_grad():
        for batch_idx, batch in enumerate(data_loader):
            if num_samples and samples_processed >= num_samples:
                break
            
            # Move to device
            ref_img = batch['ref_image'].to(device)
            src_imgs = [img.to(device) for img in batch['src_images']]
            intrinsics = batch['intrinsics'].to(device)
            
            # Ground truth poses
            gt_poses = batch.get('relative_poses')
            if gt_poses is not None:
                gt_poses = gt_poses.to(device)
            
            # Run inference
            output = model(
                ref_img=ref_img,
                src_imgs=src_imgs,
                intrinsics=intrinsics,
                init_depth=None,
                init_poses=None,
                num_iterations=iterations,
                return_all_iterations=False,
            )
            
            pred_poses = output['poses']
            
            # Compute metrics if ground truth available
            if gt_poses is not None:
                # Compute metrics for each source view
                batch_size, num_src_views, _, _ = pred_poses.shape
                
                for b in range(batch_size):
                    for src_idx in range(num_src_views):
                        pred_pose = pred_poses[b, src_idx]
                        gt_pose = gt_poses[b, src_idx]
                        
                        metrics = compute_pose_metrics(pred_pose, gt_pose)
                        
                        for key in all_metrics.keys():
                            if key in metrics:
                                all_metrics[key].append(metrics[key])
            
            samples_processed += batch_size
            
            if batch_idx % 10 == 0:
                print(f"  Processed {samples_processed} samples")
    
    return all_metrics


def compute_statistics(metrics: Dict[str, List[float]]) -> Dict[str, Dict[str, float]]:
    """Compute statistics (mean, std, median) for each metric.
    
    Args:
        metrics: Dictionary of metric names to lists of values
    
    Returns:
        Dictionary with statistics for each metric
    """
    statistics = {}
    
    for metric_name, values in metrics.items():
        if not values:
            continue
        
        values_array = np.array(values)
        
        statistics[metric_name] = {
            'mean': float(np.mean(values_array)),
            'std': float(np.std(values_array)),
            'median': float(np.median(values_array)),
            'min': float(np.min(values_array)),
            'max': float(np.max(values_array)),
            'count': len(values_array),
        }
    
    return statistics


def save_results(
    statistics: Dict[str, Dict[str, Any]],
    output_dir: Path,
    dataset_name: str,
    model_info: Dict[str, Any],
):
    """Save evaluation results to files.
    
    Args:
        statistics: Computed statistics
        output_dir: Output directory
        dataset_name: Name of dataset
        model_info: Information about model and evaluation settings
    """
    # Create output directory
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Save statistics as JSON
    stats_file = output_dir / f"{dataset_name}_statistics.json"
    with open(stats_file, "w") as f:
        json.dump(statistics, f, indent=2)
    print(f"Statistics saved to {stats_file}")
    
    # Save summary as text
    summary_file = output_dir / f"{dataset_name}_summary.txt"
    with open(summary_file, "w") as f:
        f.write("DeepSFM Evaluation Summary\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("Model Information:\n")
        for key, value in model_info.items():
            f.write(f"  {key}: {value}\n")
        
        f.write("\n" + "=" * 50 + "\n\n")
        
        f.write("Depth Metrics:\n")
        depth_metrics = ['abs_rel', 'sq_rel', 'rmse', 'rmse_log', 'a1', 'a2', 'a3']
        for metric in depth_metrics:
            if metric in statistics:
                stats = statistics[metric]
                f.write(f"  {metric}:\n")
                f.write(f"    Mean: {stats['mean']:.6f}\n")
                f.write(f"    Std:  {stats['std']:.6f}\n")
                f.write(f"    Median: {stats['median']:.6f}\n")
                f.write(f"    Min-Max: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
                f.write(f"    Samples: {stats['count']}\n")
        
        f.write("\n" + "=" * 50 + "\n\n")
        
        f.write("Pose Metrics:\n")
        pose_metrics = ['rotation_error', 'translation_error', 'translation_angular_error']
        for metric in pose_metrics:
            if metric in statistics:
                stats = statistics[metric]
                f.write(f"  {metric}:\n")
                f.write(f"    Mean: {stats['mean']:.6f}\n")
                f.write(f"    Std:  {stats['std']:.6f}\n")
                f.write(f"    Median: {stats['median']:.6f}\n")
                f.write(f"    Min-Max: [{stats['min']:.6f}, {stats['max']:.6f}]\n")
                f.write(f"    Samples: {stats['count']}\n")
    
    print(f"Summary saved to {summary_file}")
    
    # Print summary to console
    print("\n" + "=" * 50)
    print("Evaluation Summary")
    print("=" * 50)
    
    print("\nDepth Metrics:")
    for metric in depth_metrics:
        if metric in statistics:
            stats = statistics[metric]
            print(f"  {metric}: {stats['mean']:.6f} ± {stats['std']:.6f} (median: {stats['median']:.6f})")
    
    print("\nPose Metrics:")
    for metric in pose_metrics:
        if metric in statistics:
            stats = statistics[metric]
            print(f"  {metric}: {stats['mean']:.6f} ± {stats['std']:.6f} (median: {stats['median']:.6f})")


def main():
    """Main evaluation function."""
    args = parse_args()
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Create dataset
    print(f"Creating {args.dataset} dataset...")
    dataset_cls = get_dataset(args.dataset)
    
    # Use validation or test split
    dataset = dataset_cls(
        data_root=args.data_root,
        split="test",  # or "val" depending on dataset
        image_size=tuple(config["data"]["image_size"]),
        num_source_views=config["data"]["num_source_views"],
        transforms=None,  # No transforms for evaluation
        load_gt_depth=True,
        load_gt_pose=True,
    )
    
    print(f"Dataset size: {len(dataset)}")
    
    # Create DeepSFM model
    print("Creating DeepSFM model...")
    depth_config = config.get("model", {})
    pose_config = config.get("model", {})
    
    model = DeepSFM(
        depth_config=depth_config,
        pose_config=pose_config,
        max_iterations=args.iterations,
        use_geometric_consistency=True,
    ).to(device)
    
    # Load checkpoints
    print(f"Loading depth checkpoint: {args.depth_checkpoint}")
    model.load_depth_checkpoint(args.depth_checkpoint)
    
    print(f"Loading pose checkpoint: {args.pose_checkpoint}")
    model.load_pose_checkpoint(args.pose_checkpoint)
    
    # Evaluate depth
    print("\nEvaluating depth estimation...")
    depth_metrics = evaluate_depth(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        iterations=args.iterations,
    )
    
    # Evaluate pose
    print("\nEvaluating pose estimation...")
    pose_metrics = evaluate_pose(
        model=model,
        dataset=dataset,
        device=device,
        batch_size=args.batch_size,
        num_samples=args.num_samples,
        iterations=args.iterations,
    )
    
    # Combine metrics
    all_metrics = {**depth_metrics, **pose_metrics}
    
    # Compute statistics
    statistics = compute_statistics(all_metrics)
    
    # Save results
    model_info = {
        'depth_checkpoint': args.depth_checkpoint,
        'pose_checkpoint': args.pose_checkpoint,
        'dataset': args.dataset,
        'data_root': args.data_root,
        'iterations': args.iterations,
        'num_samples': args.num_samples if args.num_samples else 'all',
        'batch_size': args.batch_size,
        'device': str(device),
    }
    
    save_results(
        statistics=statistics,
        output_dir=output_dir,
        dataset_name=args.dataset,
        model_info=model_info,
    )
    
    print(f"\nEvaluation complete! Results saved to {output_dir.absolute()}")


if __name__ == "__main__":
    main()