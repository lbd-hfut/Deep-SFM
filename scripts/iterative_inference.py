#!/usr/bin/env python3
"""
Iterative inference script for DeepSFM.
Alternates between depth and pose refinement to perform bundle adjustment.
"""

import os
import sys
import argparse
import yaml
import torch
import numpy as np
from PIL import Image
from pathlib import Path
from typing import List, Dict, Any, Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.deepsfm import DeepSFM
from models.psnet import PSNet
from models.posenet import PoseNet
from datasets import get_dataset
from datasets.transforms import create_val_transforms


def load_images(image_paths: List[Path], image_size: tuple) -> List[torch.Tensor]:
    """Load and preprocess images."""
    images = []
    for img_path in image_paths:
        img = Image.open(img_path).convert('RGB')
        img = img.resize((image_size[1], image_size[0]), Image.BILINEAR)
        img = np.array(img).astype(np.float32) / 255.0
        img = torch.from_numpy(img).permute(2, 0, 1).unsqueeze(0)  # (1, 3, H, W)
        images.append(img)
    return images


def load_intrinsics_from_file(intrinsics_path: Path, image_size: tuple) -> torch.Tensor:
    """Load camera intrinsics from file.
    
    Supports:
        - TXT file with 3x3 matrix
        - JSON/YAML with 'intrinsics' key
        - Default intrinsics based on image size
    """
    if not intrinsics_path.exists():
        # Create default intrinsics (assuming typical focal length)
        height, width = image_size
        focal_length = max(height, width) * 1.2
        intrinsics = torch.tensor([
            [focal_length, 0, width / 2],
            [0, focal_length, height / 2],
            [0, 0, 1]
        ], dtype=torch.float32).unsqueeze(0)  # (1, 3, 3)
        return intrinsics
    
    # Try to load from file
    suffix = intrinsics_path.suffix.lower()
    if suffix == '.txt':
        matrix = np.loadtxt(intrinsics_path)
        if matrix.shape == (3, 3):
            intrinsics = torch.from_numpy(matrix).float().unsqueeze(0)
            return intrinsics
    
    # Default intrinsics
    height, width = image_size
    focal_length = max(height, width) * 1.2
    intrinsics = torch.tensor([
        [focal_length, 0, width / 2],
        [0, focal_length, height / 2],
        [0, 0, 1]
    ], dtype=torch.float32).unsqueeze(0)
    
    return intrinsics


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Iterative inference for DeepSFM bundle adjustment"
    )
    parser.add_argument(
        "--input-dir",
        type=str,
        required=True,
        help="Directory containing input images"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="output",
        help="Directory to save results"
    )
    parser.add_argument(
        "--depth-checkpoint",
        type=str,
        required=True,
        help="Path to pretrained depth subnet checkpoint"
    )
    parser.add_argument(
        "--pose-checkpoint",
        type=str,
        required=True,
        help="Path to pretrained pose subnet checkpoint"
    )
    parser.add_argument(
        "--config",
        type=str,
        default="configs/depth_default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of alternating iterations"
    )
    parser.add_argument(
        "--image-size",
        type=int,
        nargs=2,
        default=[384, 512],
        help="Image size (height width)"
    )
    parser.add_argument(
        "--visualize",
        action="store_true",
        help="Generate visualization images"
    )
    parser.add_argument(
        "--save-npy",
        action="store_true",
        help="Save results as numpy files"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use"
    )
    
    return parser.parse_args()


def main():
    """Main inference function."""
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
    
    # Create DeepSFM model
    print("Creating DeepSFM model...")
    depth_config = config.get("model", {})
    pose_config = config.get("model", {})  # Can have separate pose config
    
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
    
    model.eval()
    
    # Find input images
    input_dir = Path(args.input_dir)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.tiff']
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(input_dir.glob(f"*{ext}"))
        image_paths.extend(input_dir.glob(f"*{ext.upper()}"))
    
    image_paths = sorted(image_paths)
    
    if len(image_paths) < 2:
        print(f"Need at least 2 images, found {len(image_paths)}")
        return
    
    print(f"Found {len(image_paths)} images")
    
    # Designate first image as reference, others as source
    ref_image_path = image_paths[0]
    src_image_paths = image_paths[1:]
    
    # Load and preprocess images
    print("Loading images...")
    image_size = tuple(args.image_size)
    
    ref_image = load_images([ref_image_path], image_size)[0].to(device)
    src_images = load_images(src_image_paths, image_size)
    src_images = [img.to(device) for img in src_images]
    
    # Load or create intrinsics
    intrinsics_path = input_dir / "intrinsics.txt"
    intrinsics = load_intrinsics_from_file(intrinsics_path, image_size).to(device)
    
    # Repeat intrinsics for batch dimension
    batch_size = 1
    intrinsics = intrinsics.repeat(batch_size, 1, 1)
    
    print(f"Reference image: {ref_image_path.name}")
    print(f"Source images: {[p.name for p in src_image_paths]}")
    print(f"Image size: {image_size}")
    print(f"Intrinsics:\n{intrinsics[0]}")
    
    # Run iterative inference
    print(f"\nRunning iterative inference ({args.iterations} iterations)...")
    with torch.no_grad():
        output = model(
            ref_img=ref_image,
            src_imgs=src_images,
            intrinsics=intrinsics,
            init_depth=None,
            init_poses=None,
            num_iterations=args.iterations,
            return_all_iterations=True,
        )
    
    # Extract results
    final_depth = output['depth'].cpu().numpy()[0, 0]  # (H, W)
    final_poses = output['poses'].cpu().numpy()[0]  # (N, 4, 4)
    
    print("\nInference complete!")
    print(f"Depth range: [{final_depth.min():.3f}, {final_depth.max():.3f}]")
    print(f"Depth mean: {final_depth.mean():.3f}")
    
    # Save results
    print("\nSaving results...")
    
    # Save depth map
    depth_output_path = output_dir / "depth.npy"
    np.save(depth_output_path, final_depth)
    print(f"Depth map saved to {depth_output_path}")
    
    # Save poses
    poses_output_path = output_dir / "poses.npy"
    np.save(poses_output_path, final_poses)
    print(f"Poses saved to {poses_output_path}")
    
    # Save depth as image (normalized for visualization)
    depth_normalized = (final_depth - final_depth.min()) / (final_depth.max() - final_depth.min() + 1e-7)
    depth_image = (depth_normalized * 255).astype(np.uint8)
    depth_img_path = output_dir / "depth.png"
    Image.fromarray(depth_image).save(depth_img_path)
    print(f"Depth visualization saved to {depth_img_path}")
    
    # Save iteration history if available
    if 'depth_history' in output:
        history_dir = output_dir / "history"
        history_dir.mkdir(exist_ok=True)
        
        depth_history = output['depth_history']
        for i, depth_iter in enumerate(depth_history):
            depth_np = depth_iter.cpu().numpy()[0, 0]
            np.save(history_dir / f"depth_iter_{i:02d}.npy", depth_np)
        
        print(f"Iteration history saved to {history_dir}")
    
    # Generate report
    report_path = output_dir / "report.txt"
    with open(report_path, "w") as f:
        f.write("DeepSFM Iterative Inference Report\n")
        f.write("=" * 40 + "\n\n")
        f.write(f"Input directory: {input_dir}\n")
        f.write(f"Output directory: {output_dir}\n")
        f.write(f"Number of images: {len(image_paths)}\n")
        f.write(f"Reference image: {ref_image_path.name}\n")
        f.write(f"Source images: {', '.join([p.name for p in src_image_paths])}\n")
        f.write(f"Image size: {image_size}\n")
        f.write(f"Iterations: {args.iterations}\n\n")
        
        f.write("Depth Statistics:\n")
        f.write(f"  Min: {final_depth.min():.6f}\n")
        f.write(f"  Max: {final_depth.max():.6f}\n")
        f.write(f"  Mean: {final_depth.mean():.6f}\n")
        f.write(f"  Std: {final_depth.std():.6f}\n\n")
        
        f.write("Relative Poses:\n")
        for i, pose in enumerate(final_poses):
            f.write(f"  Source {i+1}:\n")
            f.write(f"    Rotation:\n")
            f.write(f"      {pose[:3, :3][0]}\n")
            f.write(f"      {pose[:3, :3][1]}\n")
            f.write(f"      {pose[:3, :3][2]}\n")
            f.write(f"    Translation: {pose[:3, 3]}\n")
    
    print(f"Report saved to {report_path}")
    
    # Visualize results if requested
    if args.visualize:
        print("\nGenerating visualizations...")
        visualize_results(
            ref_image_path=ref_image_path,
            src_image_paths=src_image_paths,
            depth_map=final_depth,
            poses=final_poses,
            output_dir=output_dir,
            image_size=image_size,
        )
    
    print(f"\nAll results saved to: {output_dir.absolute()}")


def visualize_results(
    ref_image_path: Path,
    src_image_paths: List[Path],
    depth_map: np.ndarray,
    poses: np.ndarray,
    output_dir: Path,
    image_size: tuple,
):
    """Generate visualization images."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        print("Matplotlib not installed, skipping visualizations")
        return
    
    # Create visualization directory
    viz_dir = output_dir / "visualizations"
    viz_dir.mkdir(exist_ok=True)
    
    # Load reference image
    ref_img = Image.open(ref_image_path).convert('RGB')
    ref_img = ref_img.resize((image_size[1], image_size[0]), Image.BILINEAR)
    ref_img_np = np.array(ref_img)
    
    # Plot depth map
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.imshow(ref_img_np)
    plt.title("Reference Image")
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    # Use viridis colormap for depth
    depth_normalized = (depth_map - depth_map.min()) / (depth_map.max() - depth_map.min() + 1e-7)
    plt.imshow(depth_normalized, cmap='viridis')
    plt.title("Depth Map")
    plt.colorbar(label='Depth (normalized)')
    plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(viz_dir / "depth_result.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot depth map overlay
    plt.figure(figsize=(8, 6))
    plt.imshow(ref_img_np, alpha=0.7)
    plt.imshow(depth_normalized, cmap='viridis', alpha=0.5)
    plt.title("Depth Overlay")
    plt.colorbar(label='Depth (normalized)')
    plt.axis('off')
    plt.savefig(viz_dir / "depth_overlay.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # Plot pose visualization (simple 3D plot if open3d available)
    try:
        import open3d as o3d
        
        # Create simple point cloud from depth
        height, width = depth_map.shape
        fx = fy = max(height, width) * 1.2
        cx, cy = width / 2, height / 2
        
        # Sample points for efficiency
        step = 10
        points = []
        colors = []
        
        for y in range(0, height, step):
            for x in range(0, width, step):
                depth = depth_map[y, x]
                if depth > 0:
                    # Back-project to 3D
                    z = depth
                    x_3d = (x - cx) * z / fx
                    y_3d = (y - cy) * z / fy
                    points.append([x_3d, y_3d, z])
                    
                    # Get color from image
                    color = ref_img_np[y, x] / 255.0
                    colors.append(color)
        
        if points:
            pcd = o3d.geometry.PointCloud()
            pcd.points = o3d.utility.Vector3dVector(points)
            pcd.colors = o3d.utility.Vector3dVector(colors)
            
            # Save point cloud
            o3d.io.write_point_cloud(str(viz_dir / "point_cloud.ply"), pcd)
            print(f"Point cloud saved to {viz_dir / 'point_cloud.ply'}")
    except ImportError:
        print("Open3D not installed, skipping 3D visualization")
    
    print(f"Visualizations saved to {viz_dir}")


if __name__ == "__main__":
    main()