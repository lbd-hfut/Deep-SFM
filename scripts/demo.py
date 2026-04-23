#!/usr/bin/env python3
"""
Demo script for DeepSFM.
Shows how to use the DeepSFM model for depth and pose estimation.
"""

import os
import sys
import torch
import numpy as np
from PIL import Image
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from models.deepsfm import DeepSFM
from models.psnet import PSNet
from models.posenet import PoseNet


def load_sample_images():
    """Load sample images for demonstration.
    Returns placeholder images if no real images are available.
    """
    # Create synthetic images for demonstration
    height, width = 384, 512
    batch_size = 1
    
    # Reference image (simple gradient)
    ref_img = torch.zeros(batch_size, 3, height, width)
    for c in range(3):
        for h in range(height):
            ref_img[0, c, h, :] = torch.linspace(0, 1, width) * (c + 1) / 3
    
    # Source image (slightly different)
    src_img = ref_img.clone()
    src_img[0, :, :, :width//2] = 0.5  # Darken left half
    
    return ref_img, [src_img]


def create_sample_intrinsics(height=384, width=512):
    """Create sample camera intrinsics."""
    focal = max(height, width) * 1.2
    intrinsics = torch.tensor([[
        [focal, 0, width/2],
        [0, focal, height/2],
        [0, 0, 1]
    ]])
    return intrinsics


def main():
    """Main demo function."""
    print("DeepSFM Demo")
    print("=" * 50)
    
    # Set device
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # Create sample data
    print("\n1. Creating sample data...")
    ref_img, src_imgs = load_sample_images()
    intrinsics = create_sample_intrinsics()
    
    # Move to device
    ref_img = ref_img.to(device)
    src_imgs = [img.to(device) for img in src_imgs]
    intrinsics = intrinsics.to(device)
    
    print(f"   Reference image shape: {ref_img.shape}")
    print(f"   Number of source images: {len(src_imgs)}")
    print(f"   Intrinsics shape: {intrinsics.shape}")
    
    # Create DeepSFM model
    print("\n2. Creating DeepSFM model...")
    model = DeepSFM(
        depth_config={
            'min_depth': 0.5,
            'max_depth': 10.0,
            'n_depth_bins': 64,
            'feature_channels': 32,
        },
        pose_config={
            'translation_std': 0.27,
            'rotation_std': 0.12,
            'n_pose_bins': 10,
            'feature_channels': 32,
        },
        max_iterations=3,
        use_geometric_consistency=True,
    ).to(device)
    
    print(f"   Depth subnet: {model.depth_net.__class__.__name__}")
    print(f"   Pose subnet: {model.pose_net.__class__.__name__}")
    print(f"   Max iterations: {model.max_iterations}")
    
    # Run inference
    print("\n3. Running inference...")
    model.eval()
    
    with torch.no_grad():
        output = model(
            ref_img=ref_img,
            src_imgs=src_imgs,
            intrinsics=intrinsics,
            init_depth=None,
            init_poses=None,
            num_iterations=2,
            return_all_iterations=True,
        )
    
    # Extract results
    final_depth = output['depth'].cpu().numpy()[0, 0]  # (H, W)
    final_poses = output['poses'].cpu().numpy()[0]  # (N, 4, 4)
    
    print(f"\n4. Results:")
    print(f"   Depth map shape: {final_depth.shape}")
    print(f"   Depth range: [{final_depth.min():.3f}, {final_depth.max():.3f}]")
    print(f"   Depth mean: {final_depth.mean():.3f}")
    print(f"   Number of poses: {len(final_poses)}")
    
    if 'depth_history' in output:
        print(f"   Number of iterations stored: {len(output['depth_history'])}")
    
    # Show pose information
    print("\n5. Estimated relative poses:")
    for i, pose in enumerate(final_poses):
        rotation = pose[:3, :3]
        translation = pose[:3, 3]
        
        # Compute rotation angles
        from scipy.spatial.transform import Rotation as R
        try:
            r = R.from_matrix(rotation)
            angles = r.as_euler('xyz', degrees=True)
            print(f"   Source {i+1}:")
            print(f"     Translation: [{translation[0]:.3f}, {translation[1]:.3f}, {translation[2]:.3f}]")
            print(f"     Rotation (degrees): [{angles[0]:.2f}, {angles[1]:.2f}, {angles[2]:.2f}]")
        except ImportError:
            print(f"   Source {i+1}: Translation: {translation}, Rotation matrix shape: {rotation.shape}")
    
    # Demonstrate geometric consistency
    print("\n6. Computing geometric consistency...")
    if len(src_imgs) > 0:
        # For demo, create a dummy source depth
        src_depth = torch.ones_like(output['depth']) * 2.0
        
        consistency = model.compute_geometric_consistency(
            depth=output['depth'],
            src_depth=src_depth,
            intrinsics=intrinsics,
            relative_pose=output['poses'][:, 0],  # First source view
        )
        
        consistency_mean = consistency.mean().item()
        print(f"   Geometric consistency error: {consistency_mean:.6f}")
    
    # Demonstrate model methods
    print("\n7. Model utilities:")
    
    # Freeze/unfreeze example
    print("   Freezing depth subnet...")
    model.freeze_depth_net()
    depth_params = sum(p.requires_grad for p in model.depth_net.parameters())
    print(f"   Trainable depth parameters: {depth_params}")
    
    print("   Unfreezing all...")
    model.unfreeze_all()
    depth_params = sum(p.requires_grad for p in model.depth_net.parameters())
    pose_params = sum(p.requires_grad for p in model.pose_net.parameters())
    print(f"   Trainable depth parameters: {depth_params}")
    print(f"   Trainable pose parameters: {pose_params}")
    
    # Save/load example (conceptual)
    print("\n8. Checkpoint handling:")
    print("   Model supports:")
    print("     - load_depth_checkpoint(path)")
    print("     - load_pose_checkpoint(path)")
    print("     - load_checkpoints(depth_path, pose_path)")
    
    print("\n" + "=" * 50)
    print("Demo complete!")
    print("\nNext steps:")
    print("1. Train depth subnet: python training/train_depth.py --config configs/depth_default.yaml")
    print("2. Train pose subnet: python training/train_pose.py --config configs/pose_default.yaml")
    print("3. Run iterative inference: python scripts/iterative_inference.py --input-dir <your_images>")
    print("4. Evaluate: python evaluation/evaluate.py --config configs/depth_default.yaml")


if __name__ == "__main__":
    main()