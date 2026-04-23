"""
DeMoN dataset loader for DeepSFM.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image
import torch

from .base import BaseDataset


class DeMoNDataset(BaseDataset):
    """DeMoN dataset loader."""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (384, 512),
        num_source_views: int = 2,
        min_depth: float = 0.5,
        max_depth: float = 10.0,
        transforms: Optional[Any] = None,
        load_gt_depth: bool = True,
        load_gt_pose: bool = True,
        load_initial_estimates: bool = False,
    ):
        """
        Args:
            data_root: Root directory of DeMoN dataset
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size (height, width)
            num_source_views: Number of source views to use
            min_depth: Minimum valid depth value
            max_depth: Maximum valid depth value
            transforms: Data transformations to apply
            load_gt_depth: Whether to load ground truth depth
            load_gt_pose: Whether to load ground truth poses
            load_initial_estimates: Whether to load initial depth/pose estimates from DeMoN
        """
        self.load_initial_estimates = load_initial_estimates
        
        super().__init__(
            data_root=data_root,
            split=split,
            image_size=image_size,
            num_source_views=num_source_views,
            min_depth=min_depth,
            max_depth=max_depth,
            transforms=transforms,
            load_gt_depth=load_gt_depth,
            load_gt_pose=load_gt_pose,
        )
    
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load DeMoN dataset samples.
        
        Expected directory structure:
            data_root/
                scene_00000/
                    0000.jpg, 0001.jpg, ...      # RGB images
                    0000.npy, 0001.npy, ...      # Ground truth depth maps
                    0000_demon.npy, ...          # DeMoN initial depth estimates (optional)
                    cam.txt                      # Camera intrinsics (3x3 matrix)
                    poses.txt                    # Ground truth poses (Nx12 flattened 3x4 matrices)
                    demon_poses.txt              # DeMoN initial pose estimates (optional)
                scene_00001/
                ...
        """
        samples = []
        
        # Get all scene directories
        scene_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        
        for scene_dir in scene_dirs:
            # Check if this is a valid scene directory
            if not scene_dir.name.startswith('scene_'):
                continue
            
            # Get all image files
            image_files = sorted(scene_dir.glob('*.jpg'))
            if len(image_files) < 2:
                continue  # Need at least 2 images for multi-view
            
            # Load camera intrinsics
            cam_file = scene_dir / 'cam.txt'
            if not cam_file.exists():
                continue
            
            intrinsics = np.loadtxt(cam_file, dtype=np.float32).reshape(3, 3)
            
            # Load poses if available
            poses_file = scene_dir / 'poses.txt'
            if poses_file.exists():
                poses_data = np.loadtxt(poses_file, dtype=np.float32)
                # Reshape to (N, 3, 4) then convert to (N, 4, 4) homogeneous matrices
                num_poses = poses_data.shape[0]
                poses = []
                for i in range(num_poses):
                    pose_3x4 = poses_data[i].reshape(3, 4)
                    pose_4x4 = np.eye(4, dtype=np.float32)
                    pose_4x4[:3, :] = pose_3x4
                    poses.append(pose_4x4)
            else:
                poses = None
            
            # Create sample
            sample = {
                'scene_id': scene_dir.name,
                'frame_ids': [int(f.stem) for f in image_files],
                'image_paths': [str(f) for f in image_files],
                'intrinsics': intrinsics,
                'poses': poses,  # List of 4x4 matrices or None
                'scene_dir': scene_dir,
            }
            
            samples.append(sample)
        
        # Split samples based on split type
        if self.split == "train":
            samples = samples[:int(0.8 * len(samples))]
        elif self.split == "val":
            samples = samples[int(0.8 * len(samples)):int(0.9 * len(samples))]
        else:  # test
            samples = samples[int(0.9 * len(samples)):]
        
        return samples
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image from file."""
        # Use PIL for consistent loading
        with Image.open(image_path) as img:
            # Convert to RGB if necessary
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img, dtype=np.float32)
    
    def _load_depth(self, depth_path: Path) -> Optional[np.ndarray]:
        """Load depth map from .npy file."""
        if not depth_path.exists():
            return None
        
        try:
            depth = np.load(depth_path)
            
            # Handle different depth formats
            if depth.ndim == 3:
                depth = depth.squeeze()
            
            # Ensure depth is 2D
            if depth.ndim != 2:
                raise ValueError(f"Invalid depth shape: {depth.shape}")
            
            return depth.astype(np.float32)
        except Exception as e:
            print(f"Error loading depth from {depth_path}: {e}")
            return None
    
    def _load_pose(self, pose_info: Any) -> Optional[np.ndarray]:
        """Load camera pose from pose array."""
        if pose_info is None:
            return None
        return pose_info.astype(np.float32)
    
    def _get_depth_path(self, sample: Dict[str, Any], frame_idx: int) -> Optional[Path]:
        """Get path to depth file for a frame."""
        scene_dir = sample['scene_dir']
        frame_id = sample['frame_ids'][frame_idx]
        
        # Try ground truth depth first
        depth_file = scene_dir / f"{frame_id:04d}.npy"
        if depth_file.exists():
            return depth_file
        
        # Try DeMoN initial depth estimate if requested
        if self.load_initial_estimates:
            demon_depth_file = scene_dir / f"{frame_id:04d}_demon.npy"
            if demon_depth_file.exists():
                return demon_depth_file
        
        return None
    
    def _get_pose_info(self, sample: Dict[str, Any], frame_idx: int) -> Optional[Any]:
        """Get pose information for a frame."""
        if sample['poses'] is not None and frame_idx < len(sample['poses']):
            return sample['poses'][frame_idx]
        
        # Try DeMoN initial pose estimate if requested
        if self.load_initial_estimates:
            demon_poses_file = sample['scene_dir'] / 'demon_poses.txt'
            if demon_poses_file.exists():
                try:
                    demon_poses_data = np.loadtxt(demon_poses_file, dtype=np.float32)
                    if frame_idx < demon_poses_data.shape[0]:
                        pose_3x4 = demon_poses_data[frame_idx].reshape(3, 4)
                        pose_4x4 = np.eye(4, dtype=np.float32)
                        pose_4x4[:3, :] = pose_3x4
                        return pose_4x4
                except Exception as e:
                    print(f"Error loading DeMoN pose from {demon_poses_file}: {e}")
        
        return None
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample with optional initial estimates."""
        sample = super().__getitem__(idx)
        
        # Add initial estimates if requested
        if self.load_initial_estimates:
            original_sample = self.samples[idx]
            ref_idx, src_idxs = self._select_frames(original_sample)
            
            # Load initial depth estimates
            init_ref_depth = self._load_depth(self._get_initial_depth_path(original_sample, ref_idx))
            init_src_depths = [self._load_depth(self._get_initial_depth_path(original_sample, i)) 
                              for i in src_idxs]
            
            # Load initial pose estimates
            init_ref_pose = self._load_pose(self._get_initial_pose_info(original_sample, ref_idx))
            init_src_poses = [self._load_pose(self._get_initial_pose_info(original_sample, i)) 
                             for i in src_idxs]
            
            # Convert to tensors if available
            if init_ref_depth is not None:
                init_ref_depth = torch.from_numpy(init_ref_depth).float().unsqueeze(0)
                if sample['ref_depth'] is None:  # Use initial as fallback
                    sample['ref_depth'] = init_ref_depth
            
            if init_ref_pose is not None and all(p is not None for p in init_src_poses):
                init_ref_pose = torch.from_numpy(init_ref_pose).float()
                init_src_poses = [torch.from_numpy(p).float() for p in init_src_poses]
                
                # Compute relative poses
                init_relative_poses = self._compute_relative_poses(
                    init_ref_pose.numpy(), [p.numpy() for p in init_src_poses]
                )
                init_relative_poses = torch.stack([torch.from_numpy(p).float() for p in init_relative_poses])
                
                sample['init_relative_poses'] = init_relative_poses
                if sample['relative_poses'] is None:  # Use initial as fallback
                    sample['relative_poses'] = init_relative_poses
            
            sample['init_ref_depth'] = init_ref_depth
            sample['init_src_depths'] = init_src_depths
        
        return sample
    
    def _get_initial_depth_path(self, sample: Dict[str, Any], frame_idx: int) -> Optional[Path]:
        """Get path to initial depth estimate from DeMoN."""
        scene_dir = sample['scene_dir']
        frame_id = sample['frame_ids'][frame_idx]
        
        demon_depth_file = scene_dir / f"{frame_id:04d}_demon.npy"
        if demon_depth_file.exists():
            return demon_depth_file
        
        return None
    
    def _get_initial_pose_info(self, sample: Dict[str, Any], frame_idx: int) -> Optional[Any]:
        """Get initial pose estimate from DeMoN."""
        demon_poses_file = sample['scene_dir'] / 'demon_poses.txt'
        if demon_poses_file.exists():
            try:
                demon_poses_data = np.loadtxt(demon_poses_file, dtype=np.float32)
                if frame_idx < demon_poses_data.shape[0]:
                    pose_3x4 = demon_poses_data[frame_idx].reshape(3, 4)
                    pose_4x4 = np.eye(4, dtype=np.float32)
                    pose_4x4[:3, :] = pose_3x4
                    return pose_4x4
            except Exception as e:
                print(f"Error loading DeMoN pose from {demon_poses_file}: {e}")
        
        return None


# Helper function to create dataset
def create_demon_dataset(
    data_root: str,
    split: str = "train",
    image_size: Tuple[int, int] = (384, 512),
    num_source_views: int = 2,
    use_augmentations: bool = True,
    load_initial_estimates: bool = False,
) -> DeMoNDataset:
    """Create DeMoN dataset with appropriate transformations."""
    from .transforms import create_train_transforms, create_val_transforms
    
    if split == "train" and use_augmentations:
        transforms = create_train_transforms(image_size)
    else:
        transforms = create_val_transforms(image_size)
    
    return DeMoNDataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
        num_source_views=num_source_views,
        transforms=transforms,
        load_initial_estimates=load_initial_estimates,
    )