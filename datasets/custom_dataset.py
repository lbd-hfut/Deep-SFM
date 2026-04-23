"""
Custom dataset loader for DeepSFM.
Allows loading user-provided data in a simple format.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import yaml
from PIL import Image

from .base import BaseDataset


class CustomDataset(BaseDataset):
    """Custom dataset loader for user-provided data."""
    
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
        config_file: Optional[str] = None,
    ):
        """
        Args:
            data_root: Root directory of custom dataset
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size (height, width)
            num_source_views: Number of source views to use
            min_depth: Minimum valid depth value
            max_depth: Maximum valid depth value
            transforms: Data transformations to apply
            load_gt_depth: Whether to load ground truth depth
            load_gt_pose: Whether to load ground truth poses
            config_file: Optional YAML config file describing dataset structure
        """
        self.config_file = config_file
        
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
        """Load custom dataset samples.
        
        Supports multiple formats:
        1. Simple format (default):
            data_root/
                scene_001/
                    images/
                        0000.jpg, 0001.jpg, ...
                    depths/ (optional)
                        0000.npy, 0001.npy, ...
                    poses.txt (optional)  # Nx12 flattened 3x4 matrices
                    intrinsics.txt  # 3x3 matrix
                scene_002/
                ...
        
        2. Config-based format (with config_file):
            Uses YAML config to specify custom structure.
        """
        samples = []
        
        if self.config_file is not None:
            # Load samples using config file
            samples = self._load_samples_from_config()
        else:
            # Load samples using simple format
            samples = self._load_samples_simple()
        
        return samples
    
    def _load_samples_simple(self) -> List[Dict[str, Any]]:
        """Load samples using simple directory structure."""
        samples = []
        
        # Get all scene directories
        scene_dirs = sorted([d for d in self.data_root.iterdir() if d.is_dir()])
        
        for scene_dir in scene_dirs:
            # Check for images directory
            images_dir = scene_dir / "images"
            if not images_dir.exists():
                continue
            
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(sorted(images_dir.glob(f'*{ext}')))
            
            if len(image_files) < 2:
                continue  # Need at least 2 images
            
            # Load intrinsics
            intrinsics_file = scene_dir / "intrinsics.txt"
            if intrinsics_file.exists():
                intrinsics = np.loadtxt(intrinsics_file, dtype=np.float32).reshape(3, 3)
            else:
                # Use default intrinsics (assume similar to DeMoN)
                intrinsics = np.array([
                    [600.0, 0, 320.0],
                    [0, 600.0, 240.0],
                    [0, 0, 1]
                ], dtype=np.float32)
            
            # Load poses if available
            poses_file = scene_dir / "poses.txt"
            if poses_file.exists() and self.load_gt_pose:
                poses_data = np.loadtxt(poses_file, dtype=np.float32)
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
                'poses': poses,
                'scene_dir': scene_dir,
                'images_dir': images_dir,
            }
            
            samples.append(sample)
        
        return samples
    
    def _load_samples_from_config(self) -> List[Dict[str, Any]]:
        """Load samples using YAML config file."""
        with open(self.config_file, 'r') as f:
            config = yaml.safe_load(f)
        
        samples = []
        
        # Get dataset structure from config
        dataset_format = config.get('format', 'simple')
        scenes = config.get('scenes', [])
        
        for scene_info in scenes:
            scene_id = scene_info.get('id', 'unknown')
            scene_path = self.data_root / scene_info['path']
            
            # Load scene data based on format
            if dataset_format == 'simple':
                sample = self._load_simple_scene(scene_path, scene_id, scene_info)
            elif dataset_format == 'colmap':
                sample = self._load_colmap_scene(scene_path, scene_id, scene_info)
            else:
                raise ValueError(f"Unknown dataset format: {dataset_format}")
            
            if sample is not None:
                samples.append(sample)
        
        return samples
    
    def _load_simple_scene(self, scene_path: Path, scene_id: str, scene_info: Dict) -> Optional[Dict[str, Any]]:
        """Load scene in simple format."""
        # Get image files
        image_pattern = scene_info.get('image_pattern', '*.jpg')
        image_files = sorted(scene_path.glob(image_pattern))
        
        if len(image_files) < 2:
            return None
        
        # Load intrinsics
        intrinsics_file = scene_path / scene_info.get('intrinsics_file', 'intrinsics.txt')
        if intrinsics_file.exists():
            intrinsics = np.loadtxt(intrinsics_file, dtype=np.float32).reshape(3, 3)
        else:
            # Try to get intrinsics from config
            intrinsics_params = scene_info.get('intrinsics', {})
            if intrinsics_params:
                fx = intrinsics_params.get('fx', 600.0)
                fy = intrinsics_params.get('fy', 600.0)
                cx = intrinsics_params.get('cx', 320.0)
                cy = intrinsics_params.get('cy', 240.0)
                intrinsics = np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)
            else:
                # Default intrinsics
                intrinsics = np.array([
                    [600.0, 0, 320.0],
                    [0, 600.0, 240.0],
                    [0, 0, 1]
                ], dtype=np.float32)
        
        # Load poses if available
        poses = None
        if self.load_gt_pose:
            poses_file = scene_path / scene_info.get('poses_file', 'poses.txt')
            if poses_file.exists():
                poses_data = np.loadtxt(poses_file, dtype=np.float32)
                num_poses = poses_data.shape[0]
                poses = []
                for i in range(num_poses):
                    pose_3x4 = poses_data[i].reshape(3, 4)
                    pose_4x4 = np.eye(4, dtype=np.float32)
                    pose_4x4[:3, :] = pose_3x4
                    poses.append(pose_4x4)
        
        return {
            'scene_id': scene_id,
            'frame_ids': list(range(len(image_files))),
            'image_paths': [str(f) for f in image_files],
            'intrinsics': intrinsics,
            'poses': poses,
            'scene_dir': scene_path,
        }
    
    def _load_colmap_scene(self, scene_path: Path, scene_id: str, scene_info: Dict) -> Optional[Dict[str, Any]]:
        """Load scene in COLMAP format."""
        # This is a placeholder - COLMAP format would need more complex parsing
        # For now, fall back to simple format
        return self._load_simple_scene(scene_path, scene_id, scene_info)
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image from file."""
        with Image.open(image_path) as img:
            if img.mode != 'RGB':
                img = img.convert('RGB')
            return np.array(img, dtype=np.float32)
    
    def _load_depth(self, depth_path: Path) -> Optional[np.ndarray]:
        """Load depth map from file."""
        if not depth_path.exists():
            return None
        
        try:
            # Try different depth formats
            if depth_path.suffix == '.npy':
                depth = np.load(depth_path)
            elif depth_path.suffix == '.png':
                # Assume 16-bit PNG depth
                depth_img = Image.open(depth_path)
                depth = np.array(depth_img, dtype=np.float32)
                depth = depth / 256.0  # Convert to meters
            else:
                # Try loading as text file
                depth = np.loadtxt(depth_path, dtype=np.float32)
            
            # Handle different depth formats
            if depth.ndim == 3:
                depth = depth.squeeze()
            
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
        
        # Try different depth file locations
        depth_dirs = [
            scene_dir / "depths",
            scene_dir / "depth",
            scene_dir / "depth_maps",
        ]
        
        depth_extensions = ['.npy', '.png', '.txt', '.exr']
        
        for depth_dir in depth_dirs:
            if depth_dir.exists():
                for ext in depth_extensions:
                    depth_file = depth_dir / f"{frame_id:04d}{ext}"
                    if depth_file.exists():
                        return depth_file
        
        return None
    
    def _get_pose_info(self, sample: Dict[str, Any], frame_idx: int) -> Optional[Any]:
        """Get pose information for a frame."""
        if sample['poses'] is not None and frame_idx < len(sample['poses']):
            return sample['poses'][frame_idx]
        return None


# Helper function to create dataset
def create_custom_dataset(
    data_root: str,
    split: str = "train",
    image_size: Tuple[int, int] = (384, 512),
    num_source_views: int = 2,
    use_augmentations: bool = True,
    config_file: Optional[str] = None,
) -> CustomDataset:
    """Create custom dataset with appropriate transformations."""
    from .transforms import create_train_transforms, create_val_transforms
    
    if split == "train" and use_augmentations:
        transforms = create_train_transforms(image_size)
    else:
        transforms = create_val_transforms(image_size)
    
    return CustomDataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
        num_source_views=num_source_views,
        transforms=transforms,
        config_file=config_file,
    )