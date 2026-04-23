"""
Base dataset class for DeepSFM.
"""

import abc
from typing import Dict, List, Tuple, Optional, Any
from pathlib import Path

import torch
from torch.utils.data import Dataset
import numpy as np


class BaseDataset(Dataset, abc.ABC):
    """Base class for all DeepSFM datasets."""
    
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
    ):
        """
        Args:
            data_root: Root directory of the dataset
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size (height, width)
            num_source_views: Number of source views to use
            min_depth: Minimum valid depth value
            max_depth: Maximum valid depth value
            transforms: Data transformations to apply
            load_gt_depth: Whether to load ground truth depth
            load_gt_pose: Whether to load ground truth poses
        """
        self.data_root = Path(data_root)
        self.split = split
        self.image_size = image_size
        self.num_source_views = num_source_views
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.transforms = transforms
        self.load_gt_depth = load_gt_depth
        self.load_gt_pose = load_gt_pose
        
        # Validate split
        assert split in ["train", "val", "test"], f"Invalid split: {split}"
        
        # Load dataset samples
        self.samples = self._load_samples()
        
        # Validate samples
        self._validate_samples()
    
    @abc.abstractmethod
    def _load_samples(self) -> List[Dict[str, Any]]:
        """Load dataset samples. Must be implemented by subclasses.
        
        Returns:
            List of sample dictionaries, each containing at least:
                - 'scene_id': Scene identifier
                - 'frame_ids': List of frame IDs in the sequence
                - 'image_paths': List of image file paths
                - 'intrinsics': Camera intrinsics matrix (3x3)
        """
        pass
    
    @abc.abstractmethod
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image from file. Must be implemented by subclasses.
        
        Args:
            image_path: Path to image file
            
        Returns:
            Image as numpy array (H, W, 3) in RGB format
        """
        pass
    
    @abc.abstractmethod
    def _load_depth(self, depth_path: Path) -> Optional[np.ndarray]:
        """Load depth map from file. Must be implemented by subclasses.
        
        Args:
            depth_path: Path to depth file
            
        Returns:
            Depth map as numpy array (H, W) or None if not available
        """
        pass
    
    @abc.abstractmethod
    def _load_pose(self, pose_info: Any) -> Optional[np.ndarray]:
        """Load camera pose from information. Must be implemented by subclasses.
        
        Args:
            pose_info: Pose information (could be path, index, etc.)
            
        Returns:
            Camera pose as 4x4 transformation matrix or None if not available
        """
        pass
    
    @abc.abstractmethod
    def _get_depth_path(self, sample: Dict[str, Any], frame_idx: int) -> Optional[Path]:
        """Get path to depth file for a frame. Must be implemented by subclasses.
        
        Args:
            sample: Sample dictionary
            frame_idx: Index of frame in the sequence
            
        Returns:
            Path to depth file or None if not available
        """
        pass
    
    @abc.abstractmethod
    def _get_pose_info(self, sample: Dict[str, Any], frame_idx: int) -> Optional[Any]:
        """Get pose information for a frame. Must be implemented by subclasses.
        
        Args:
            sample: Sample dictionary
            frame_idx: Index of frame in the sequence
            
        Returns:
            Pose information or None if not available
        """
        pass
    
    def _validate_samples(self):
        """Validate loaded samples."""
        assert len(self.samples) > 0, "No samples loaded"
        
        for sample in self.samples:
            assert 'scene_id' in sample, "Sample missing 'scene_id'"
            assert 'frame_ids' in sample, "Sample missing 'frame_ids'"
            assert 'image_paths' in sample, "Sample missing 'image_paths'"
            assert 'intrinsics' in sample, "Sample missing 'intrinsics'"
            
            # Check that all image files exist
            for img_path in sample['image_paths']:
                assert Path(img_path).exists(), f"Image file not found: {img_path}"
            
            # Check intrinsics matrix shape
            intrinsics = sample['intrinsics']
            assert intrinsics.shape == (3, 3), f"Invalid intrinsics shape: {intrinsics.shape}"
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Dict[str, Any]:
        """Get a sample from the dataset.
        
        Returns a dictionary containing:
            - 'ref_image': Reference image tensor (3, H, W)
            - 'src_images': List of source image tensors (N, 3, H, W)
            - 'intrinsics': Camera intrinsics tensor (3, 3)
            - 'ref_depth': Reference depth tensor (1, H, W) or None
            - 'src_depths': List of source depth tensors or None
            - 'ref_pose': Reference camera pose tensor (4, 4) or None
            - 'src_poses': List of source camera poses or None
            - 'relative_poses': Relative poses from ref to src (N, 4, 4) or None
            - 'scene_id': Scene identifier
            - 'frame_ids': Frame IDs used
            - 'valid_depth_mask': Mask of valid depth pixels
        """
        sample = self.samples[idx]
        
        # Select reference frame and source frames
        ref_idx, src_idxs = self._select_frames(sample)
        
        # Load images
        ref_image = self._load_image(sample['image_paths'][ref_idx])
        src_images = [self._load_image(sample['image_paths'][i]) for i in src_idxs]
        
        # Load depths if requested
        if self.load_gt_depth:
            ref_depth = self._load_depth(self._get_depth_path(sample, ref_idx))
            src_depths = [self._load_depth(self._get_depth_path(sample, i)) for i in src_idxs]
        else:
            ref_depth = None
            src_depths = None
        
        # Load poses if requested
        if self.load_gt_pose:
            ref_pose = self._load_pose(self._get_pose_info(sample, ref_idx))
            src_poses = [self._load_pose(self._get_pose_info(sample, i)) for i in src_idxs]
            
            # Compute relative poses
            if ref_pose is not None and all(p is not None for p in src_poses):
                relative_poses = self._compute_relative_poses(ref_pose, src_poses)
            else:
                relative_poses = None
        else:
            ref_pose = None
            src_poses = None
            relative_poses = None
        
        # Get intrinsics
        intrinsics = sample['intrinsics'].copy()
        
        # Apply transformations
        if self.transforms is not None:
            transform_inputs = {
                'ref_image': ref_image,
                'src_images': src_images,
                'intrinsics': intrinsics,
                'ref_depth': ref_depth,
                'src_depths': src_depths,
                'ref_pose': ref_pose,
                'src_poses': src_poses,
                'relative_poses': relative_poses,
            }
            
            transformed = self.transforms(transform_inputs)
            
            ref_image = transformed['ref_image']
            src_images = transformed['src_images']
            intrinsics = transformed['intrinsics']
            ref_depth = transformed['ref_depth']
            src_depths = transformed['src_depths']
            relative_poses = transformed['relative_poses']
        
        # Convert to tensors if not already
        if not isinstance(ref_image, torch.Tensor):
            ref_image = torch.from_numpy(ref_image).float()
        if not isinstance(src_images[0], torch.Tensor):
            src_images = [torch.from_numpy(img).float() for img in src_images]
        if not isinstance(intrinsics, torch.Tensor):
            intrinsics = torch.from_numpy(intrinsics).float()
        
        # Stack source images
        src_images = torch.stack(src_images, dim=0)  # (N, 3, H, W)
        
        # Prepare depth tensors
        if ref_depth is not None:
            if not isinstance(ref_depth, torch.Tensor):
                ref_depth = torch.from_numpy(ref_depth).float()
            ref_depth = ref_depth.unsqueeze(0)  # (1, H, W)
            
            # Create valid depth mask
            valid_depth_mask = (ref_depth > self.min_depth) & (ref_depth < self.max_depth)
            ref_depth = torch.clamp(ref_depth, self.min_depth, self.max_depth)
        else:
            valid_depth_mask = None
        
        if src_depths is not None and all(d is not None for d in src_depths):
            src_depths = [torch.from_numpy(d).float().unsqueeze(0) if not isinstance(d, torch.Tensor) else d 
                         for d in src_depths]
            src_depths = torch.stack(src_depths, dim=0)  # (N, 1, H, W)
        else:
            src_depths = None
        
        # Prepare relative poses tensor
        if relative_poses is not None:
            if not isinstance(relative_poses[0], torch.Tensor):
                relative_poses = [torch.from_numpy(p).float() for p in relative_poses]
            relative_poses = torch.stack(relative_poses, dim=0)  # (N, 4, 4)
        
        # Prepare output dictionary
        output = {
            'ref_image': ref_image,
            'src_images': src_images,
            'intrinsics': intrinsics,
            'ref_depth': ref_depth,
            'src_depths': src_depths,
            'relative_poses': relative_poses,
            'scene_id': sample['scene_id'],
            'frame_ids': [sample['frame_ids'][ref_idx]] + [sample['frame_ids'][i] for i in src_idxs],
            'valid_depth_mask': valid_depth_mask,
        }
        
        return output
    
    def _select_frames(self, sample: Dict[str, Any]) -> Tuple[int, List[int]]:
        """Select reference frame and source frames from a sequence.
        
        Args:
            sample: Sample dictionary
            
        Returns:
            Tuple of (reference_index, list_of_source_indices)
        """
        num_frames = len(sample['frame_ids'])
        
        # For training, randomly select frames
        if self.split == "train":
            # Select random reference frame
            ref_idx = np.random.randint(0, num_frames)
            
            # Select source frames (avoid selecting the same frame)
            available_idxs = [i for i in range(num_frames) if i != ref_idx]
            num_to_select = min(self.num_source_views, len(available_idxs))
            src_idxs = np.random.choice(available_idxs, size=num_to_select, replace=False).tolist()
        
        # For validation/test, use fixed pattern
        else:
            # Use middle frame as reference
            ref_idx = num_frames // 2
            
            # Select evenly spaced source frames
            step = max(1, num_frames // (self.num_source_views + 1))
            src_idxs = []
            for i in range(self.num_source_views):
                idx = (ref_idx + (i + 1) * step) % num_frames
                if idx != ref_idx:
                    src_idxs.append(idx)
            
            # If not enough source frames, use closest frames
            if len(src_idxs) < self.num_source_views:
                all_idxs = [i for i in range(num_frames) if i != ref_idx]
                src_idxs = all_idxs[:self.num_source_views]
        
        return ref_idx, src_idxs
    
    def _compute_relative_poses(
        self,
        ref_pose: np.ndarray,
        src_poses: List[np.ndarray],
    ) -> List[np.ndarray]:
        """Compute relative poses from reference to source cameras.
        
        Args:
            ref_pose: Reference camera pose (4x4)
            src_poses: List of source camera poses (4x4)
            
        Returns:
            List of relative poses (4x4)
        """
        relative_poses = []
        
        # Compute inverse of reference pose
        ref_pose_inv = np.linalg.inv(ref_pose)
        
        for src_pose in src_poses:
            # Relative pose: T_ref_to_src = T_src * T_ref^{-1}
            relative_pose = src_pose @ ref_pose_inv
            relative_poses.append(relative_pose)
        
        return relative_poses
    
    def get_camera_params(self, idx: int) -> Dict[str, Any]:
        """Get camera parameters for a sample without loading images."""
        sample = self.samples[idx]
        
        # Select frames
        ref_idx, src_idxs = self._select_frames(sample)
        
        # Get intrinsics
        intrinsics = sample['intrinsics'].copy()
        
        # Get poses if available
        if self.load_gt_pose:
            ref_pose = self._load_pose(self._get_pose_info(sample, ref_idx))
            src_poses = [self._load_pose(self._get_pose_info(sample, i)) for i in src_idxs]
            
            if ref_pose is not None and all(p is not None for p in src_poses):
                relative_poses = self._compute_relative_poses(ref_pose, src_poses)
                relative_poses = [torch.from_numpy(p).float() for p in relative_poses]
                relative_poses = torch.stack(relative_poses, dim=0)
            else:
                relative_poses = None
        else:
            relative_poses = None
        
        # Convert intrinsics to tensor
        intrinsics = torch.from_numpy(intrinsics).float()
        
        return {
            'intrinsics': intrinsics,
            'relative_poses': relative_poses,
            'scene_id': sample['scene_id'],
            'frame_ids': [sample['frame_ids'][ref_idx]] + [sample['frame_ids'][i] for i in src_idxs],
        }