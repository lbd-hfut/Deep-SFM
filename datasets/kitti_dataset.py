"""
KITTI dataset loader for DeepSFM.
"""

import os
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from PIL import Image

from .base import BaseDataset


class KITTIDataset(BaseDataset):
    """KITTI dataset loader for depth estimation."""
    
    def __init__(
        self,
        data_root: str,
        split: str = "train",
        image_size: Tuple[int, int] = (384, 512),
        num_source_views: int = 2,
        min_depth: float = 0.5,
        max_depth: float = 80.0,  # KITTI has larger depth range
        transforms: Optional[Any] = None,
        load_gt_depth: bool = True,
        load_gt_pose: bool = False,  # KITTI poses are in separate files
        use_velodyne: bool = True,
        use_generated: bool = False,
    ):
        """
        Args:
            data_root: Root directory of KITTI dataset
            split: Dataset split ('train', 'val', 'test')
            image_size: Target image size (height, width)
            num_source_views: Number of source views to use
            min_depth: Minimum valid depth value
            max_depth: Maximum valid depth value
            transforms: Data transformations to apply
            load_gt_depth: Whether to load ground truth depth
            load_gt_pose: Whether to load ground truth poses
            use_velodyne: Use Velodyne scans for depth (sparse)
            use_generated: Use generated dense depth maps if available
        """
        self.use_velodyne = use_velodyne
        self.use_generated = use_generated
        
        # KITTI-specific paths
        self.image_dir = Path(data_root) / "images"
        self.depth_dir = Path(data_root) / "depth"
        self.calib_dir = Path(data_root) / "calibration"
        self.pose_dir = Path(data_root) / "poses"
        
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
        """Load KITTI dataset samples.
        
        Expected directory structure (Eigen split):
            data_root/
                images/
                    train/
                        date_sequence/
                            image_02/data/
                                0000000000.png, 0000000001.png, ...
                    val/
                        ... (similar)
                depth/
                    train/
                        date_sequence/
                            proj_depth/
                                groundtruth/
                                    image_02/
                                        0000000000.png, ...
                                velodyne_raw/
                                    image_02/
                                        0000000000.png, ...
                calibration/
                    date_sequence/
                        calib_cam_to_cam.txt
                poses/
                    date_sequence.txt  # (optional)
        """
        samples = []
        
        # Determine split directory
        split_dir = "train" if self.split in ["train", "val"] else "test"
        image_split_dir = self.image_dir / split_dir
        
        if not image_split_dir.exists():
            raise FileNotFoundError(f"Image directory not found: {image_split_dir}")
        
        # Get all sequence directories
        for seq_dir in sorted(image_split_dir.iterdir()):
            if not seq_dir.is_dir():
                continue
            
            date_sequence = seq_dir.name
            
            # Get camera directory (image_02 for left camera)
            cam_dir = seq_dir / "image_02" / "data"
            if not cam_dir.exists():
                continue
            
            # Get all image files
            image_files = sorted(cam_dir.glob("*.png"))
            if len(image_files) < 2:
                continue
            
            # Load calibration
            calib_file = self.calib_dir / date_sequence / "calib_cam_to_cam.txt"
            if calib_file.exists():
                intrinsics = self._load_calibration(calib_file, camera_id=2)  # left color camera
            else:
                # Use default KITTI intrinsics
                intrinsics = np.array([
                    [707.0912, 0, 601.8873],
                    [0, 707.0912, 183.1104],
                    [0, 0, 1]
                ], dtype=np.float32)
            
            # Load poses if available
            poses_file = self.pose_dir / f"{date_sequence}.txt"
            if poses_file.exists() and self.load_gt_pose:
                poses = self._load_poses(poses_file)
            else:
                poses = None
            
            # Create sample
            sample = {
                'scene_id': date_sequence,
                'frame_ids': [int(f.stem) for f in image_files],
                'image_paths': [str(f) for f in image_files],
                'intrinsics': intrinsics,
                'poses': poses,  # List of 4x4 matrices or None
                'date_sequence': date_sequence,
                'cam_dir': cam_dir,
            }
            
            samples.append(sample)
        
        # Split into train/val if needed
        if self.split == "train":
            samples = samples[:int(0.8 * len(samples))]
        elif self.split == "val":
            samples = samples[int(0.8 * len(samples)):]
        # test split uses all samples
        
        return samples
    
    def _load_calibration(self, calib_file: Path, camera_id: int = 2) -> np.ndarray:
        """Load camera calibration from KITTI calibration file."""
        with open(calib_file, 'r') as f:
            lines = f.readlines()
        
        # Find camera matrix for specified camera
        for line in lines:
            if line.startswith(f'P_rect_{camera_id:02d}'):
                parts = line.strip().split()[1:]
                P = np.array([float(p) for p in parts], dtype=np.float32).reshape(3, 4)
                
                # Extract intrinsic matrix (3x3) from projection matrix
                K = P[:, :3]
                return K
        
        # Fallback to default
        return np.array([
            [707.0912, 0, 601.8873],
            [0, 707.0912, 183.1104],
            [0, 0, 1]
        ], dtype=np.float32)
    
    def _load_poses(self, poses_file: Path) -> Optional[List[np.ndarray]]:
        """Load camera poses from KITTI pose file."""
        try:
            poses_data = np.loadtxt(poses_file, dtype=np.float32)
            
            # KITTI poses are 3x4 matrices (rotation + translation)
            poses = []
            for i in range(poses_data.shape[0]):
                pose_3x4 = poses_data[i].reshape(3, 4)
                pose_4x4 = np.eye(4, dtype=np.float32)
                pose_4x4[:3, :] = pose_3x4
                poses.append(pose_4x4)
            
            return poses
        except Exception as e:
            print(f"Error loading poses from {poses_file}: {e}")
            return None
    
    def _load_image(self, image_path: Path) -> np.ndarray:
        """Load image from file."""
        with Image.open(image_path) as img:
            # KITTI images are already RGB
            return np.array(img, dtype=np.float32)
    
    def _load_depth(self, depth_path: Path) -> Optional[np.ndarray]:
        """Load depth map from file."""
        if not depth_path.exists():
            return None
        
        try:
            # KITTI depth maps are stored as 16-bit PNG
            depth_img = Image.open(depth_path)
            depth = np.array(depth_img, dtype=np.float32)
            
            # KITTI depth is stored as uint16, divide by 256 to get meters
            depth = depth / 256.0
            
            # Handle invalid values (0 typically means no measurement)
            depth[depth == 0] = np.nan
            
            return depth
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
        date_sequence = sample['date_sequence']
        frame_id = sample['frame_ids'][frame_idx]
        
        # Determine depth type
        depth_type = "groundtruth" if self.use_generated else "velodyne_raw"
        
        # Construct depth path
        depth_file = self.depth_dir / self.split / date_sequence / "proj_depth" / depth_type / "image_02" / f"{frame_id:010d}.png"
        
        if depth_file.exists():
            return depth_file
        
        return None
    
    def _get_pose_info(self, sample: Dict[str, Any], frame_idx: int) -> Optional[Any]:
        """Get pose information for a frame."""
        if sample['poses'] is not None and frame_idx < len(sample['poses']):
            return sample['poses'][frame_idx]
        return None
    
    def _select_frames(self, sample: Dict[str, Any]) -> Tuple[int, List[int]]:
        """Select reference frame and source frames for KITTI.
        
        KITTI is a sequential dataset, so we select nearby frames.
        """
        num_frames = len(sample['frame_ids'])
        
        # For training, randomly select frames within a window
        if self.split == "train":
            # Select random reference frame (avoid edges)
            ref_idx = np.random.randint(1, num_frames - 1)
            
            # Select nearby frames as sources
            window_size = min(10, num_frames // 2)
            available_idxs = list(range(max(0, ref_idx - window_size), 
                                       min(num_frames, ref_idx + window_size + 1)))
            available_idxs.remove(ref_idx)
            
            num_to_select = min(self.num_source_views, len(available_idxs))
            src_idxs = np.random.choice(available_idxs, size=num_to_select, replace=False).tolist()
        
        # For validation/test, use fixed pattern
        else:
            # Use middle frame as reference
            ref_idx = num_frames // 2
            
            # Select frames at fixed offsets
            offsets = [-2, -1, 1, 2]  # Common for KITTI
            src_idxs = []
            for offset in offsets:
                idx = ref_idx + offset
                if 0 <= idx < num_frames:
                    src_idxs.append(idx)
            
            # Limit to requested number
            src_idxs = src_idxs[:self.num_source_views]
        
        return ref_idx, src_idxs


# Helper function to create dataset
def create_kitti_dataset(
    data_root: str,
    split: str = "train",
    image_size: Tuple[int, int] = (384, 512),
    num_source_views: int = 2,
    use_augmentations: bool = True,
    use_velodyne: bool = True,
    use_generated: bool = False,
) -> KITTIDataset:
    """Create KITTI dataset with appropriate transformations."""
    from .transforms import create_train_transforms, create_val_transforms
    
    if split == "train" and use_augmentations:
        transforms = create_train_transforms(image_size)
    else:
        transforms = create_val_transforms(image_size)
    
    return KITTIDataset(
        data_root=data_root,
        split=split,
        image_size=image_size,
        num_source_views=num_source_views,
        transforms=transforms,
        use_velodyne=use_velodyne,
        use_generated=use_generated,
    )