#!/usr/bin/env python3
"""
Unified DeepSFM model for iterative bundle adjustment.
Combines depth subnet (PSNet) and pose subnet (PoseNet) for alternating refinement.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional, Any

from .psnet import PSNet
from .posenet import PoseNet
from .submodules import inverse_warp


class DeepSFM(nn.Module):
    """Unified DeepSFM model for iterative depth and pose estimation."""
    
    def __init__(
        self,
        depth_model: Optional[PSNet] = None,
        pose_model: Optional[PoseNet] = None,
        depth_config: Optional[Dict[str, Any]] = None,
        pose_config: Optional[Dict[str, Any]] = None,
        max_iterations: int = 5,
        use_geometric_consistency: bool = True,
    ):
        """
        Args:
            depth_model: Pretrained depth subnet (PSNet)
            pose_model: Pretrained pose subnet (PoseNet)
            depth_config: Configuration for depth subnet if not provided
            pose_config: Configuration for pose subnet if not provided
            max_iterations: Maximum number of alternating iterations
            use_geometric_consistency: Whether to use geometric consistency loss
        """
        super().__init__()
        
        # Initialize or use provided subnets
        if depth_model is not None:
            self.depth_net = depth_model
        else:
            depth_config = depth_config or {}
            self.depth_net = PSNet(
                min_depth=depth_config.get('min_depth', 0.5),
                max_depth=depth_config.get('max_depth', 10.0),
                n_depth_bins=depth_config.get('n_depth_bins', 64),
                feature_channels=depth_config.get('feature_channels', 32),
                use_geometric_consistency=use_geometric_consistency,
            )
        
        if pose_model is not None:
            self.pose_net = pose_model
        else:
            pose_config = pose_config or {}
            self.pose_net = PoseNet(
                translation_std=pose_config.get('translation_std', 0.27),
                rotation_std=pose_config.get('rotation_std', 0.12),
                n_pose_bins=pose_config.get('n_pose_bins', 10),
                feature_channels=pose_config.get('feature_channels', 32),
                use_geometric_consistency=use_geometric_consistency,
            )
        
        self.max_iterations = max_iterations
        self.use_geometric_consistency = use_geometric_consistency
        
        # Iteration counter (for tracking during inference)
        self.current_iteration = 0
    
    def forward(
        self,
        ref_img: torch.Tensor,
        src_imgs: List[torch.Tensor],
        intrinsics: torch.Tensor,
        init_depth: Optional[torch.Tensor] = None,
        init_poses: Optional[torch.Tensor] = None,
        num_iterations: Optional[int] = None,
        return_all_iterations: bool = False,
    ) -> Dict[str, Any]:
        """
        Forward pass with alternating depth and pose refinement.
        
        Args:
            ref_img: Reference image tensor (B, 3, H, W)
            src_imgs: List of source image tensors, each (B, 3, H, W)
            intrinsics: Camera intrinsics matrix (B, 3, 3)
            init_depth: Initial depth estimate (B, 1, H, W) or None
            init_poses: Initial relative poses (B, N, 4, 4) or None
            num_iterations: Number of alternating iterations (default: max_iterations)
            return_all_iterations: Whether to return results from all iterations
        
        Returns:
            Dictionary containing:
                - 'depth': Final depth estimate (B, 1, H, W)
                - 'poses': Final relative poses (B, N, 4, 4)
                - 'depth_history': List of depth estimates per iteration (if return_all_iterations)
                - 'pose_history': List of pose estimates per iteration (if return_all_iterations)
                - 'iteration': Final iteration number
        """
        batch_size = ref_img.shape[0]
        num_src_views = len(src_imgs)
        
        # Convert src_imgs list to tensor (B, N, C, H, W)
        src_imgs_tensor = torch.stack(src_imgs, dim=1)  # (B, N, C, H, W)
        
        # Initialize depth and poses if not provided
        if init_depth is None:
            # Initial depth estimation using depth subnet with identity poses
            init_poses_identity = torch.eye(4, device=ref_img.device).unsqueeze(0).unsqueeze(0).repeat(
                batch_size, num_src_views, 1, 1
            )
            init_depth = self.depth_net(
                ref_img=ref_img,
                src_imgs=src_imgs,
                intrinsics=intrinsics,
                src_poses=init_poses_identity,
                init_depth=None,
                init_pose=None,
            )['depth']
        
        if init_poses is None:
            # Initial pose estimation using pose subnet with initial depth
            init_poses = self.pose_net(
                ref_img=ref_img,
                src_imgs=src_imgs,
                intrinsics=intrinsics,
                init_depth=init_depth,
                init_pose=None,
            )['poses']
        
        current_depth = init_depth
        current_poses = init_poses
        
        # Store history if requested
        depth_history = [current_depth] if return_all_iterations else None
        pose_history = [current_poses] if return_all_iterations else None
        
        # Determine number of iterations
        n_iter = num_iterations if num_iterations is not None else self.max_iterations
        
        # Alternating optimization
        for iter_idx in range(n_iter):
            self.current_iteration = iter_idx
            
            # Step 1: Refine depth using current poses
            depth_output = self.depth_net(
                ref_img=ref_img,
                src_imgs=src_imgs,
                intrinsics=intrinsics,
                src_poses=current_poses,
                init_depth=current_depth,
                init_pose=current_poses,
            )
            current_depth = depth_output['depth']
            
            # Step 2: Refine poses using updated depth
            pose_output = self.pose_net(
                ref_img=ref_img,
                src_imgs=src_imgs,
                intrinsics=intrinsics,
                init_depth=current_depth,
                init_pose=current_poses,
            )
            current_poses = pose_output['poses']
            
            # Store history
            if return_all_iterations:
                depth_history.append(current_depth)
                pose_history.append(current_poses)
        
        # Prepare output
        output = {
            'depth': current_depth,
            'poses': current_poses,
            'iteration': n_iter,
        }
        
        if return_all_iterations:
            output['depth_history'] = depth_history
            output['pose_history'] = pose_history
        
        return output
    
    def forward_single_iteration(
        self,
        ref_img: torch.Tensor,
        src_imgs: List[torch.Tensor],
        intrinsics: torch.Tensor,
        current_depth: torch.Tensor,
        current_poses: torch.Tensor,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Perform a single iteration of alternating optimization.
        
        Args:
            ref_img: Reference image (B, 3, H, W)
            src_imgs: List of source images, each (B, 3, H, W)
            intrinsics: Camera intrinsics (B, 3, 3)
            current_depth: Current depth estimate (B, 1, H, W)
            current_poses: Current relative poses (B, N, 4, 4)
        
        Returns:
            Updated depth and poses
        """
        # Refine depth
        depth_output = self.depth_net(
            ref_img=ref_img,
            src_imgs=src_imgs,
            intrinsics=intrinsics,
            src_poses=current_poses,
            init_depth=current_depth,
            init_pose=current_poses,
        )
        updated_depth = depth_output['depth']
        
        # Refine poses
        pose_output = self.pose_net(
            ref_img=ref_img,
            src_imgs=src_imgs,
            intrinsics=intrinsics,
            init_depth=updated_depth,
            init_pose=current_poses,
        )
        updated_poses = pose_output['poses']
        
        return updated_depth, updated_poses
    
    def compute_geometric_consistency(
        self,
        depth: torch.Tensor,
        src_depth: torch.Tensor,
        intrinsics: torch.Tensor,
        relative_pose: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute geometric consistency between depth maps.
        
        Args:
            depth: Depth in reference view (B, 1, H, W)
            src_depth: Depth in source view (B, 1, H, W)
            intrinsics: Camera intrinsics (B, 3, 3)
            relative_pose: Relative pose from reference to source (B, 4, 4)
        
        Returns:
            Consistency error map (B, 1, H, W)
        """
        # Warp source depth to reference view
        warped_src_depth, valid_mask = inverse_warp(
            src_depth,
            depth,
            intrinsics,
            relative_pose,
        )
        
        # Compute absolute difference
        diff = torch.abs(depth - warped_src_depth)
        
        # Apply validity mask
        diff = diff * valid_mask.unsqueeze(1)
        
        return diff
    
    def load_depth_checkpoint(self, checkpoint_path: str) -> None:
        """Load pretrained weights for depth subnet."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            self.depth_net.load_state_dict(checkpoint['state_dict'])
        else:
            self.depth_net.load_state_dict(checkpoint)
        print(f"Loaded depth checkpoint from {checkpoint_path}")
    
    def load_pose_checkpoint(self, checkpoint_path: str) -> None:
        """Load pretrained weights for pose subnet."""
        checkpoint = torch.load(checkpoint_path, map_location='cpu')
        if 'state_dict' in checkpoint:
            self.pose_net.load_state_dict(checkpoint['state_dict'])
        else:
            self.pose_net.load_state_dict(checkpoint)
        print(f"Loaded pose checkpoint from {checkpoint_path}")
    
    def load_checkpoints(self, depth_checkpoint: str, pose_checkpoint: str) -> None:
        """Load both depth and pose checkpoints."""
        self.load_depth_checkpoint(depth_checkpoint)
        self.load_pose_checkpoint(pose_checkpoint)
    
    def freeze_depth_net(self) -> None:
        """Freeze depth subnet parameters."""
        for param in self.depth_net.parameters():
            param.requires_grad = False
    
    def freeze_pose_net(self) -> None:
        """Freeze pose subnet parameters."""
        for param in self.pose_net.parameters():
            param.requires_grad = False
    
    def unfreeze_all(self) -> None:
        """Unfreeze all parameters."""
        for param in self.depth_net.parameters():
            param.requires_grad = True
        for param in self.pose_net.parameters():
            param.requires_grad = True