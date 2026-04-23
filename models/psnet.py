"""
Depth Subnet (PSNet) for DeepSFM.
Estimates dense depth maps from multi-view images.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, List, Optional, Dict, Any

from .submodules import (
    FeatureExtraction,
    DisparityRegression,
    CostVolume3D,
    convbn,
    BasicBlock,
)


class PSNet(nn.Module):
    """Depth estimation subnet using cost volume approach."""
    
    def __init__(
        self,
        min_depth: float = 0.5,
        max_depth: float = 10.0,
        n_depth_bins: int = 64,
        feature_channels: int = 32,
        use_geometric_consistency: bool = True,
        use_depth_augmentation: bool = False,
    ):
        super().__init__()
        
        self.min_depth = min_depth
        self.max_depth = max_depth
        self.n_depth_bins = n_depth_bins
        self.feature_channels = feature_channels
        self.use_geometric_consistency = use_geometric_consistency
        self.use_depth_augmentation = use_depth_augmentation
        
        # Feature extraction
        self.feature_extraction = FeatureExtraction(
            in_channels=3,
            feature_channels=feature_channels
        )
        
        # Cost volume construction and regularization
        self.cost_volume_3d = CostVolume3D(
            feature_channels=feature_channels,
            max_disparity=n_depth_bins
        )
        
        # Disparity regression
        self.disparity_regression = DisparityRegression(
            max_disparity=n_depth_bins
        )
        
        # 2D refinement network
        self.refinement_net = self._build_refinement_net()
        
        # Depth augmentation (optional)
        if use_depth_augmentation:
            self.depth_augmentation = self._build_depth_augmentation()
        
        # Geometric consistency cost (optional)
        if use_geometric_consistency:
            self.geo_cost_channels = 2  # Depth warping error channels
        else:
            self.geo_cost_channels = 0
    
    def _build_refinement_net(self) -> nn.Module:
        """Build 2D CNN for refining depth predictions."""
        layers = []
        
        # Input: features + coarse depth
        in_channels = self.feature_channels + 1
        
        # Refinement blocks
        layers.append(convbn(in_channels, self.feature_channels))
        layers.append(BasicBlock(self.feature_channels, self.feature_channels))
        layers.append(BasicBlock(self.feature_channels, self.feature_channels))
        layers.append(BasicBlock(self.feature_channels, self.feature_channels))
        
        # Final depth prediction
        layers.append(nn.Conv2d(self.feature_channels, 1, kernel_size=3, padding=1, bias=False))
        
        return nn.Sequential(*layers)
    
    def _build_depth_augmentation(self) -> nn.Module:
        """Build network for depth augmentation."""
        return nn.Sequential(
            convbn(1, self.feature_channels // 2),
            BasicBlock(self.feature_channels // 2, self.feature_channels // 2),
            nn.Conv2d(self.feature_channels // 2, self.feature_channels, kernel_size=3, padding=1, bias=False)
        )
    
    def build_cost_volume(
        self,
        ref_feat: torch.Tensor,
        src_feat: torch.Tensor,
        depth_hypos: torch.Tensor,
        intrinsics: torch.Tensor,
        src_pose: torch.Tensor,
        init_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Build 3D cost volume by warping source features to reference view.
        
        Args:
            ref_feat: Reference features (B, C, H, W)
            src_feat: Source features (B, C, H, W)
            depth_hypos: Depth hypotheses (B, D, H, W)
            intrinsics: Camera intrinsics (B, 3, 3)
            src_pose: Relative pose from ref to src (B, 4, 4)
            init_depth: Initial depth for geometric consistency (B, 1, H, W)
            
        Returns:
            Cost volume of shape (B, C*2 + geo_channels, D, H, W)
        """
        batch_size, channels, height, width = ref_feat.shape
        num_depth = depth_hypos.shape[1]
        
        # Initialize cost volume
        cost_volume = []
        
        # For each depth hypothesis
        for d in range(num_depth):
            # Get current depth map
            depth = depth_hypos[:, d:d+1]  # (B, 1, H, W)
            
            # Warp source features to reference view using current depth
            # This is a placeholder - actual implementation requires inverse_warp module
            warped_feat = self._warp_features(src_feat, depth, intrinsics, src_pose)
            
            # Compute matching cost: absolute difference
            cost = torch.abs(ref_feat - warped_feat)  # (B, C, H, W)
            
            # Add geometric consistency cost if enabled
            if self.use_geometric_consistency and init_depth is not None:
                geo_cost = self._compute_geometric_consistency(
                    depth, init_depth, intrinsics, src_pose
                )
                cost = torch.cat([cost, geo_cost], dim=1)  # (B, C + 2, H, W)
            
            cost_volume.append(cost)
        
        # Stack along depth dimension
        cost_volume = torch.stack(cost_volume, dim=2)  # (B, C + geo_channels, D, H, W)
        
        return cost_volume
    
    def _warp_features(
        self,
        src_feat: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        src_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Warp source features to reference view using depth and pose.
        
        Note: This is a simplified placeholder. Actual implementation requires
        differentiable inverse warping with proper handling of occlusions.
        """
        # TODO: Implement proper differentiable inverse warping
        # For now, return identity (no warping) for initial setup
        return src_feat
    
    def _compute_geometric_consistency(
        self,
        depth: torch.Tensor,
        init_depth: torch.Tensor,
        intrinsics: torch.Tensor,
        src_pose: torch.Tensor,
    ) -> torch.Tensor:
        """Compute geometric consistency cost between predicted and initial depth.
        
        Measures consistency between depth predicted from reference view and
        depth warped from source view.
        """
        # TODO: Implement geometric consistency computation
        # This should compute two channels:
        # 1. Depth from reference projected to source view
        # 2. Depth from source warped to reference view
        # The difference between these provides geometric consistency cue
        
        batch_size, _, height, width = depth.shape
        device = depth.device
        
        # Placeholder: return zeros
        return torch.zeros(batch_size, 2, height, width, device=device)
    
    def generate_depth_hypos(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
        init_depth: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Generate depth hypotheses for cost volume construction.
        
        Args:
            batch_size: Batch size
            height: Feature map height
            width: Feature map width
            device: Device for tensor creation
            init_depth: Initial depth map (optional, for adaptive sampling)
            
        Returns:
            Depth hypotheses of shape (B, D, H, W)
        """
        if init_depth is not None and self.use_depth_augmentation:
            # Adaptive sampling around initial depth
            depth_hypos = self._adaptive_depth_sampling(init_depth)
        else:
            # Uniform sampling in inverse depth (disparity) space
            depth_hypos = self._uniform_depth_sampling(batch_size, height, width, device)
        
        return depth_hypos
    
    def _uniform_depth_sampling(
        self,
        batch_size: int,
        height: int,
        width: int,
        device: torch.device,
    ) -> torch.Tensor:
        """Uniform sampling in inverse depth space."""
        # Create disparity values
        disparity = torch.linspace(0, self.n_depth_bins - 1, self.n_depth_bins, device=device)
        disparity = disparity.view(1, -1, 1, 1).expand(batch_size, -1, height, width)
        
        # Convert disparity to depth: depth = min_depth * n_bins / (disparity + epsilon)
        epsilon = 1e-6
        depth = self.min_depth * self.n_depth_bins / (disparity + epsilon)
        depth = torch.clamp(depth, self.min_depth, self.max_depth)
        
        return depth
    
    def _adaptive_depth_sampling(self, init_depth: torch.Tensor) -> torch.Tensor:
        """Adaptive sampling around initial depth estimate."""
        # TODO: Implement adaptive depth sampling
        # This should sample depths in a range around the initial depth
        # based on uncertainty or other criteria
        
        batch_size, _, height, width = init_depth.shape
        device = init_depth.device
        
        # For now, use uniform sampling
        return self._uniform_depth_sampling(batch_size, height, width, device)
    
    def forward(
        self,
        ref_img: torch.Tensor,
        src_imgs: torch.Tensor,
        intrinsics: torch.Tensor,
        src_poses: torch.Tensor,
        init_depth: Optional[torch.Tensor] = None,
        init_pose: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Forward pass of depth subnet.
        
        Args:
            ref_img: Reference image (B, 3, H, W)
            src_imgs: Source images (B, N, 3, H, W) where N is number of source views
            intrinsics: Camera intrinsics (B, 3, 3)
            src_poses: Relative poses from ref to src (B, N, 4, 4)
            init_depth: Initial depth map (optional)
            init_pose: Initial pose (optional, not used in depth subnet)
            
        Returns:
            Dictionary containing:
                - 'depth': Refined depth map (B, 1, H, W)
                - 'depth_coarse': Coarse depth map before refinement
                - 'prob_volume': Probability volume (B, D, H, W)
                - 'depth_hypos': Depth hypotheses used
        """
        batch_size, num_src, _, height, width = src_imgs.shape
        device = ref_img.device
        
        # Extract features from reference image
        ref_feat_full, ref_feat_half, ref_feat_quarter, ref_feat_eighth = \
            self.feature_extraction(ref_img)
        
        # Use features at 1/4 resolution for cost volume
        ref_feat = ref_feat_quarter
        feat_height, feat_width = ref_feat.shape[-2:]
        
        # Generate depth hypotheses
        depth_hypos = self.generate_depth_hypos(
            batch_size, feat_height, feat_width, device, init_depth
        )
        
        # Initialize cost volumes list
        cost_volumes = []
        
        # Process each source view
        for i in range(num_src):
            src_img = src_imgs[:, i]
            src_pose = src_poses[:, i]
            
            # Extract features from source image
            src_feat_full, src_feat_half, src_feat_quarter, src_feat_eighth = \
                self.feature_extraction(src_img)
            src_feat = src_feat_quarter
            
            # Build cost volume for this source view
            cost_volume = self.build_cost_volume(
                ref_feat=ref_feat,
                src_feat=src_feat,
                depth_hypos=depth_hypos,
                intrinsics=intrinsics,
                src_pose=src_pose,
                init_depth=init_depth,
            )
            cost_volumes.append(cost_volume)
        
        # Average cost volumes from multiple source views
        cost_volume = torch.stack(cost_volumes, dim=0).mean(dim=0)
        
        # Regularize cost volume and get probability volume
        prob_volume = self.cost_volume_3d(cost_volume)  # (B, D, H, W)
        
        # Convert probability volume to coarse depth
        disparity_coarse = self.disparity_regression(prob_volume)
        depth_coarse = self.min_depth * self.n_depth_bins / (disparity_coarse + 1e-6)
        depth_coarse = torch.clamp(depth_coarse, self.min_depth, self.max_depth)
        
        # Refine coarse depth using 2D CNN
        # Upsample features and depth to full resolution
        depth_coarse_full = F.interpolate(
            depth_coarse, size=(height, width), mode='bilinear', align_corners=False
        )
        ref_feat_full_resized = F.interpolate(
            ref_feat_full, size=(height, width), mode='bilinear', align_corners=False
        )
        
        # Concatenate features with coarse depth
        refinement_input = torch.cat([ref_feat_full_resized, depth_coarse_full], dim=1)
        
        # Refine depth
        depth_refinement = self.refinement_net(refinement_input)
        depth_final = depth_coarse_full + depth_refinement
        depth_final = torch.clamp(depth_final, self.min_depth, self.max_depth)
        
        # Apply depth augmentation if enabled
        if self.use_depth_augmentation and init_depth is not None:
            depth_aug = self.depth_augmentation(init_depth)
            depth_final = depth_final + depth_aug
        
        return {
            'depth': depth_final,
            'depth_coarse': depth_coarse_full,
            'prob_volume': prob_volume,
            'depth_hypos': depth_hypos,
        }
    
    def compute_loss(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        weights: Optional[Dict[str, float]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute loss for depth subnet.
        
        Args:
            predictions: Output from forward pass
            targets: Ground truth containing 'depth' and optionally 'mask'
            weights: Loss weights for different components
            
        Returns:
            Dictionary of loss values
        """
        if weights is None:
            weights = {
                'depth': 1.0,
                'smoothness': 0.1,
                'geometric': 0.5,
            }
        
        pred_depth = predictions['depth']
        gt_depth = targets['depth']
        mask = targets.get('mask', torch.ones_like(gt_depth))
        
        losses = {}
        
        # Depth loss (smooth L1)
        depth_diff = torch.abs(pred_depth - gt_depth)
        depth_loss = torch.sum(depth_diff * mask) / (torch.sum(mask) + 1e-6)
        losses['depth'] = depth_loss * weights['depth']
        
        # Smoothness loss (edge-aware)
        if weights.get('smoothness', 0.0) > 0:
            smoothness_loss = self._compute_smoothness_loss(pred_depth, predictions.get('ref_img'))
            losses['smoothness'] = smoothness_loss * weights['smoothness']
        
        # Geometric consistency loss (if enabled)
        if self.use_geometric_consistency and weights.get('geometric', 0.0) > 0:
            # TODO: Implement geometric consistency loss
            geo_loss = torch.tensor(0.0, device=pred_depth.device)
            losses['geometric'] = geo_loss * weights['geometric']
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses
    
    def _compute_smoothness_loss(self, depth: torch.Tensor, image: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Compute edge-aware smoothness loss."""
        # Gradient of depth
        depth_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        depth_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        
        if image is not None:
            # Weight by image gradients (edge-aware)
            image_dx = torch.mean(torch.abs(image[:, :, :, :-1] - image[:, :, :, 1:]), dim=1, keepdim=True)
            image_dy = torch.mean(torch.abs(image[:, :, :-1, :] - image[:, :, 1:, :]), dim=1, keepdim=True)
            depth_dx *= torch.exp(-image_dx)
            depth_dy *= torch.exp(-image_dy)
        
        return depth_dx.mean() + depth_dy.mean()


# Simplified version for testing
class SimplePSNet(PSNet):
    """Simplified PSNet for initial testing without complex warping."""
    
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        # Override with simpler cost volume building
        self.use_geometric_consistency = False
        self.use_depth_augmentation = False
    
    def build_cost_volume(self, ref_feat, src_feat, depth_hypos, **kwargs):
        """Simplified cost volume using feature difference only."""
        batch_size, channels, height, width = ref_feat.shape
        num_depth = depth_hypos.shape[1]
        
        # Simple cost: absolute difference expanded along depth
        cost = torch.abs(ref_feat.unsqueeze(2) - src_feat.unsqueeze(2))
        cost = cost.expand(-1, -1, num_depth, -1, -1)
        
        return cost