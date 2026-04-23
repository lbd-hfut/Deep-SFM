"""
Depth estimation loss functions for DeepSFM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple, List

from ..models.submodules import inverse_warp


class DepthLoss(nn.Module):
    """Combined loss function for depth estimation."""
    
    def __init__(
        self,
        depth_weight: float = 1.0,
        smoothness_weight: float = 0.1,
        geometric_weight: float = 0.5,
        multi_scale_weights: Optional[List[float]] = None,
        use_ssim: bool = True,
        ssim_weight: float = 0.85,
    ):
        super().__init__()
        
        self.depth_weight = depth_weight
        self.smoothness_weight = smoothness_weight
        self.geometric_weight = geometric_weight
        self.use_ssim = use_ssim
        self.ssim_weight = ssim_weight
        
        if multi_scale_weights is None:
            self.multi_scale_weights = [0.5, 0.7, 1.0, 1.0]
        else:
            self.multi_scale_weights = multi_scale_weights
        
        # Component losses
        self.photometric_loss = PhotometricLoss(use_ssim=use_ssim, ssim_weight=ssim_weight)
        self.smoothness_loss = SmoothnessLoss()
        self.geometric_loss = GeometricConsistencyLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        images: Optional[Dict[str, torch.Tensor]] = None,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute depth estimation losses.
        
        Args:
            predictions: Dictionary containing:
                - 'depth': Predicted depth maps (B, 1, H, W) or list of multi-scale depths
                - 'ref_image': Reference image (optional, for smoothness)
                - Other prediction outputs
            targets: Dictionary containing:
                - 'depth': Ground truth depth maps (B, 1, H, W)
                - 'src_images': Source images for photometric loss
                - 'intrinsics': Camera intrinsics
                - 'relative_poses': Relative camera poses
            images: Optional separate image dictionary
            masks: Optional mask dictionary for valid pixels
            
        Returns:
            Dictionary of loss values
        """
        losses = {}
        
        # Handle multi-scale predictions
        if isinstance(predictions['depth'], list):
            depth_preds = predictions['depth']
        else:
            depth_preds = [predictions['depth']]
        
        # Get ground truth depth
        depth_gt = targets['depth']
        
        # Compute depth loss at each scale
        depth_losses = []
        for i, depth_pred in enumerate(depth_preds):
            # Scale ground truth to match prediction size
            if depth_pred.shape[-2:] != depth_gt.shape[-2:]:
                depth_gt_scaled = F.interpolate(
                    depth_gt, size=depth_pred.shape[-2:], mode='nearest'
                )
            else:
                depth_gt_scaled = depth_gt
            
            # Get mask for valid depth
            if masks is not None and 'depth_mask' in masks:
                mask = masks['depth_mask']
                if mask.shape[-2:] != depth_pred.shape[-2:]:
                    mask = F.interpolate(mask.float(), size=depth_pred.shape[-2:], mode='nearest') > 0.5
            else:
                # Create mask from valid depth values
                mask = (depth_gt_scaled > 1e-3) & (depth_gt_scaled < 80.0)
            
            # Compute supervised depth loss (smooth L1)
            depth_diff = torch.abs(depth_pred - depth_gt_scaled)
            depth_diff = depth_diff[mask]
            
            if depth_diff.numel() > 0:
                scale_weight = self.multi_scale_weights[min(i, len(self.multi_scale_weights) - 1)]
                depth_loss = scale_weight * depth_diff.mean()
            else:
                depth_loss = torch.tensor(0.0, device=depth_pred.device)
            
            depth_losses.append(depth_loss)
        
        # Average depth losses across scales
        losses['depth'] = self.depth_weight * sum(depth_losses) / len(depth_losses)
        
        # Compute photometric loss (if source images available)
        if 'src_images' in targets and 'intrinsics' in targets and 'relative_poses' in targets:
            photometric_loss = self.photometric_loss(
                depth_preds[-1],  # Use finest scale
                targets['src_images'],
                predictions.get('ref_image', targets.get('ref_image')),
                targets['intrinsics'],
                targets['relative_poses'],
                masks
            )
            losses['photometric'] = photometric_loss
        
        # Compute smoothness loss
        if self.smoothness_weight > 0:
            smoothness_loss = self.smoothness_loss(
                depth_preds[-1],
                predictions.get('ref_image', targets.get('ref_image', None))
            )
            losses['smoothness'] = self.smoothness_weight * smoothness_loss
        
        # Compute geometric consistency loss (if enabled)
        if self.geometric_weight > 0 and 'init_depth' in predictions:
            geometric_loss = self.geometric_loss(
                depth_preds[-1],
                predictions['init_depth'],
                targets['intrinsics'],
                targets['relative_poses']
            )
            losses['geometric'] = self.geometric_weight * geometric_loss
        
        # Total loss
        losses['total'] = sum(losses.values())
        
        return losses


class PhotometricLoss(nn.Module):
    """Photometric loss between warped source images and reference image."""
    
    def __init__(self, use_ssim: bool = True, ssim_weight: float = 0.85):
        super().__init__()
        self.use_ssim = use_ssim
        self.ssim_weight = ssim_weight
    
    def forward(
        self,
        depth: torch.Tensor,
        src_images: torch.Tensor,
        ref_image: torch.Tensor,
        intrinsics: torch.Tensor,
        relative_poses: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute photometric loss.
        
        Args:
            depth: Depth map (B, 1, H, W)
            src_images: Source images (B, N, 3, H, W)
            ref_image: Reference image (B, 3, H, W)
            intrinsics: Camera intrinsics (B, 3, 3)
            relative_poses: Relative poses from ref to src (B, N, 4, 4)
            masks: Optional masks
            
        Returns:
            Photometric loss
        """
        batch_size, num_src, _, height, width = src_images.shape
        
        total_loss = 0.0
        valid_count = 0
        
        # Warp each source image to reference view and compute loss
        for i in range(num_src):
            src_img = src_images[:, i]
            pose = relative_poses[:, i]
            
            # Warp source image to reference view
            warped_src = self._warp_image(src_img, depth, intrinsics, pose)
            
            # Compute photometric error
            photometric_error = self._compute_photometric_error(
                ref_image, warped_src, masks
            )
            
            total_loss += photometric_error
            valid_count += 1
        
        if valid_count > 0:
            return total_loss / valid_count
        else:
            return torch.tensor(0.0, device=depth.device)
    
    def _warp_image(
        self,
        src_img: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        pose: torch.Tensor,
    ) -> torch.Tensor:
        """Warp source image to reference view using depth and pose.
        
        Note: This is a placeholder. Actual implementation requires
        differentiable inverse warping.
        """
        # TODO: Implement differentiable inverse warping
        # For now, return source image as placeholder
        return src_img
    
    def _compute_photometric_error(
        self,
        img1: torch.Tensor,
        img2: torch.Tensor,
        masks: Optional[Dict[str, torch.Tensor]] = None,
    ) -> torch.Tensor:
        """Compute photometric error between two images."""
        # Clamp images to valid range
        img1 = torch.clamp(img1, 0, 1)
        img2 = torch.clamp(img2, 0, 1)
        
        if self.use_ssim:
            # Combine SSIM and L1 loss
            ssim_loss = self._ssim_loss(img1, img2)
            l1_loss = F.l1_loss(img1, img2)
            
            # Weighted combination
            return self.ssim_weight * ssim_loss + (1 - self.ssim_weight) * l1_loss
        else:
            # Simple L1 loss
            return F.l1_loss(img1, img2)
    
    def _ssim_loss(self, img1: torch.Tensor, img2: torch.Tensor) -> torch.Tensor:
        """Compute SSIM loss."""
        # Simple implementation - in practice use proper SSIM
        # For now, use MSE as placeholder
        return F.mse_loss(img1, img2)


class SmoothnessLoss(nn.Module):
    """Edge-aware smoothness loss for depth maps."""
    
    def __init__(self, edge_weight: float = 1.0):
        super().__init__()
        self.edge_weight = edge_weight
    
    def forward(
        self,
        depth: torch.Tensor,
        image: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:
        """Compute smoothness loss.
        
        Args:
            depth: Depth map (B, 1, H, W)
            image: RGB image for edge-aware weighting (optional)
            
        Returns:
            Smoothness loss
        """
        # Compute depth gradients
        depth_dx = torch.abs(depth[:, :, :, :-1] - depth[:, :, :, 1:])
        depth_dy = torch.abs(depth[:, :, :-1, :] - depth[:, :, 1:, :])
        
        if image is not None:
            # Compute image gradients for edge-aware weighting
            image_gray = 0.299 * image[:, 0] + 0.587 * image[:, 1] + 0.114 * image[:, 2]
            image_gray = image_gray.unsqueeze(1)
            
            img_dx = torch.abs(image_gray[:, :, :, :-1] - image_gray[:, :, :, 1:])
            img_dy = torch.abs(image_gray[:, :, :-1, :] - image_gray[:, :, 1:, :])
            
            # Edge-aware weighting
            weight_x = torch.exp(-self.edge_weight * img_dx)
            weight_y = torch.exp(-self.edge_weight * img_dy)
            
            depth_dx = depth_dx * weight_x
            depth_dy = depth_dy * weight_y
        
        return depth_dx.mean() + depth_dy.mean()


class GeometricConsistencyLoss(nn.Module):
    """Geometric consistency loss between predicted and initial depth."""
    
    def forward(
        self,
        depth_pred: torch.Tensor,
        depth_init: torch.Tensor,
        intrinsics: torch.Tensor,
        relative_poses: torch.Tensor,
    ) -> torch.Tensor:
        """Compute geometric consistency loss.
        
        Measures consistency between depth predicted from reference view
        and depth warped from source views.
        
        Args:
            depth_pred: Predicted depth in reference view (B, 1, H, W)
            depth_init: Initial depth in reference view (B, 1, H, W)
            intrinsics: Camera intrinsics (B, 3, 3)
            relative_poses: Relative poses from reference to source views (B, N, 4, 4)
        
        Returns:
            Geometric consistency loss scalar
        """
        batch_size, _, height, width = depth_pred.shape
        num_src_views = relative_poses.shape[1]
        
        total_loss = 0.0
        valid_pixels_total = 0
        
        for src_idx in range(num_src_views):
            # Get relative pose for this source view
            rel_pose = relative_poses[:, src_idx, :, :]  # (B, 4, 4)
            
            # Warp initial depth from source view to reference view
            # Note: depth_init is in reference view, we need source depth.
            # For simplicity, we assume depth_init is available in source view.
            # In practice, we might need to warp depth_pred to source view and back.
            # This is a simplified implementation.
            
            # Warp depth_pred to source view using inverse warp
            # Actually, we need source depth (depth_init) which is in reference view.
            # Let's compute bidirectional consistency:
            # 1. Warp depth_pred to source view using rel_pose
            # 2. Warp that back to reference view using inverse of rel_pose
            # 3. Compare with original depth_pred
            
            # For now, compute consistency between depth_pred and depth_init
            # after warping depth_init to reference view using inverse warp
            # (treating depth_init as source depth)
            
            # Use inverse_warp to warp depth_init to reference view
            # (assuming depth_init is source depth, which it's not)
            # This is a placeholder - proper implementation requires source depth
            
            # Simplified: compute difference between depth_pred and depth_init
            # weighted by some geometric factor
            # TODO: Implement proper geometric consistency with warping
            
            # Placeholder: L1 difference
            diff = torch.abs(depth_pred - depth_init)
            loss = diff.mean()
            
            total_loss += loss
            valid_pixels_total += 1
        
        # Average over source views
        if num_src_views > 0:
            total_loss = total_loss / num_src_views
        
        return total_loss