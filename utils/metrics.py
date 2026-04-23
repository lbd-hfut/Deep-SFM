"""
Metrics computation for DeepSFM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Tuple, Optional
import numpy as np


class MetricsTracker:
    """Track metrics during training."""
    
    def __init__(self):
        self.metrics = {}
        self.counts = {}
    
    def update(self, metrics_dict: Dict[str, float]):
        """Update metrics with new values."""
        for key, value in metrics_dict.items():
            if not isinstance(value, (int, float, torch.Tensor)):
                continue
            
            if isinstance(value, torch.Tensor):
                value = value.item()
            
            if key in self.metrics:
                self.metrics[key] += value
                self.counts[key] += 1
            else:
                self.metrics[key] = value
                self.counts[key] = 1
    
    def compute(self) -> Dict[str, float]:
        """Compute average metrics."""
        avg_metrics = {}
        for key, total in self.metrics.items():
            count = self.counts[key]
            avg_metrics[key] = total / count
        return avg_metrics
    
    def reset(self):
        """Reset metrics."""
        self.metrics = {}
        self.counts = {}


def compute_depth_metrics(
    depth_pred: torch.Tensor,
    depth_gt: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    min_depth: float = 0.5,
    max_depth: float = 80.0,
) -> Dict[str, float]:
    """Compute depth estimation metrics.
    
    Args:
        depth_pred: Predicted depth map (B, 1, H, W)
        depth_gt: Ground truth depth map (B, 1, H, W)
        mask: Valid depth mask (B, 1, H, W)
        min_depth: Minimum valid depth
        max_depth: Maximum valid depth
        
    Returns:
        Dictionary of depth metrics
    """
    if mask is None:
        mask = (depth_gt > min_depth) & (depth_gt < max_depth)
    
    # Apply mask
    depth_pred_masked = depth_pred[mask]
    depth_gt_masked = depth_gt[mask]
    
    if depth_pred_masked.numel() == 0:
        return {}
    
    # Ensure predictions are positive
    depth_pred_masked = torch.clamp(depth_pred_masked, min=1e-6)
    depth_gt_masked = torch.clamp(depth_gt_masked, min=1e-6)
    
    # Compute metrics
    metrics = {}
    
    # Absolute relative error
    abs_rel = torch.mean(torch.abs(depth_pred_masked - depth_gt_masked) / depth_gt_masked)
    metrics['abs_rel'] = abs_rel.item()
    
    # Squared relative error
    sq_rel = torch.mean(((depth_pred_masked - depth_gt_masked) ** 2) / depth_gt_masked)
    metrics['sq_rel'] = sq_rel.item()
    
    # Root mean squared error
    rmse = torch.sqrt(torch.mean((depth_pred_masked - depth_gt_masked) ** 2))
    metrics['rmse'] = rmse.item()
    
    # Root mean squared error in log space
    rmse_log = torch.sqrt(torch.mean((torch.log(depth_pred_masked) - torch.log(depth_gt_masked)) ** 2))
    metrics['rmse_log'] = rmse_log.item()
    
    # Accuracy thresholds
    thresh = torch.max(depth_pred_masked / depth_gt_masked, depth_gt_masked / depth_pred_masked)
    
    a1 = (thresh < 1.25).float().mean()
    a2 = (thresh < 1.25 ** 2).float().mean()
    a3 = (thresh < 1.25 ** 3).float().mean()
    
    metrics['a1'] = a1.item()
    metrics['a2'] = a2.item()
    metrics['a3'] = a3.item()
    
    # Mean absolute error
    mae = torch.mean(torch.abs(depth_pred_masked - depth_gt_masked))
    metrics['mae'] = mae.item()
    
    # Mean log10 error
    log10 = torch.mean(torch.abs(torch.log10(depth_pred_masked) - torch.log10(depth_gt_masked)))
    metrics['log10'] = log10.item()
    
    return metrics


def compute_pose_metrics(
    pose_pred: torch.Tensor,
    pose_gt: torch.Tensor,
) -> Dict[str, float]:
    """Compute pose estimation metrics.
    
    Args:
        pose_pred: Predicted pose matrices (B, 4, 4) or (B, N, 4, 4)
        pose_gt: Ground truth pose matrices (B, 4, 4) or (B, N, 4, 4)
        
    Returns:
        Dictionary of pose metrics
    """
    metrics = {}
    
    # Handle multiple poses (average across source views)
    if pose_pred.dim() == 4 and pose_pred.shape[1] > 1:
        # Average metrics across source views
        batch_size, num_poses = pose_pred.shape[:2]
        
        trans_errors = []
        rot_errors = []
        
        for i in range(num_poses):
            pose_pred_i = pose_pred[:, i]
            pose_gt_i = pose_gt[:, i] if pose_gt.dim() == 4 else pose_gt
            
            # Compute errors for this pose
            trans_error, rot_error = _compute_pose_error(pose_pred_i, pose_gt_i)
            
            trans_errors.append(trans_error)
            rot_errors.append(rot_error)
        
        # Average across poses
        trans_error = torch.stack(trans_errors).mean()
        rot_error = torch.stack(rot_errors).mean()
    else:
        # Single pose
        trans_error, rot_error = _compute_pose_error(pose_pred, pose_gt)
    
    metrics['translation_error'] = trans_error.item()
    metrics['rotation_error'] = rot_error.item()
    
    return metrics


def _compute_pose_error(
    pose_pred: torch.Tensor,
    pose_gt: torch.Tensor,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """Compute translation and rotation errors between poses."""
    # Extract translation vectors
    trans_pred = pose_pred[:, :3, 3]
    trans_gt = pose_gt[:, :3, 3]
    
    # Translation error (L2 distance)
    trans_error = torch.norm(trans_pred - trans_gt, dim=1).mean()
    
    # Extract rotation matrices
    R_pred = pose_pred[:, :3, :3]
    R_gt = pose_gt[:, :3, :3]
    
    # Compute relative rotation: R_rel = R_gt^T * R_pred
    R_rel = torch.bmm(R_gt.transpose(1, 2), R_pred)
    
    # Convert rotation matrix to angle-axis and compute angle
    # Using trace: angle = acos((trace(R) - 1) / 2)
    trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
    angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
    
    # Convert to degrees
    rot_error = torch.rad2deg(angle).mean()
    
    return trans_error, rot_error


def compute_reprojection_error(
    points_3d: torch.Tensor,
    points_2d: torch.Tensor,
    pose: torch.Tensor,
    intrinsics: torch.Tensor,
) -> torch.Tensor:
    """Compute reprojection error for 3D points.
    
    Args:
        points_3d: 3D points in world coordinates (B, N, 3)
        points_2d: 2D image points (B, N, 2)
        pose: Camera pose (B, 4, 4)
        intrinsics: Camera intrinsics (B, 3, 3)
        
    Returns:
        Reprojection error
    """
    # Transform 3D points to camera coordinates
    R = pose[:, :3, :3]  # (B, 3, 3)
    t = pose[:, :3, 3:]  # (B, 3, 1)
    
    # points_3d: (B, N, 3) -> (B, N, 3, 1)
    points_3d_homo = torch.cat([points_3d, torch.ones_like(points_3d[..., :1])], dim=-1)
    points_3d_homo = points_3d_homo.unsqueeze(-1)  # (B, N, 4, 1)
    
    # Transform to camera coordinates
    pose_inv = torch.inverse(pose)
    points_cam = torch.matmul(pose_inv.unsqueeze(1), points_3d_homo)[..., :3, 0]
    
    # Project to image plane
    points_cam = points_cam / (points_cam[..., 2:] + 1e-8)
    
    fx = intrinsics[:, 0, 0].unsqueeze(-1).unsqueeze(-1)
    fy = intrinsics[:, 1, 1].unsqueeze(-1).unsqueeze(-1)
    cx = intrinsics[:, 0, 2].unsqueeze(-1).unsqueeze(-1)
    cy = intrinsics[:, 1, 2].unsqueeze(-1).unsqueeze(-1)
    
    u = fx * points_cam[..., 0] + cx
    v = fy * points_cam[..., 1] + cy
    
    # Compute reprojection error
    pred_points = torch.stack([u, v], dim=-1)
    error = torch.norm(pred_points - points_2d, dim=-1)
    
    return error.mean()


def compute_photometric_error(
    img1: torch.Tensor,
    img2: torch.Tensor,
    use_ssim: bool = True,
) -> torch.Tensor:
    """Compute photometric error between two images."""
    if use_ssim:
        # Simplified SSIM + L1 combination
        l1_loss = F.l1_loss(img1, img2)
        # For proper SSIM, use a library like pytorch-msssim
        ssim_loss = l1_loss  # Placeholder
        return 0.85 * ssim_loss + 0.15 * l1_loss
    else:
        return F.l1_loss(img1, img2)


def compute_bundle_adjustment_metrics(
    depths: torch.Tensor,
    poses: torch.Tensor,
    intrinsics: torch.Tensor,
    gt_depths: torch.Tensor,
    gt_poses: torch.Tensor,
) -> Dict[str, float]:
    """Compute bundle adjustment metrics."""
    metrics = {}
    
    # Depth metrics
    depth_metrics = compute_depth_metrics(depths, gt_depths)
    metrics.update({f'depth_{k}': v for k, v in depth_metrics.items()})
    
    # Pose metrics
    pose_metrics = compute_pose_metrics(poses, gt_poses)
    metrics.update({f'pose_{k}': v for k, v in pose_metrics.items()})
    
    # Combined error (weighted sum)
    if 'depth_rmse' in metrics and 'pose_translation_error' in metrics:
        combined_error = 0.5 * metrics['depth_rmse'] + 0.5 * metrics['pose_translation_error']
        metrics['combined_error'] = combined_error
    
    return metrics