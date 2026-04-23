"""
Pose estimation loss functions for DeepSFM.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, Optional, Tuple
import math


class PoseLoss(nn.Module):
    """Combined loss function for pose estimation."""
    
    def __init__(
        self,
        translation_weight: float = 1.0,
        rotation_weight: float = 1.0,
        geometric_weight: float = 0.5,
        use_angular_loss: bool = True,
    ):
        super().__init__()
        
        self.translation_weight = translation_weight
        self.rotation_weight = rotation_weight
        self.geometric_weight = geometric_weight
        self.use_angular_loss = use_angular_loss
        
        # Component losses
        self.translation_loss = TranslationLoss()
        self.rotation_loss = RotationLoss(use_angular=use_angular_loss)
        self.geometric_loss = GeometricConsistencyLoss()
    
    def forward(
        self,
        predictions: Dict[str, torch.Tensor],
        targets: Dict[str, torch.Tensor],
        depth: Optional[torch.Tensor] = None,
        intrinsics: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """Compute pose estimation losses.
        
        Args:
            predictions: Dictionary containing:
                - 'translation': Predicted translation (B, 3)
                - 'rotation': Predicted rotation in angle-axis (B, 3)
                - 'transform': Predicted transformation matrix (B, 4, 4)
                - Other prediction outputs
            targets: Dictionary containing:
                - 'translation': Ground truth translation (B, 3)
                - 'rotation': Ground truth rotation in angle-axis (B, 3)
                - 'transform': Ground truth transformation matrix (B, 4, 4)
                - 'relative_poses': Relative poses for geometric consistency
            depth: Depth map for geometric consistency (optional)
            intrinsics: Camera intrinsics for geometric consistency (optional)
            
        Returns:
            Dictionary of loss values
        """
        losses = {}
        
        # Translation loss
        if 'translation' in predictions and 'translation' in targets:
            trans_loss = self.translation_loss(
                predictions['translation'], targets['translation']
            )
            losses['translation'] = self.translation_weight * trans_loss
        
        # Rotation loss
        if 'rotation' in predictions and 'rotation' in targets:
            rot_loss = self.rotation_loss(
                predictions['rotation'], targets['rotation']
            )
            losses['rotation'] = self.rotation_weight * rot_loss
        
        # Transformation matrix loss (alternative)
        if 'transform' in predictions and 'transform' in targets:
            # Convert transformation matrices to translation and rotation
            pred_trans, pred_rot = self._decompose_transform(predictions['transform'])
            gt_trans, gt_rot = self._decompose_transform(targets['transform'])
            
            trans_loss = self.translation_loss(pred_trans, gt_trans)
            rot_loss = self.rotation_loss(pred_rot, gt_rot)
            
            losses['transform_trans'] = self.translation_weight * trans_loss
            losses['transform_rot'] = self.rotation_weight * rot_loss
        
        # Geometric consistency loss (if depth available)
        if (self.geometric_weight > 0 and depth is not None and 
            intrinsics is not None and 'relative_poses' in targets):
            
            geometric_loss = self.geometric_loss(
                predictions['transform'],
                depth,
                intrinsics,
                targets['relative_poses']
            )
            losses['geometric'] = self.geometric_weight * geometric_loss
        
        # Total loss
        if losses:
            losses['total'] = sum(losses.values())
        
        return losses
    
    def _decompose_transform(self, transform: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Decompose transformation matrix into translation and rotation."""
        # Extract translation
        translation = transform[:, :3, 3]
        
        # Extract rotation matrix and convert to angle-axis
        rotation_matrix = transform[:, :3, :3]
        rotation = self._matrix_to_angle_axis(rotation_matrix)
        
        return translation, rotation
    
    def _matrix_to_angle_axis(self, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to angle-axis representation."""
        batch_size = rotation_matrix.shape[0]
        device = rotation_matrix.device
        
        # Simplified conversion using Rodrigues' formula
        # For exact implementation, use pytorch3d or similar
        
        # Trace of rotation matrix
        trace = rotation_matrix[:, 0, 0] + rotation_matrix[:, 1, 1] + rotation_matrix[:, 2, 2]
        
        # Angle
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
        
        # Axis (unnormalized)
        axis = torch.stack([
            rotation_matrix[:, 2, 1] - rotation_matrix[:, 1, 2],
            rotation_matrix[:, 0, 2] - rotation_matrix[:, 2, 0],
            rotation_matrix[:, 1, 0] - rotation_matrix[:, 0, 1]
        ], dim=1)
        
        # Normalize axis
        axis_norm = torch.norm(axis, dim=1, keepdim=True)
        axis = axis / (axis_norm + 1e-8)
        
        # Angle-axis representation
        rotation = axis * angle.unsqueeze(-1)
        
        return rotation


class TranslationLoss(nn.Module):
    """Translation estimation loss."""
    
    def __init__(self, loss_type: str = 'l1', normalize: bool = True):
        super().__init__()
        self.loss_type = loss_type
        self.normalize = normalize
    
    def forward(
        self,
        pred_translation: torch.Tensor,
        gt_translation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute translation loss.
        
        Args:
            pred_translation: Predicted translation (B, 3)
            gt_translation: Ground truth translation (B, 3)
            
        Returns:
            Translation loss
        """
        if self.normalize:
            # Normalize translations
            pred_norm = torch.norm(pred_translation, dim=1, keepdim=True)
            gt_norm = torch.norm(gt_translation, dim=1, keepdim=True)
            
            pred_normalized = pred_translation / (pred_norm + 1e-8)
            gt_normalized = gt_translation / (gt_norm + 1e-8)
            
            if self.loss_type == 'l1':
                loss = F.l1_loss(pred_normalized, gt_normalized)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(pred_normalized, gt_normalized)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
        else:
            if self.loss_type == 'l1':
                loss = F.l1_loss(pred_translation, gt_translation)
            elif self.loss_type == 'l2':
                loss = F.mse_loss(pred_translation, gt_translation)
            else:
                raise ValueError(f"Unknown loss type: {self.loss_type}")
        
        return loss


class RotationLoss(nn.Module):
    """Rotation estimation loss."""
    
    def __init__(self, use_angular: bool = True, loss_type: str = 'angular'):
        super().__init__()
        self.use_angular = use_angular
        self.loss_type = loss_type
    
    def forward(
        self,
        pred_rotation: torch.Tensor,
        gt_rotation: torch.Tensor,
    ) -> torch.Tensor:
        """Compute rotation loss.
        
        Args:
            pred_rotation: Predicted rotation in angle-axis (B, 3)
            gt_rotation: Ground truth rotation in angle-axis (B, 3)
            
        Returns:
            Rotation loss
        """
        if self.use_angular:
            if self.loss_type == 'angular':
                return self._angular_distance(pred_rotation, gt_rotation)
            elif self.loss_type == 'quaternion':
                return self._quaternion_distance(pred_rotation, gt_rotation)
            else:
                raise ValueError(f"Unknown rotation loss type: {self.loss_type}")
        else:
            # Simple L1/L2 loss on angle-axis vectors
            return F.l1_loss(pred_rotation, gt_rotation)
    
    def _angular_distance(self, rot1: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
        """Compute angular distance between two rotations in angle-axis format."""
        # Convert to rotation matrices
        R1 = self._angle_axis_to_matrix(rot1)
        R2 = self._angle_axis_to_matrix(rot2)
        
        # Compute relative rotation: R_rel = R2^T * R1
        R_rel = torch.bmm(R2.transpose(1, 2), R1)
        
        # Trace of relative rotation
        trace = R_rel[:, 0, 0] + R_rel[:, 1, 1] + R_rel[:, 2, 2]
        
        # Angular distance: acos((trace - 1) / 2)
        angular_distance = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
        
        return angular_distance.mean()
    
    def _quaternion_distance(self, rot1: torch.Tensor, rot2: torch.Tensor) -> torch.Tensor:
        """Compute distance between quaternions."""
        # Convert angle-axis to quaternions
        q1 = self._angle_axis_to_quaternion(rot1)
        q2 = self._angle_axis_to_quaternion(rot2)
        
        # Dot product between quaternions
        dot = torch.abs(torch.sum(q1 * q2, dim=1))
        
        # Angular distance: 2 * acos(|dot|)
        angular_distance = 2 * torch.acos(torch.clamp(dot, -1, 1))
        
        return angular_distance.mean()
    
    def _angle_axis_to_matrix(self, angle_axis: torch.Tensor) -> torch.Tensor:
        """Convert angle-axis to rotation matrix (simplified)."""
        batch_size = angle_axis.shape[0]
        device = angle_axis.device
        
        angle = torch.norm(angle_axis, dim=1, keepdim=True)
        axis = angle_axis / (angle + 1e-8)
        
        # Skew-symmetric matrix
        zero = torch.zeros(batch_size, device=device)
        skew = torch.stack([
            zero, -axis[:, 2], axis[:, 1],
            axis[:, 2], zero, -axis[:, 0],
            -axis[:, 1], axis[:, 0], zero
        ], dim=1).view(batch_size, 3, 3)
        
        # Rodrigues' formula: R = I + sin(theta) * K + (1 - cos(theta)) * K^2
        I = torch.eye(3, device=device).unsqueeze(0).repeat(batch_size, 1, 1)
        sin_theta = torch.sin(angle).unsqueeze(-1)
        cos_theta = torch.cos(angle).unsqueeze(-1)
        
        R = I + sin_theta * skew + (1 - cos_theta) * torch.bmm(skew, skew)
        
        return R
    
    def _angle_axis_to_quaternion(self, angle_axis: torch.Tensor) -> torch.Tensor:
        """Convert angle-axis to quaternion (simplified)."""
        angle = torch.norm(angle_axis, dim=1, keepdim=True)
        axis = angle_axis / (angle + 1e-8)
        
        sin_half_angle = torch.sin(angle / 2)
        cos_half_angle = torch.cos(angle / 2)
        
        q = torch.cat([cos_half_angle, sin_half_angle * axis], dim=1)
        
        return q


class GeometricConsistencyLoss(nn.Module):
    """Geometric consistency loss for pose estimation."""
    
    def forward(
        self,
        pred_transform: torch.Tensor,
        depth: torch.Tensor,
        intrinsics: torch.Tensor,
        gt_relative_poses: torch.Tensor,
    ) -> torch.Tensor:
        """Compute geometric consistency loss for pose.
        
        Measures consistency between depth maps warped using different poses.
        """
        # TODO: Implement geometric consistency loss for pose
        # This should compare reprojection errors using predicted vs ground truth poses
        
        # Placeholder: return zero loss
        return torch.tensor(0.0, device=pred_transform.device)