"""
Pose subnet trainer for DeepSFM.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import numpy as np

from .trainer import Trainer
from ..losses.pose_losses import PoseLoss
from ..utils.metrics import compute_pose_metrics


class PoseTrainer(Trainer):
    """Trainer for pose estimation subnet."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        config: Dict[str, Any] = None,
        device: torch.device = None,
        log_dir: str = "logs/pose",
        checkpoint_dir: str = "checkpoints/pose",
    ):
        super().__init__(
            model=model,
            train_loader=train_loader,
            val_loader=val_loader,
            config=config,
            device=device,
            log_dir=log_dir,
            checkpoint_dir=checkpoint_dir,
        )
        
        # Pose-specific loss function
        self.loss_fn = PoseLoss(
            translation_weight=config.get('translation_weight', 1.0),
            rotation_weight=config.get('rotation_weight', 1.0),
            geometric_weight=config.get('geometric_weight', 0.5),
        )
    
    def train_step(self, batch: Dict[str, Any], batch_idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Perform a single training step for pose estimation."""
        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.config.get('amp', True)):
            predictions = self.model(
                ref_img=batch['ref_image'],
                src_imgs=batch['src_images'],
                intrinsics=batch['intrinsics'],
                init_pose=batch.get('init_relative_poses'),
                init_depth=batch.get('init_ref_depth'),
            )
            
            # Compute loss
            targets = {
                'translation': self._extract_translation(batch['relative_poses']),
                'rotation': self._extract_rotation(batch['relative_poses']),
                'transform': batch['relative_poses'],
                'relative_poses': batch['relative_poses'],
            }
            
            losses = self.loss_fn(
                predictions, 
                targets,
                depth=batch.get('ref_depth'),
                intrinsics=batch['intrinsics']
            )
        
        # Backward pass
        self.optimizer.zero_grad()
        self.scaler.scale(losses['total']).backward()
        
        # Gradient clipping
        if self.config.get('gradient_clip', 1.0) > 0:
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(
                self.model.parameters(),
                self.config.get('gradient_clip', 1.0)
            )
        
        self.scaler.step(self.optimizer)
        self.scaler.update()
        
        # Convert losses to scalar values for logging
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                    for k, v in losses.items()}
        
        # Compute metrics if ground truth poses are available
        metrics = {}
        if 'relative_poses' in batch and batch['relative_poses'] is not None:
            pred_transform = predictions['transform']
            gt_transform = batch['relative_poses']
            
            pose_metrics = compute_pose_metrics(pred_transform, gt_transform)
            metrics.update(pose_metrics)
        
        return loss_dict, metrics
    
    def val_step(self, batch: Dict[str, Any], batch_idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Perform a single validation step for pose estimation."""
        # Forward pass
        with torch.no_grad():
            predictions = self.model(
                ref_img=batch['ref_image'],
                src_imgs=batch['src_images'],
                intrinsics=batch['intrinsics'],
                init_pose=batch.get('init_relative_poses'),
                init_depth=batch.get('init_ref_depth'),
            )
            
            # Compute loss
            targets = {
                'translation': self._extract_translation(batch['relative_poses']),
                'rotation': self._extract_rotation(batch['relative_poses']),
                'transform': batch['relative_poses'],
                'relative_poses': batch['relative_poses'],
            }
            
            losses = self.loss_fn(
                predictions, 
                targets,
                depth=batch.get('ref_depth'),
                intrinsics=batch['intrinsics']
            )
        
        # Convert losses to scalar values
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                    for k, v in losses.items()}
        
        # Compute metrics if ground truth poses are available
        metrics = {}
        if 'relative_poses' in batch and batch['relative_poses'] is not None:
            pred_transform = predictions['transform']
            gt_transform = batch['relative_poses']
            
            pose_metrics = compute_pose_metrics(pred_transform, gt_transform)
            metrics.update(pose_metrics)
        
        return loss_dict, metrics
    
    def _extract_translation(self, transform: torch.Tensor) -> torch.Tensor:
        """Extract translation from transformation matrix."""
        # transform shape: (B, N, 4, 4) or (B, 4, 4)
        if transform.dim() == 4:
            # Multiple poses: average translation across source views
            translation = transform[:, :, :3, 3]  # (B, N, 3)
            return translation.mean(dim=1)  # (B, 3)
        else:
            # Single pose
            return transform[:, :3, 3]  # (B, 3)
    
    def _extract_rotation(self, transform: torch.Tensor) -> torch.Tensor:
        """Extract rotation in angle-axis format from transformation matrix."""
        # transform shape: (B, N, 4, 4) or (B, 4, 4)
        if transform.dim() == 4:
            # Multiple poses: average rotation across source views
            batch_size, num_poses = transform.shape[:2]
            rotations = []
            
            for i in range(num_poses):
                R = transform[:, i, :3, :3]
                rot = self._matrix_to_angle_axis(R)
                rotations.append(rot)
            
            rotations = torch.stack(rotations, dim=1)  # (B, N, 3)
            return rotations.mean(dim=1)  # (B, 3)
        else:
            # Single pose
            R = transform[:, :3, :3]
            return self._matrix_to_angle_axis(R)
    
    def _matrix_to_angle_axis(self, rotation_matrix: torch.Tensor) -> torch.Tensor:
        """Convert rotation matrix to angle-axis representation."""
        # Simplified conversion
        batch_size = rotation_matrix.shape[0]
        device = rotation_matrix.device
        
        # Trace of rotation matrix
        trace = rotation_matrix[:, 0, 0] + rotation_matrix[:, 1, 1] + rotation_matrix[:, 2, 2]
        
        # Angle
        angle = torch.acos(torch.clamp((trace - 1) / 2, -1, 1))
        
        # Axis
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
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for pose training."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        learning_rate = optimizer_config.get('lr', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 0.0001)
        
        if optimizer_type == 'adam':
            optimizer = torch.optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            optimizer = torch.optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        return optimizer


def create_pose_trainer(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader = None,
    config: Dict[str, Any] = None,
) -> PoseTrainer:
    """Create pose trainer with configuration."""
    default_config = {
        'epochs': 50,
        'optimizer': {
            'type': 'adam',
            'lr': 0.001,
            'weight_decay': 0.0001,
        },
        'scheduler': {
            'type': 'step',
            'step_size': 10,
            'gamma': 0.7,
        },
        'translation_weight': 1.0,
        'rotation_weight': 1.0,
        'geometric_weight': 0.5,
        'amp': True,
        'gradient_clip': 1.0,
        'log_interval': 10,
        'save_interval': 5,
    }
    
    if config:
        # Merge with default config
        for key, value in config.items():
            if isinstance(value, dict) and key in default_config:
                default_config[key].update(value)
            else:
                default_config[key] = value
    
    return PoseTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=default_config,
    )