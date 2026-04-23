"""
Depth subnet trainer for DeepSFM.
"""

import torch
import torch.nn as nn
from typing import Dict, Any, Tuple
import numpy as np

from .trainer import Trainer
from ..losses.depth_losses import DepthLoss
from ..utils.metrics import compute_depth_metrics


class DepthTrainer(Trainer):
    """Trainer for depth estimation subnet."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader = None,
        config: Dict[str, Any] = None,
        device: torch.device = None,
        log_dir: str = "logs/depth",
        checkpoint_dir: str = "checkpoints/depth",
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
        
        # Depth-specific loss function
        self.loss_fn = DepthLoss(
            depth_weight=config.get('depth_weight', 1.0),
            smoothness_weight=config.get('smoothness_weight', 0.1),
            geometric_weight=config.get('geometric_weight', 0.5),
            multi_scale_weights=config.get('multi_scale_weights', None),
        )
    
    def train_step(self, batch: Dict[str, Any], batch_idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Perform a single training step for depth estimation."""
        # Forward pass
        with torch.cuda.amp.autocast(enabled=self.config.get('amp', True)):
            predictions = self.model(
                ref_img=batch['ref_image'],
                src_imgs=batch['src_images'],
                intrinsics=batch['intrinsics'],
                src_poses=batch['relative_poses'],
                init_depth=batch.get('init_ref_depth'),
                init_pose=batch.get('init_relative_poses'),
            )
            
            # Compute loss
            targets = {
                'depth': batch['ref_depth'],
                'src_images': batch['src_images'],
                'intrinsics': batch['intrinsics'],
                'relative_poses': batch['relative_poses'],
                'ref_image': batch['ref_image'],
            }
            
            masks = {
                'depth_mask': batch.get('valid_depth_mask'),
            }
            
            losses = self.loss_fn(predictions, targets, masks=masks)
        
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
        
        # Compute metrics if ground truth depth is available
        metrics = {}
        if 'ref_depth' in batch and batch['ref_depth'] is not None:
            depth_pred = predictions['depth']
            depth_gt = batch['ref_depth']
            mask = batch.get('valid_depth_mask', torch.ones_like(depth_gt) > 0)
            
            depth_metrics = compute_depth_metrics(depth_pred, depth_gt, mask)
            metrics.update(depth_metrics)
        
        return loss_dict, metrics
    
    def val_step(self, batch: Dict[str, Any], batch_idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Perform a single validation step for depth estimation."""
        # Forward pass
        with torch.no_grad():
            predictions = self.model(
                ref_img=batch['ref_image'],
                src_imgs=batch['src_images'],
                intrinsics=batch['intrinsics'],
                src_poses=batch['relative_poses'],
                init_depth=batch.get('init_ref_depth'),
                init_pose=batch.get('init_relative_poses'),
            )
            
            # Compute loss
            targets = {
                'depth': batch['ref_depth'],
                'src_images': batch['src_images'],
                'intrinsics': batch['intrinsics'],
                'relative_poses': batch['relative_poses'],
                'ref_image': batch['ref_image'],
            }
            
            masks = {
                'depth_mask': batch.get('valid_depth_mask'),
            }
            
            losses = self.loss_fn(predictions, targets, masks=masks)
        
        # Convert losses to scalar values
        loss_dict = {k: v.item() if isinstance(v, torch.Tensor) else v 
                    for k, v in losses.items()}
        
        # Compute metrics if ground truth depth is available
        metrics = {}
        if 'ref_depth' in batch and batch['ref_depth'] is not None:
            depth_pred = predictions['depth']
            depth_gt = batch['ref_depth']
            mask = batch.get('valid_depth_mask', torch.ones_like(depth_gt) > 0)
            
            depth_metrics = compute_depth_metrics(depth_pred, depth_gt, mask)
            metrics.update(depth_metrics)
        
        return loss_dict, metrics
    
    def _create_optimizer(self) -> torch.optim.Optimizer:
        """Create optimizer for depth training."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        learning_rate = optimizer_config.get('lr', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 0.0001)
        
        # Different learning rates for different parts of the model
        if self.config.get('differential_lr', False):
            # Separate parameters for feature extraction and depth prediction
            feature_params = []
            depth_params = []
            
            for name, param in self.model.named_parameters():
                if 'feature_extraction' in name:
                    feature_params.append(param)
                else:
                    depth_params.append(param)
            
            optimizer = torch.optim.Adam([
                {'params': feature_params, 'lr': learning_rate * 0.1},
                {'params': depth_params, 'lr': learning_rate}
            ], weight_decay=weight_decay)
        else:
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


def create_depth_trainer(
    model: nn.Module,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader = None,
    config: Dict[str, Any] = None,
) -> DepthTrainer:
    """Create depth trainer with configuration."""
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
        'depth_weight': 1.0,
        'smoothness_weight': 0.1,
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
    
    return DepthTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=default_config,
    )