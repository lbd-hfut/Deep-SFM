"""
Base trainer class for DeepSFM.
"""

import os
import time
import logging
from typing import Dict, List, Optional, Any, Tuple
from pathlib import Path

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
import numpy as np

from ..utils.metrics import MetricsTracker
from ..utils.checkpoint import save_checkpoint, load_checkpoint


class Trainer:
    """Base trainer class for DeepSFM models."""
    
    def __init__(
        self,
        model: nn.Module,
        train_loader: DataLoader,
        val_loader: Optional[DataLoader] = None,
        config: Optional[Dict[str, Any]] = None,
        device: Optional[torch.device] = None,
        log_dir: str = "logs",
        checkpoint_dir: str = "checkpoints",
    ):
        """
        Args:
            model: PyTorch model to train
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            config: Training configuration dictionary
            device: Device to use for training
            log_dir: Directory for logging
            checkpoint_dir: Directory for saving checkpoints
        """
        self.model = model
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.config = config or {}
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.log_dir = Path(log_dir)
        self.checkpoint_dir = Path(checkpoint_dir)
        
        # Create directories
        self.log_dir.mkdir(parents=True, exist_ok=True)
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup logging
        self._setup_logging()
        
        # Move model to device
        self.model = self.model.to(self.device)
        
        # Training state
        self.current_epoch = 0
        self.best_metric = float('inf')
        self.metrics_tracker = MetricsTracker()
        
        # TensorBoard writer
        self.writer = SummaryWriter(log_dir=self.log_dir)
        
        # Setup optimizer and scheduler
        self.optimizer = self._create_optimizer()
        self.scheduler = self._create_scheduler()
        
        # Gradient scaler for mixed precision training
        self.scaler = torch.cuda.amp.GradScaler(enabled=self.config.get('amp', True))
        
        # Load checkpoint if specified
        if 'checkpoint' in self.config and self.config['checkpoint']:
            self.load_checkpoint(self.config['checkpoint'])
    
    def _setup_logging(self):
        """Setup logging configuration."""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(self.log_dir / 'training.log'),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(self.__class__.__name__)
    
    def _create_optimizer(self) -> optim.Optimizer:
        """Create optimizer based on config."""
        optimizer_config = self.config.get('optimizer', {})
        optimizer_type = optimizer_config.get('type', 'adam')
        learning_rate = optimizer_config.get('lr', 0.001)
        weight_decay = optimizer_config.get('weight_decay', 0.0001)
        
        if optimizer_type == 'adam':
            optimizer = optim.Adam(
                self.model.parameters(),
                lr=learning_rate,
                weight_decay=weight_decay
            )
        elif optimizer_type == 'sgd':
            momentum = optimizer_config.get('momentum', 0.9)
            optimizer = optim.SGD(
                self.model.parameters(),
                lr=learning_rate,
                momentum=momentum,
                weight_decay=weight_decay
            )
        else:
            raise ValueError(f"Unknown optimizer type: {optimizer_type}")
        
        return optimizer
    
    def _create_scheduler(self) -> Optional[optim.lr_scheduler._LRScheduler]:
        """Create learning rate scheduler based on config."""
        scheduler_config = self.config.get('scheduler', {})
        scheduler_type = scheduler_config.get('type', 'step')
        
        if scheduler_type == 'step':
            step_size = scheduler_config.get('step_size', 10)
            gamma = scheduler_config.get('gamma', 0.7)
            scheduler = optim.lr_scheduler.StepLR(
                self.optimizer,
                step_size=step_size,
                gamma=gamma
            )
        elif scheduler_type == 'cosine':
            epochs = self.config.get('epochs', 50)
            scheduler = optim.lr_scheduler.CosineAnnealingLR(
                self.optimizer,
                T_max=epochs
            )
        elif scheduler_type == 'plateau':
            factor = scheduler_config.get('factor', 0.5)
            patience = scheduler_config.get('patience', 5)
            scheduler = optim.lr_scheduler.ReduceLROnPlateau(
                self.optimizer,
                mode='min',
                factor=factor,
                patience=patience
            )
        else:
            scheduler = None
        
        return scheduler
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch."""
        self.model.train()
        self.metrics_tracker.reset()
        
        epoch_start_time = time.time()
        
        for batch_idx, batch in enumerate(self.train_loader):
            # Move batch to device
            batch = self._move_batch_to_device(batch)
            
            # Training step
            losses, metrics = self.train_step(batch, batch_idx)
            
            # Update metrics
            self.metrics_tracker.update(losses)
            if metrics:
                self.metrics_tracker.update(metrics)
            
            # Log progress
            if batch_idx % self.config.get('log_interval', 10) == 0:
                self.logger.info(
                    f"Epoch {epoch} [{batch_idx}/{len(self.train_loader)}] - "
                    f"Loss: {losses.get('total', 0):.4f}"
                )
        
        # Compute epoch metrics
        epoch_metrics = self.metrics_tracker.compute()
        epoch_metrics['epoch_time'] = time.time() - epoch_start_time
        
        # Log to TensorBoard
        for key, value in epoch_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'train/{key}', value, epoch)
        
        return epoch_metrics
    
    def train_step(self, batch: Dict[str, Any], batch_idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Perform a single training step.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def validate(self, epoch: int) -> Dict[str, float]:
        """Validate the model."""
        if self.val_loader is None:
            return {}
        
        self.model.eval()
        self.metrics_tracker.reset()
        
        val_start_time = time.time()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(self.val_loader):
                # Move batch to device
                batch = self._move_batch_to_device(batch)
                
                # Validation step
                losses, metrics = self.val_step(batch, batch_idx)
                
                # Update metrics
                self.metrics_tracker.update(losses)
                if metrics:
                    self.metrics_tracker.update(metrics)
        
        # Compute validation metrics
        val_metrics = self.metrics_tracker.compute()
        val_metrics['val_time'] = time.time() - val_start_time
        
        # Log to TensorBoard
        for key, value in val_metrics.items():
            if isinstance(value, (int, float)):
                self.writer.add_scalar(f'val/{key}', value, epoch)
        
        return val_metrics
    
    def val_step(self, batch: Dict[str, Any], batch_idx: int) -> Tuple[Dict[str, torch.Tensor], Dict[str, float]]:
        """Perform a single validation step.
        
        Must be implemented by subclasses.
        """
        raise NotImplementedError
    
    def _move_batch_to_device(self, batch: Dict[str, Any]) -> Dict[str, Any]:
        """Move batch tensors to device."""
        device_batch = {}
        for key, value in batch.items():
            if isinstance(value, torch.Tensor):
                device_batch[key] = value.to(self.device)
            elif isinstance(value, dict):
                device_batch[key] = self._move_batch_to_device(value)
            elif isinstance(value, list):
                device_batch[key] = [
                    v.to(self.device) if isinstance(v, torch.Tensor) else v
                    for v in value
                ]
            else:
                device_batch[key] = value
        return device_batch
    
    def train(self, epochs: Optional[int] = None):
        """Main training loop."""
        epochs = epochs or self.config.get('epochs', 50)
        
        self.logger.info(f"Starting training for {epochs} epochs")
        self.logger.info(f"Using device: {self.device}")
        
        for epoch in range(self.current_epoch, epochs):
            self.current_epoch = epoch
            
            # Train for one epoch
            train_metrics = self.train_epoch(epoch)
            
            # Validate
            val_metrics = self.validate(epoch)
            
            # Update learning rate scheduler
            if self.scheduler is not None:
                if isinstance(self.scheduler, optim.lr_scheduler.ReduceLROnPlateau):
                    val_loss = val_metrics.get('total', train_metrics.get('total', 1.0))
                    self.scheduler.step(val_loss)
                else:
                    self.scheduler.step()
            
            # Log epoch results
            self._log_epoch_results(epoch, train_metrics, val_metrics)
            
            # Save checkpoint
            if epoch % self.config.get('save_interval', 5) == 0:
                self.save_checkpoint(f"checkpoint_epoch_{epoch}.pth")
            
            # Save best model
            val_loss = val_metrics.get('total', float('inf'))
            if val_loss < self.best_metric:
                self.best_metric = val_loss
                self.save_checkpoint("best_model.pth")
                self.logger.info(f"New best model saved with validation loss: {val_loss:.4f}")
        
        # Final cleanup
        self.writer.close()
        self.logger.info("Training completed")
    
    def _log_epoch_results(self, epoch: int, train_metrics: Dict[str, float], val_metrics: Dict[str, float]):
        """Log epoch results."""
        log_msg = f"\nEpoch {epoch} Summary:\n"
        log_msg += "Training:\n"
        for key, value in train_metrics.items():
            if isinstance(value, (int, float)):
                log_msg += f"  {key}: {value:.4f}\n"
        
        if val_metrics:
            log_msg += "Validation:\n"
            for key, value in val_metrics.items():
                if isinstance(value, (int, float)):
                    log_msg += f"  {key}: {value:.4f}\n"
        
        self.logger.info(log_msg)
    
    def save_checkpoint(self, filename: str):
        """Save training checkpoint."""
        checkpoint_path = self.checkpoint_dir / filename
        
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler else None,
            'best_metric': self.best_metric,
            'config': self.config,
        }
        
        save_checkpoint(checkpoint, checkpoint_path)
        self.logger.info(f"Checkpoint saved: {checkpoint_path}")
    
    def load_checkpoint(self, checkpoint_path: str):
        """Load training checkpoint."""
        checkpoint = load_checkpoint(checkpoint_path, self.device)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if self.scheduler and checkpoint['scheduler_state_dict']:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_metric = checkpoint['best_metric']
        
        self.logger.info(f"Checkpoint loaded from {checkpoint_path}")
        self.logger.info(f"Resuming from epoch {self.current_epoch}")
    
    def evaluate(self, test_loader: DataLoader) -> Dict[str, float]:
        """Evaluate the model on test data."""
        self.model.eval()
        self.metrics_tracker.reset()
        
        with torch.no_grad():
            for batch_idx, batch in enumerate(test_loader):
                batch = self._move_batch_to_device(batch)
                _, metrics = self.val_step(batch, batch_idx)
                
                if metrics:
                    self.metrics_tracker.update(metrics)
        
        test_metrics = self.metrics_tracker.compute()
        
        self.logger.info("Test Results:")
        for key, value in test_metrics.items():
            if isinstance(value, (int, float)):
                self.logger.info(f"  {key}: {value:.4f}")
        
        return test_metrics