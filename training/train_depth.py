#!/usr/bin/env python3
"""
Training script for DeepSFM depth subnet (PSNet).
"""

import os
import sys
import yaml
import argparse
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

from models.psnet import PSNet
from datasets import get_dataset
from datasets.transforms import create_train_transforms, create_val_transforms
from training.depth_trainer import DepthTrainer
from configs.main_config import load_config, merge_configs


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Train DeepSFM depth subnet")
    parser.add_argument(
        "--config",
        type=str,
        default="configs/depth_default.yaml",
        help="Path to configuration file"
    )
    parser.add_argument(
        "--resume",
        type=str,
        default=None,
        help="Path to checkpoint to resume training from"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use (cuda or cpu)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        help="Enable debug mode (smaller dataset, fewer epochs)"
    )
    return parser.parse_args()


def create_datasets(config: Dict[str, Any], debug: bool = False):
    """Create training and validation datasets."""
    dataset_name = config["data"]["dataset"]
    data_root = config["data"]["data_root"]
    image_size = tuple(config["data"]["image_size"])
    num_source_views = config["data"]["num_source_views"]
    augmentations = config["data"].get("augmentations", {})
    
    # Get dataset class
    dataset_cls = get_dataset(dataset_name)
    
    # Create transforms for training
    scale_range = augmentations.get("random_scale", [0.8, 1.2])
    use_color_jitter = augmentations.get("random_brightness", 0.0) > 0 or augmentations.get("random_contrast", 0.0) > 0
    
    train_transforms = create_train_transforms(
        image_size=image_size,
        scale_range=tuple(scale_range),
        use_color_jitter=use_color_jitter,
    )
    
    # Create transforms for validation
    val_transforms = create_val_transforms(image_size=image_size)
    
    # Create training dataset
    train_dataset = dataset_cls(
        data_root=data_root,
        split="train",
        image_size=image_size,
        num_source_views=num_source_views,
        transforms=train_transforms,
        load_gt_depth=True,
        load_gt_pose=True,
    )
    
    # Create validation dataset
    val_dataset = dataset_cls(
        data_root=data_root,
        split="val",
        image_size=image_size,
        num_source_views=num_source_views,
        transforms=val_transforms,
        load_gt_depth=True,
        load_gt_pose=True,
    )
    
    # Apply debug mode if needed
    if debug:
        # Limit dataset size for debugging
        train_dataset._samples = train_dataset._samples[:10]
        val_dataset._samples = val_dataset._samples[:5]
    
    return train_dataset, val_dataset


def create_model(config: Dict[str, Any], device: torch.device) -> nn.Module:
    """Create depth subnet model."""
    model_config = config["model"]
    
    model = PSNet(
        min_depth=model_config["min_depth"],
        max_depth=model_config["max_depth"],
        n_depth_bins=model_config["n_depth_bins"],
        feature_channels=model_config["feature_channels"],
        use_geometric_consistency=model_config["use_geometric_consistency"],
    )
    
    # Load checkpoint if provided
    checkpoint_path = model_config.get("checkpoint")
    if checkpoint_path and os.path.exists(checkpoint_path):
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device)
        if "state_dict" in checkpoint:
            model.load_state_dict(checkpoint["state_dict"])
        else:
            model.load_state_dict(checkpoint)
        print("Checkpoint loaded successfully")
    
    return model.to(device)


def main():
    """Main training function."""
    args = parse_args()
    
    # Load configuration
    with open(args.config, "r") as f:
        config = yaml.safe_load(f)
    
    # Apply debug settings if enabled
    if args.debug:
        config["training"]["batch_size"] = 2
        config["training"]["epochs"] = 2
        config["training"]["num_workers"] = 0
        config["data"]["debug"] = True
    
    # Set device
    device = torch.device(args.device)
    print(f"Using device: {device}")
    
    # Create datasets
    print("Creating datasets...")
    train_dataset, val_dataset = create_datasets(config["data"], debug=args.debug)
    
    # Create data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=True,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=True,
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=config["training"]["batch_size"],
        shuffle=False,
        num_workers=config["training"]["num_workers"],
        pin_memory=True,
        drop_last=False,
    )
    
    # Create model
    print("Creating model...")
    model = create_model(config, device)
    
    # Create trainer
    trainer = DepthTrainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        config=config,
        device=device,
        log_dir=config["logging"]["log_dir"],
        checkpoint_dir=config["logging"]["save_dir"],
    )
    
    # Resume training if checkpoint provided
    if args.resume:
        print(f"Resuming training from {args.resume}")
        trainer.load_checkpoint(args.resume)
    
    # Start training
    print("Starting training...")
    trainer.train(
        epochs=config["training"]["epochs"],
        save_frequency=config["logging"]["save_frequency"],
        eval_frequency=config["evaluation"]["eval_frequency"],
    )
    
    print("Training completed!")


if __name__ == "__main__":
    main()