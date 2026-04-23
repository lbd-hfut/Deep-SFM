#!/usr/bin/env python3
"""
Data preparation script for DeepSFM datasets.
Supports DeMoN, KITTI, and custom datasets.
"""

import os
import sys
import argparse
import subprocess
from pathlib import Path
from typing import Optional

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))


def prepare_demon_data(data_root: Path, force_download: bool = False) -> None:
    """Prepare DeMoN dataset.
    
    Args:
        data_root: Root directory for DeMoN data
        force_download: Force re-download even if files exist
    """
    print(f"Preparing DeMoN dataset in {data_root}")
    data_root.mkdir(parents=True, exist_ok=True)
    
    # Check if data already exists
    train_h5 = data_root / "train" / "train.h5"
    val_h5 = data_root / "val" / "val.h5"
    
    if train_h5.exists() and val_h5.exists() and not force_download:
        print("DeMoN data already exists. Use --force-download to re-download.")
        return
    
    # Download script location
    download_script = Path(__file__).parent / "download_demon_data.py"
    
    if download_script.exists():
        print(f"Running download script: {download_script}")
        subprocess.run([sys.executable, str(download_script), "--output_dir", str(data_root)], check=True)
    else:
        print(f"Download script not found at {download_script}")
        print("Please download DeMoN dataset manually and place it in the following structure:")
        print(f"  {data_root}/train/train.h5")
        print(f"  {data_root}/val/val.h5")
        print("You can download from: https://dl.fbaipublicfiles.com/DeMoN/dataset.tar")
        print("Or follow instructions at: https://github.com/facebookresearch/DeMoN")
    
    print("DeMoN dataset preparation complete.")


def prepare_kitti_data(data_root: Path, force_download: bool = False) -> None:
    """Prepare KITTI dataset for depth estimation.
    
    Args:
        data_root: Root directory for KITTI data
        force_download: Force re-download even if files exist
    """
    print(f"Preparing KITTI dataset in {data_root}")
    data_root.mkdir(parents=True, exist_ok=True)
    
    # Check for required files
    required_files = [
        "data_depth_annotated.zip",
        "data_depth_velodyne.zip",
        "data_depth_selection.zip",
    ]
    
    missing_files = []
    for f in required_files:
        if not (data_root / f).exists():
            missing_files.append(f)
    
    if missing_files and not force_download:
        print(f"Missing KITTI files: {missing_files}")
        print("Please download KITTI depth dataset manually:")
        print("  http://www.cvlibs.net/datasets/kitti/eval_depth.php")
        print("Required files:")
        for f in required_files:
            print(f"    {f}")
        print(f"Place them in: {data_root}")
        print("Then run this script again with --force-download to extract.")
        return
    
    # Extract files if needed
    print("KITTI dataset preparation requires manual downloading.")
    print("After downloading, extract the files and organize them as follows:")
    print(f"  {data_root}/train/")
    print(f"  {data_root}/val/")
    print("For detailed instructions, see the datasets/kitti_dataset.py file.")
    
    print("KITTI dataset preparation complete.")


def prepare_custom_data(data_root: Path, config_path: Optional[Path] = None) -> None:
    """Prepare custom dataset.
    
    Args:
        data_root: Root directory for custom data
        config_path: Path to custom dataset configuration file
    """
    print(f"Preparing custom dataset in {data_root}")
    data_root.mkdir(parents=True, exist_ok=True)
    
    if config_path:
        print(f"Using configuration from {config_path}")
        # Load and validate configuration
        # This would typically parse a YAML/JSON config for custom dataset structure
    else:
        print("No configuration provided. Creating default directory structure.")
    
    # Create default directory structure
    splits = ["train", "val", "test"]
    for split in splits:
        split_dir = data_root / split
        split_dir.mkdir(exist_ok=True)
        
        # Create placeholder files
        (split_dir / "images").mkdir(exist_ok=True)
        (split_dir / "depths").mkdir(exist_ok=True)
        (split_dir / "poses").mkdir(exist_ok=True)
        (split_dir / "intrinsics").mkdir(exist_ok=True)
    
    print("Created directory structure for custom dataset.")
    print("Please place your data in the following structure:")
    print("  data_root/train/images/*.png")
    print("  data_root/train/depths/*.npy")
    print("  data_root/train/poses/*.txt")
    print("  data_root/train/intrinsics/*.txt")
    print("See datasets/custom_dataset.py for format details.")
    
    print("Custom dataset preparation complete.")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Prepare datasets for DeepSFM")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        choices=["demon", "kitti", "custom"],
        help="Dataset to prepare"
    )
    parser.add_argument(
        "--data-root",
        type=str,
        default=None,
        help="Root directory for dataset (default: data/<dataset>)"
    )
    parser.add_argument(
        "--force-download",
        action="store_true",
        help="Force re-download of dataset files"
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="Path to configuration file (for custom datasets)"
    )
    return parser.parse_args()


def main():
    """Main data preparation function."""
    args = parse_args()
    
    # Determine data root
    if args.data_root:
        data_root = Path(args.data_root)
    else:
        data_root = Path("data") / args.dataset
    
    # Prepare dataset based on type
    if args.dataset == "demon":
        prepare_demon_data(data_root, args.force_download)
    elif args.dataset == "kitti":
        prepare_kitti_data(data_root, args.force_download)
    elif args.dataset == "custom":
        config_path = Path(args.config) if args.config else None
        prepare_custom_data(data_root, config_path)
    else:
        raise ValueError(f"Unknown dataset: {args.dataset}")
    
    print(f"\nDataset preparation complete for {args.dataset}.")
    print(f"Data located at: {data_root.absolute()}")
    print("\nNext steps:")
    print("1. Verify the data structure is correct")
    print("2. Run training with:")
    print(f"   python training/train_depth.py --config configs/depth_default.yaml")
    print(f"   python training/train_pose.py --config configs/pose_default.yaml")


if __name__ == "__main__":
    main()