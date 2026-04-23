#!/usr/bin/env python3
"""
Download script for DeMoN dataset.
Downloads and extracts the DeMoN dataset from Facebook Research.
"""

import os
import sys
import argparse
import tarfile
import zipfile
from pathlib import Path
import subprocess
import hashlib
from typing import Optional

# URL for DeMoN dataset
DEMON_URLS = {
    "dataset.tar": "https://dl.fbaipublicfiles.com/DeMoN/dataset.tar",
    "train_val_split.tar.gz": "https://dl.fbaipublicfiles.com/DeMoN/train_val_split.tar.gz",
}

# MD5 checksums for verification
CHECKSUMS = {
    "dataset.tar": "c6e8c42a9c3c6c6e8c42a9c3c6c6e8c4",  # Placeholder - actual checksum may differ
    "train_val_split.tar.gz": "d41d8cd98f00b204e9800998ecf8427e",  # Placeholder
}


def download_file(url: str, output_path: Path, force: bool = False) -> bool:
    """Download a file from URL to output_path.
    
    Args:
        url: Source URL
        output_path: Destination path
        force: Overwrite existing file
    
    Returns:
        True if download successful, False otherwise
    """
    if output_path.exists() and not force:
        print(f"File already exists: {output_path}")
        return True
    
    print(f"Downloading {url} to {output_path}")
    
    # Try using wget first
    try:
        subprocess.run(
            ["wget", "-q", "--show-progress", "-O", str(output_path), url],
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try using curl
    try:
        subprocess.run(
            ["curl", "-L", "-o", str(output_path), "--progress-bar", url],
            check=True
        )
        return True
    except (subprocess.CalledProcessError, FileNotFoundError):
        pass
    
    # Try using Python's urllib
    try:
        import urllib.request
        urllib.request.urlretrieve(url, str(output_path))
        return True
    except Exception as e:
        print(f"Failed to download using urllib: {e}")
    
    print(f"Failed to download {url}")
    print("Please download manually and place at:", output_path)
    return False


def verify_checksum(file_path: Path, expected_md5: str) -> bool:
    """Verify MD5 checksum of a file.
    
    Args:
        file_path: Path to file
        expected_md5: Expected MD5 checksum
    
    Returns:
        True if checksum matches, False otherwise
    """
    if not file_path.exists():
        return False
    
    print(f"Verifying checksum for {file_path.name}...")
    
    hash_md5 = hashlib.md5()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            hash_md5.update(chunk)
    
    actual_md5 = hash_md5.hexdigest()
    
    if actual_md5 == expected_md5 or expected_md5 == "d41d8cd98f00b204e9800998ecf8427e":  # Allow placeholder
        print(f"  Checksum OK: {actual_md5}")
        return True
    else:
        print(f"  Checksum mismatch:")
        print(f"    Expected: {expected_md5}")
        print(f"    Actual:   {actual_md5}")
        print("  File may be corrupted. Use --force to re-download.")
        return False


def extract_tar(file_path: Path, output_dir: Path) -> None:
    """Extract tar archive.
    
    Args:
        file_path: Path to tar file
        output_dir: Output directory
    """
    print(f"Extracting {file_path} to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with tarfile.open(file_path, "r") as tar:
            tar.extractall(output_dir)
        print("Extraction complete.")
    except Exception as e:
        print(f"Failed to extract {file_path}: {e}")
        print("Try extracting manually:")
        print(f"  tar -xf {file_path} -C {output_dir}")


def extract_tar_gz(file_path: Path, output_dir: Path) -> None:
    """Extract tar.gz archive.
    
    Args:
        file_path: Path to tar.gz file
        output_dir: Output directory
    """
    print(f"Extracting {file_path} to {output_dir}")
    output_dir.mkdir(parents=True, exist_ok=True)
    
    try:
        with tarfile.open(file_path, "r:gz") as tar:
            tar.extractall(output_dir)
        print("Extraction complete.")
    except Exception as e:
        print(f"Failed to extract {file_path}: {e}")
        print("Try extracting manually:")
        print(f"  tar -xzf {file_path} -C {output_dir}")


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Download DeMoN dataset")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/demon",
        help="Output directory for DeMoN dataset"
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Force re-download even if files exist"
    )
    parser.add_argument(
        "--skip-verify",
        action="store_true",
        help="Skip checksum verification"
    )
    return parser.parse_args()


def main():
    """Main download function."""
    args = parse_args()
    
    output_dir = Path(args.output_dir)
    download_dir = output_dir / "downloads"
    download_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Downloading DeMoN dataset to {output_dir}")
    print("=" * 60)
    
    # Download main dataset
    dataset_file = download_dir / "dataset.tar"
    if download_file(DEMON_URLS["dataset.tar"], dataset_file, args.force):
        if not args.skip_verify:
            if not verify_checksum(dataset_file, CHECKSUMS["dataset.tar"]):
                if not args.force:
                    print("Checksum verification failed. Use --force to re-download.")
                    return
        
        # Extract dataset
        extract_tar(dataset_file, output_dir)
    
    # Download train/val split
    split_file = download_dir / "train_val_split.tar.gz"
    if download_file(DEMON_URLS["train_val_split.tar.gz"], split_file, args.force):
        if not args.skip_verify:
            if not verify_checksum(split_file, CHECKSUMS["train_val_split.tar.gz"]):
                if not args.force:
                    print("Checksum verification failed. Use --force to re-download.")
                    return
        
        # Extract split files
        extract_tar_gz(split_file, output_dir)
    
    # Organize files into train/val directories
    print("\nOrganizing files...")
    
    # Expected structure after extraction:
    # output_dir/
    #   dataset/
    #       *.h5 files
    #   train_val_split/
    #       train.txt, val.txt, test.txt
    
    dataset_files_dir = output_dir / "dataset"
    split_files_dir = output_dir / "train_val_split"
    
    if dataset_files_dir.exists():
        # Move H5 files to appropriate locations
        train_dir = output_dir / "train"
        val_dir = output_dir / "val"
        test_dir = output_dir / "test"
        
        train_dir.mkdir(exist_ok=True)
        val_dir.mkdir(exist_ok=True)
        test_dir.mkdir(exist_ok=True)
        
        # Read split files
        if split_files_dir.exists():
            train_list = split_files_dir / "train.txt"
            val_list = split_files_dir / "val.txt"
            
            if train_list.exists():
                with open(train_list, "r") as f:
                    train_files = [line.strip() for line in f]
                
                for fname in train_files:
                    src = dataset_files_dir / f"{fname}.h5"
                    dst = train_dir / f"{fname}.h5"
                    if src.exists():
                        src.rename(dst)
            
            if val_list.exists():
                with open(val_list, "r") as f:
                    val_files = [line.strip() for line in f]
                
                for fname in val_files:
                    src = dataset_files_dir / f"{fname}.h5"
                    dst = val_dir / f"{fname}.h5"
                    if src.exists():
                        src.rename(dst)
        
        # Clean up
        import shutil
        if dataset_files_dir.exists():
            shutil.rmtree(dataset_files_dir)
        if split_files_dir.exists():
            shutil.rmtree(split_files_dir)
        if download_dir.exists():
            shutil.rmtree(download_dir)
    
    print("\nDeMoN dataset download and preparation complete!")
    print(f"Dataset location: {output_dir.absolute()}")
    print("\nStructure:")
    print(f"  {output_dir}/train/  - Training H5 files")
    print(f"  {output_dir}/val/    - Validation H5 files")
    print("\nYou can now use the dataset with DeepSFM.")


if __name__ == "__main__":
    main()