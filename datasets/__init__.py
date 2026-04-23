"""
DeepSFM datasets package.
"""

from .base import BaseDataset
from .demon_dataset import DeMoNDataset
from .kitti_dataset import KITTIDataset
from .custom_dataset import CustomDataset
from .transforms import (
    RandomCrop,
    RandomScale,
    RandomColorJitter,
    Normalize,
    ToTensor,
    Compose,
)


def get_dataset(name: str):
    """Get dataset class by name.
    
    Args:
        name: Dataset name ('demon', 'kitti', or 'custom')
    
    Returns:
        Dataset class
    
    Raises:
        ValueError: If dataset name is not recognized
    """
    dataset_map = {
        "demon": DeMoNDataset,
        "kitti": KITTIDataset,
        "custom": CustomDataset,
    }
    
    name_lower = name.lower()
    if name_lower not in dataset_map:
        raise ValueError(
            f"Unknown dataset: {name}. "
            f"Available options: {list(dataset_map.keys())}"
        )
    
    return dataset_map[name_lower]


__all__ = [
    "BaseDataset",
    "DeMoNDataset",
    "KITTIDataset",
    "CustomDataset",
    "RandomCrop",
    "RandomScale",
    "RandomColorJitter",
    "Normalize",
    "ToTensor",
    "Compose",
    "get_dataset",
]