"""
Main configuration module for DeepSFM.
Centralizes all configuration settings for easy access and modification.
"""

from dataclasses import dataclass, field
from typing import List, Tuple, Optional, Dict, Any
import yaml
import os
import torch


@dataclass
class ModelConfig:
    """Configuration for model architecture."""
    # Depth subnet parameters
    min_depth: float = 0.5
    max_depth: float = 10.0
    n_depth_bins: int = 64
    depth_feature_channels: int = 32
    
    # Pose subnet parameters  
    translation_std: float = 0.27
    rotation_std: float = 0.12  # radians
    n_pose_bins: int = 10
    pose_feature_channels: int = 32
    
    # Common parameters
    use_geometric_consistency: bool = True
    checkpoint_path: Optional[str] = None


@dataclass
class DataConfig:
    """Configuration for data loading and processing."""
    dataset: str = "demon"  # demon, kitti, custom
    data_root: str = "data/demon"
    image_size: Tuple[int, int] = (384, 512)  # height, width
    num_source_views: int = 2
    
    # Augmentations
    augmentations: Dict[str, Any] = field(default_factory=lambda: {
        "random_crop": True,
        "random_scale": [0.8, 1.2],
        "random_brightness": 0.2,
        "random_contrast": 0.2,
        "random_horizontal_flip": False  # Generally not used for SfM
    })
    
    # Data splitting
    train_split: float = 0.8
    val_split: float = 0.1
    test_split: float = 0.1


@dataclass
class TrainingConfig:
    """Configuration for training."""
    # General training parameters
    batch_size: int = 4
    num_workers: int = 4
    epochs: int = 50
    learning_rate: float = 0.001
    weight_decay: float = 0.0001
    
    # Learning rate scheduling
    lr_scheduler: str = "step"  # step, cosine, plateau
    lr_step_size: int = 10
    lr_gamma: float = 0.7
    lr_patience: int = 5  # for plateau scheduler
    
    # Optimization
    optimizer: str = "adam"  # adam, sgd
    gradient_clip: float = 1.0
    use_amp: bool = True  # Automatic Mixed Precision
    
    # Depth subnet loss weights
    depth_loss_weight: float = 1.0
    smoothness_weight: float = 0.1
    depth_geometric_weight: float = 0.5
    multi_scale_weights: List[float] = field(default_factory=lambda: [0.5, 0.7, 1.0, 1.0])
    
    # Pose subnet loss weights
    translation_weight: float = 1.0
    rotation_weight: float = 1.0
    pose_geometric_weight: float = 0.5


@dataclass
class LoggingConfig:
    """Configuration for logging and saving."""
    log_dir: str = "logs"
    save_dir: str = "checkpoints"
    save_frequency: int = 5  # Save checkpoint every N epochs
    
    # TensorBoard
    use_tensorboard: bool = True
    tensorboard_dir: str = "runs"
    
    # Console logging
    verbose: bool = True
    print_frequency: int = 10  # Print progress every N batches
    
    # Visualization
    save_visualizations: bool = True
    visualization_dir: str = "visualizations"
    vis_frequency: int = 100  # Save visualizations every N batches


@dataclass
class EvaluationConfig:
    """Configuration for evaluation."""
    metrics: List[str] = field(default_factory=lambda: [
        # Depth metrics
        "abs_rel", "sq_rel", "rmse", "rmse_log",
        "a1", "a2", "a3", "l1_inv", "scale_inv",
        # Pose metrics
        "rotation_error", "translation_error", "translation_angular_error"
    ])
    
    eval_frequency: int = 1  # Evaluate every N epochs
    save_predictions: bool = True
    prediction_dir: str = "predictions"
    
    # Iterative evaluation
    iterative_steps: int = 3  # Number of depth-pose refinement iterations


@dataclass
class Config:
    """Main configuration class."""
    model: ModelConfig = field(default_factory=ModelConfig)
    data: DataConfig = field(default_factory=DataConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    logging: LoggingConfig = field(default_factory=LoggingConfig)
    evaluation: EvaluationConfig = field(default_factory=EvaluationConfig)
    
    # Runtime settings
    device: str = "cuda" if torch.cuda.is_available() else "cpu"
    seed: int = 42
    deterministic: bool = True
    
    def __post_init__(self):
        """Post-initialization setup."""
        import torch
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
    
    @classmethod
    def from_yaml(cls, yaml_path: str) -> "Config":
        """Load configuration from YAML file."""
        with open(yaml_path, 'r') as f:
            config_dict = yaml.safe_load(f)
        
        # Convert nested dictionaries to appropriate dataclasses
        # This is a simplified version - in practice you'd want more robust parsing
        config = cls()
        
        # Update config with values from YAML
        for section, values in config_dict.items():
            if hasattr(config, section):
                section_obj = getattr(config, section)
                for key, value in values.items():
                    if hasattr(section_obj, key):
                        setattr(section_obj, key, value)
        
        return config
    
    def to_yaml(self, yaml_path: str):
        """Save configuration to YAML file."""
        config_dict = {
            "model": self.model.__dict__,
            "data": self.data.__dict__,
            "training": self.training.__dict__,
            "logging": self.logging.__dict__,
            "evaluation": self.evaluation.__dict__,
            "runtime": {
                "device": self.device,
                "seed": self.seed,
                "deterministic": self.deterministic
            }
        }
        
        os.makedirs(os.path.dirname(yaml_path), exist_ok=True)
        with open(yaml_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False)


# Default configuration
default_config = Config()


if __name__ == "__main__":
    # Example usage
    config = Config()
    print(f"Device: {config.device}")
    print(f"Model min depth: {config.model.min_depth}")
    print(f"Training batch size: {config.training.batch_size}")