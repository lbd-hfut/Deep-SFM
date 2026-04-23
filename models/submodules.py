"""
Shared building blocks for DeepSFM models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


def convbn(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    """2D convolution with batch normalization and ReLU."""
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True)
    )


def convbn_3d(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> nn.Sequential:
    """3D convolution with batch normalization and ReLU."""
    return nn.Sequential(
        nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False),
        nn.BatchNorm3d(out_channels),
        nn.ReLU(inplace=True)
    )


def conv_3d(in_channels: int, out_channels: int, kernel_size: int = 3, stride: int = 1, padding: int = 1) -> nn.Conv3d:
    """3D convolution without batch norm."""
    return nn.Conv3d(in_channels, out_channels, kernel_size=kernel_size, stride=stride, padding=padding, bias=False)


class BasicBlock(nn.Module):
    """Basic residual block for feature extraction."""
    
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1):
        super().__init__()
        
        self.conv1 = convbn(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.downsample = None
        if stride != 1 or in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(out_channels)
            )
        
        self.relu = nn.ReLU(inplace=True)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.relu(out)
        
        return out


class FeatureExtraction(nn.Module):
    """Multi-scale feature extraction with pooling branches."""
    
    def __init__(self, in_channels: int = 3, feature_channels: int = 32):
        super().__init__()
        
        # Initial convolution
        self.in_conv = convbn(in_channels, feature_channels, kernel_size=3, stride=2, padding=1)
        
        # Residual blocks
        self.block1 = BasicBlock(feature_channels, feature_channels)
        self.block2 = BasicBlock(feature_channels, feature_channels)
        self.block3 = BasicBlock(feature_channels, feature_channels)
        
        # Pooling branches for multi-scale features
        self.pool1 = nn.AvgPool2d(2, stride=2)
        self.pool2 = nn.AvgPool2d(4, stride=4)
        self.pool3 = nn.AvgPool2d(8, stride=8)
        
        # Output convolutions for each scale
        self.out_conv1 = convbn(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1)
        self.out_conv2 = convbn(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1)
        self.out_conv3 = convbn(feature_channels, feature_channels, kernel_size=3, stride=1, padding=1)
    
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Extract features at multiple scales.
        
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Tuple of features at scales: full, 1/2, 1/4, 1/8
        """
        # Initial feature extraction
        x = self.in_conv(x)
        
        # Process through residual blocks
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        
        # Full resolution features
        feat_full = self.out_conv1(x)
        
        # Half resolution
        feat_half = self.pool1(x)
        feat_half = self.out_conv2(feat_half)
        
        # Quarter resolution
        feat_quarter = self.pool2(x)
        feat_quarter = self.out_conv3(feat_quarter)
        
        # Eighth resolution
        feat_eighth = self.pool3(x)
        
        return feat_full, feat_half, feat_quarter, feat_eighth


class DisparityRegression(nn.Module):
    """Soft argmax regression for depth/disparity from probability volume."""
    
    def __init__(self, max_disparity: int):
        super().__init__()
        self.max_disparity = max_disparity
        
        # Create disparity/disparity cost volume
        self.register_buffer('disparity', torch.arange(0, max_disparity, dtype=torch.float32).view(1, -1, 1, 1))
    
    def forward(self, prob_volume: torch.Tensor) -> torch.Tensor:
        """Compute disparity/depth from probability volume using soft argmax.
        
        Args:
            prob_volume: Probability volume of shape (B, D, H, W)
            
        Returns:
            Disparity/depth map of shape (B, 1, H, W)
        """
        # Soft argmax: sum(disparity * softmax(prob_volume))
        prob_volume = F.softmax(prob_volume, dim=1)
        disparity = torch.sum(prob_volume * self.disparity, dim=1, keepdim=True)
        return disparity


class PoseRegression(nn.Module):
    """Soft argmax regression for pose from probability volume."""
    
    def __init__(self, n_bins: int, std_tr: float, std_rot: float):
        super().__init__()
        self.n_bins = n_bins
        self.std_tr = std_tr
        self.std_rot = std_rot
        
        # Create sampling grid for translation and rotation
        self._create_sampling_grid()
    
    def _create_sampling_grid(self):
        """Create sampling grid for translation and rotation dimensions."""
        # Translation samples: uniform in [-std_tr, std_tr]
        trans_samples = torch.linspace(-self.std_tr, self.std_tr, self.n_bins)
        
        # Rotation samples: uniform in [-std_rot, std_rot] (radians)
        rot_samples = torch.linspace(-self.std_rot, self.std_rot, self.n_bins)
        
        # Create 3D grid for translation (tx, ty, tz)
        trans_grid = torch.stack(torch.meshgrid(
            trans_samples, trans_samples, trans_samples, indexing='ij'
        ), dim=-1).reshape(-1, 3)
        
        # Create 3D grid for rotation (rx, ry, rz)
        rot_grid = torch.stack(torch.meshgrid(
            rot_samples, rot_samples, rot_samples, indexing='ij'
        ), dim=-1).reshape(-1, 3)
        
        self.register_buffer('trans_grid', trans_grid)
        self.register_buffer('rot_grid', rot_grid)
    
    def forward(self, trans_prob: torch.Tensor, rot_prob: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Compute translation and rotation from probability volumes.
        
        Args:
            trans_prob: Translation probability volume of shape (B, N^3, ...)
            rot_prob: Rotation probability volume of shape (B, N^3, ...)
            
        Returns:
            Tuple of (translation, rotation) tensors
        """
        batch_size = trans_prob.shape[0]
        
        # Reshape probability volumes to (B, N^3)
        trans_prob = trans_prob.view(batch_size, -1)
        rot_prob = rot_prob.view(batch_size, -1)
        
        # Apply softmax
        trans_prob = F.softmax(trans_prob, dim=1)
        rot_prob = F.softmax(rot_prob, dim=1)
        
        # Weighted sum of samples
        translation = torch.sum(trans_prob.unsqueeze(-1) * self.trans_grid.unsqueeze(0), dim=1)
        rotation = torch.sum(rot_prob.unsqueeze(-1) * self.rot_grid.unsqueeze(0), dim=1)
        
        return translation, rotation


class CostVolume3D(nn.Module):
    """3D cost volume construction for feature matching."""
    
    def __init__(self, feature_channels: int = 32, max_disparity: int = 64):
        super().__init__()
        self.feature_channels = feature_channels
        self.max_disparity = max_disparity
        
        # Cost volume regularization network
        self.conv0 = convbn_3d(feature_channels * 2, feature_channels)
        self.conv1 = convbn_3d(feature_channels, feature_channels * 2, stride=2)
        self.conv2 = convbn_3d(feature_channels * 2, feature_channels * 2)
        self.conv3 = convbn_3d(feature_channels * 2, feature_channels * 4, stride=2)
        self.conv4 = convbn_3d(feature_channels * 4, feature_channels * 4)
        self.conv5 = convbn_3d(feature_channels * 4, feature_channels * 8, stride=2)
        self.conv6 = convbn_3d(feature_channels * 8, feature_channels * 8)
        self.conv7 = nn.Sequential(
            nn.ConvTranspose3d(feature_channels * 8, feature_channels * 4, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(feature_channels * 4),
            nn.ReLU(inplace=True)
        )
        self.conv8 = nn.Sequential(
            nn.ConvTranspose3d(feature_channels * 4, feature_channels * 2, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(feature_channels * 2),
            nn.ReLU(inplace=True)
        )
        self.conv9 = nn.Sequential(
            nn.ConvTranspose3d(feature_channels * 2, feature_channels, kernel_size=3, padding=1, output_padding=1, stride=2, bias=False),
            nn.BatchNorm3d(feature_channels),
            nn.ReLU(inplace=True)
        )
        
        # Probability volume prediction
        self.prob_conv = nn.Conv3d(feature_channels, 1, kernel_size=3, stride=1, padding=1, bias=False)
    
    def forward(self, cost_volume: torch.Tensor) -> torch.Tensor:
        """Regularize cost volume and predict probability volume.
        
        Args:
            cost_volume: Initial cost volume of shape (B, C*2, D, H, W)
            
        Returns:
            Probability volume of shape (B, D, H, W)
        """
        # Hourglass-style 3D CNN
        out = self.conv0(cost_volume)
        out = self.conv1(out)
        out = self.conv2(out)
        out = self.conv3(out)
        out = self.conv4(out)
        out = self.conv5(out)
        out = self.conv6(out)
        out = self.conv7(out)
        out = self.conv8(out)
        out = self.conv9(out)
        
        # Probability volume
        prob_volume = self.prob_conv(out)
        prob_volume = prob_volume.squeeze(1)  # Remove channel dimension
        
        return prob_volume


def inverse_warp(
    src_image: torch.Tensor,
    depth: torch.Tensor,
    intrinsics: torch.Tensor,
    relative_pose: torch.Tensor,
) -> torch.Tensor:
    """Differentiable inverse warping of source image to reference view.
    
    Args:
        src_image: Source image tensor (B, C, H, W)
        depth: Depth map in reference view (B, 1, H, W)
        intrinsics: Camera intrinsics matrix (B, 3, 3)
        relative_pose: Relative pose from reference to source (B, 4, 4)
    
    Returns:
        Warped source image in reference view (B, C, H, W)
    """
    batch_size, channels, height, width = src_image.shape
    
    # Create grid of pixel coordinates in reference view
    y_coords, x_coords = torch.meshgrid(
        torch.arange(height, device=src_image.device, dtype=torch.float32),
        torch.arange(width, device=src_image.device, dtype=torch.float32),
        indexing='ij'
    )
    
    # Normalize coordinates to [-1, 1] for grid_sample
    x_coords = 2.0 * x_coords / (width - 1) - 1.0
    y_coords = 2.0 * y_coords / (height - 1) - 1.0
    
    # Create homogeneous pixel coordinates (B, 3, H*W)
    ones = torch.ones_like(x_coords)
    pixel_coords = torch.stack([x_coords, y_coords, ones], dim=0)  # (3, H, W)
    pixel_coords = pixel_coords.unsqueeze(0).repeat(batch_size, 1, 1, 1)  # (B, 3, H, W)
    pixel_coords = pixel_coords.reshape(batch_size, 3, -1)  # (B, 3, H*W)
    
    # Project pixels to 3D space using depth
    intrinsics_inv = torch.inverse(intrinsics)
    cam_coords = torch.matmul(intrinsics_inv, pixel_coords)  # (B, 3, H*W)
    cam_coords = cam_coords * depth.reshape(batch_size, 1, -1)
    
    # Transform to source camera coordinates
    rotation = relative_pose[:, :3, :3]  # (B, 3, 3)
    translation = relative_pose[:, :3, 3:4]  # (B, 3, 1)
    src_cam_coords = torch.matmul(rotation, cam_coords) + translation  # (B, 3, H*W)
    
    # Project back to source image plane
    src_pixel_coords = torch.matmul(intrinsics, src_cam_coords)  # (B, 3, H*W)
    
    # Normalize by depth (perspective division)
    src_depth = src_pixel_coords[:, 2:3, :]  # (B, 1, H*W)
    src_pixel_coords = src_pixel_coords[:, :2, :] / (src_depth + 1e-7)  # (B, 2, H*W)
    
    # Reshape back to image dimensions and normalize to [-1, 1] for grid_sample
    src_pixel_coords = src_pixel_coords.reshape(batch_size, 2, height, width)
    
    # Normalize coordinates to [-1, 1] (grid_sample expects this)
    src_pixel_coords[:, 0, :, :] = 2.0 * src_pixel_coords[:, 0, :, :] / (width - 1) - 1.0
    src_pixel_coords[:, 1, :, :] = 2.0 * src_pixel_coords[:, 1, :, :] / (height - 1) - 1.0
    
    # Permute to (B, H, W, 2) for grid_sample
    src_pixel_coords = src_pixel_coords.permute(0, 2, 3, 1)
    
    # Sample source image using bilinear interpolation
    warped_image = F.grid_sample(
        src_image,
        src_pixel_coords,
        mode='bilinear',
        padding_mode='zeros',
        align_corners=False
    )
    
    # Create validity mask (pixels that project inside source image)
    valid_mask = (
        (src_pixel_coords[:, :, :, 0] >= -1.0) &
        (src_pixel_coords[:, :, :, 0] <= 1.0) &
        (src_pixel_coords[:, :, :, 1] >= -1.0) &
        (src_pixel_coords[:, :, :, 1] <= 1.0) &
        (src_depth.reshape(batch_size, 1, height, width) > 0)
    ).float()
    
    # Apply validity mask (set invalid pixels to 0)
    warped_image = warped_image * valid_mask.unsqueeze(1)
    
    return warped_image, valid_mask