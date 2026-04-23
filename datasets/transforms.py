"""
Data transformations for DeepSFM datasets.
"""

import random
from typing import Dict, List, Tuple, Optional, Any
import numpy as np
import torch
import torchvision.transforms.functional as F


class Compose:
    """Compose several transforms together."""
    
    def __init__(self, transforms: List[Any]):
        self.transforms = transforms
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        for transform in self.transforms:
            inputs = transform(inputs)
        return inputs


class RandomCrop:
    """Random crop transformation."""
    
    def __init__(self, size: Tuple[int, int]):
        """
        Args:
            size: Target size (height, width)
        """
        self.size = size
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Get input dimensions
        ref_image = inputs['ref_image']
        if isinstance(ref_image, np.ndarray):
            h, w = ref_image.shape[:2]
        else:
            h, w = ref_image.shape[-2:]
        
        th, tw = self.size
        
        # Random crop coordinates
        if h > th and w > tw:
            i = random.randint(0, h - th)
            j = random.randint(0, w - tw)
        else:
            # If image is smaller than target size, pad instead
            i = 0
            j = 0
        
        # Apply crop to all images and depth maps
        inputs = self._crop_inputs(inputs, i, j, th, tw)
        
        return inputs
    
    def _crop_inputs(
        self,
        inputs: Dict[str, Any],
        i: int,
        j: int,
        h: int,
        w: int,
    ) -> Dict[str, Any]:
        """Apply crop to all inputs."""
        # Crop reference image
        if inputs['ref_image'] is not None:
            inputs['ref_image'] = self._crop_tensor(inputs['ref_image'], i, j, h, w)
        
        # Crop source images
        if inputs['src_images'] is not None:
            inputs['src_images'] = [
                self._crop_tensor(img, i, j, h, w) for img in inputs['src_images']
            ]
        
        # Crop reference depth
        if inputs['ref_depth'] is not None:
            inputs['ref_depth'] = self._crop_tensor(inputs['ref_depth'], i, j, h, w)
        
        # Crop source depths
        if inputs['src_depths'] is not None:
            inputs['src_depths'] = [
                self._crop_tensor(depth, i, j, h, w) for depth in inputs['src_depths']
            ]
        
        # Adjust intrinsics matrix for crop
        if inputs['intrinsics'] is not None:
            intrinsics = inputs['intrinsics'].copy()
            intrinsics[0, 2] -= j  # cx
            intrinsics[1, 2] -= i  # cy
            inputs['intrinsics'] = intrinsics
        
        return inputs
    
    def _crop_tensor(self, tensor: Any, i: int, j: int, h: int, w: int) -> Any:
        """Crop a tensor or numpy array."""
        if isinstance(tensor, np.ndarray):
            if tensor.ndim == 3:
                return tensor[i:i+h, j:j+w, :]
            else:
                return tensor[i:i+h, j:j+w]
        elif isinstance(tensor, torch.Tensor):
            if tensor.dim() == 3:
                return tensor[:, i:i+h, j:j+w]
            else:
                return tensor[i:i+h, j:j+w]
        else:
            raise TypeError(f"Unsupported tensor type: {type(tensor)}")


class RandomScale:
    """Random scaling transformation."""
    
    def __init__(self, scale_range: Tuple[float, float]):
        """
        Args:
            scale_range: Range of scale factors (min, max)
        """
        self.scale_range = scale_range
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Random scale factor
        scale = random.uniform(*self.scale_range)
        
        # Apply scale to all images and depth maps
        inputs = self._scale_inputs(inputs, scale)
        
        return inputs
    
    def _scale_inputs(self, inputs: Dict[str, Any], scale: float) -> Dict[str, Any]:
        """Apply scale to all inputs."""
        # Scale reference image
        if inputs['ref_image'] is not None:
            inputs['ref_image'] = self._scale_tensor(inputs['ref_image'], scale)
        
        # Scale source images
        if inputs['src_images'] is not None:
            inputs['src_images'] = [
                self._scale_tensor(img, scale) for img in inputs['src_images']
            ]
        
        # Scale reference depth
        if inputs['ref_depth'] is not None:
            inputs['ref_depth'] = self._scale_tensor(inputs['ref_depth'], scale)
        
        # Scale source depths
        if inputs['src_depths'] is not None:
            inputs['src_depths'] = [
                self._scale_tensor(depth, scale) for depth in inputs['src_depths']
            ]
        
        # Adjust intrinsics matrix for scale
        if inputs['intrinsics'] is not None:
            intrinsics = inputs['intrinsics'].copy()
            intrinsics[:2, :] *= scale  # fx, fy, cx, cy
            inputs['intrinsics'] = intrinsics
        
        return inputs
    
    def _scale_tensor(self, tensor: Any, scale: float) -> Any:
        """Scale a tensor or numpy array."""
        if isinstance(tensor, np.ndarray):
            import cv2
            if tensor.ndim == 3:
                h, w = tensor.shape[:2]
                new_h, new_w = int(h * scale), int(w * scale)
                return cv2.resize(tensor, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            else:
                h, w = tensor.shape
                new_h, new_w = int(h * scale), int(w * scale)
                return cv2.resize(tensor, (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        elif isinstance(tensor, torch.Tensor):
            # Use torch.nn.functional.interpolate
            import torch.nn.functional as F
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
                tensor = F.interpolate(tensor, scale_factor=scale, mode='bilinear', align_corners=False)
                return tensor.squeeze(0)
            else:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
                tensor = F.interpolate(tensor, scale_factor=scale, mode='nearest')
                return tensor.squeeze(0).squeeze(0)
        else:
            raise TypeError(f"Unsupported tensor type: {type(tensor)}")


class RandomColorJitter:
    """Random color jitter transformation."""
    
    def __init__(
        self,
        brightness: float = 0.2,
        contrast: float = 0.2,
        saturation: float = 0.2,
        hue: float = 0.1,
    ):
        self.brightness = brightness
        self.contrast = contrast
        self.saturation = saturation
        self.hue = hue
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Only apply to images (not depth maps)
        if random.random() < 0.5:
            # Apply same jitter to all images in the sample
            brightness_factor = random.uniform(max(0, 1 - self.brightness), 1 + self.brightness)
            contrast_factor = random.uniform(max(0, 1 - self.contrast), 1 + self.contrast)
            saturation_factor = random.uniform(max(0, 1 - self.saturation), 1 + self.saturation)
            hue_factor = random.uniform(-self.hue, self.hue)
            
            # Apply to reference image
            if inputs['ref_image'] is not None:
                inputs['ref_image'] = self._jitter_tensor(
                    inputs['ref_image'], brightness_factor, contrast_factor,
                    saturation_factor, hue_factor
                )
            
            # Apply to source images
            if inputs['src_images'] is not None:
                inputs['src_images'] = [
                    self._jitter_tensor(
                        img, brightness_factor, contrast_factor,
                        saturation_factor, hue_factor
                    )
                    for img in inputs['src_images']
                ]
        
        return inputs
    
    def _jitter_tensor(
        self,
        tensor: Any,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
    ) -> Any:
        """Apply color jitter to a tensor."""
        if isinstance(tensor, np.ndarray):
            # Convert to tensor, apply jitter, convert back
            tensor_torch = torch.from_numpy(tensor.transpose(2, 0, 1)).float() / 255.0
            tensor_torch = self._jitter_torch(tensor_torch, brightness, contrast, saturation, hue)
            tensor_torch = (tensor_torch * 255.0).clamp(0, 255).byte()
            return tensor_torch.numpy().transpose(1, 2, 0)
        elif isinstance(tensor, torch.Tensor):
            # Assume tensor is in [0, 1] range
            return self._jitter_torch(tensor, brightness, contrast, saturation, hue)
        else:
            raise TypeError(f"Unsupported tensor type: {type(tensor)}")
    
    def _jitter_torch(
        self,
        tensor: torch.Tensor,
        brightness: float,
        contrast: float,
        saturation: float,
        hue: float,
    ) -> torch.Tensor:
        """Apply color jitter to torch tensor."""
        # Brightness
        if brightness > 0:
            tensor = F.adjust_brightness(tensor, brightness)
        
        # Contrast
        if contrast > 0:
            tensor = F.adjust_contrast(tensor, contrast)
        
        # Saturation
        if saturation > 0:
            tensor = F.adjust_saturation(tensor, saturation)
        
        # Hue
        if hue > 0:
            tensor = F.adjust_hue(tensor, hue)
        
        return tensor


class Normalize:
    """Normalize images to [-1, 1] range."""
    
    def __init__(self, mean: Tuple[float, float, float] = (0.5, 0.5, 0.5),
                       std: Tuple[float, float, float] = (0.5, 0.5, 0.5)):
        self.mean = mean
        self.std = std
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Normalize reference image
        if inputs['ref_image'] is not None:
            inputs['ref_image'] = self._normalize_tensor(inputs['ref_image'])
        
        # Normalize source images
        if inputs['src_images'] is not None:
            inputs['src_images'] = [
                self._normalize_tensor(img) for img in inputs['src_images']
            ]
        
        return inputs
    
    def _normalize_tensor(self, tensor: Any) -> Any:
        """Normalize a tensor."""
        if isinstance(tensor, np.ndarray):
            tensor = tensor.astype(np.float32) / 255.0
            tensor = (tensor - np.array(self.mean)) / np.array(self.std)
            return tensor
        elif isinstance(tensor, torch.Tensor):
            # Assume tensor is in [0, 1] range
            tensor = (tensor - torch.tensor(self.mean, device=tensor.device).view(3, 1, 1)) \
                     / torch.tensor(self.std, device=tensor.device).view(3, 1, 1)
            return tensor
        else:
            raise TypeError(f"Unsupported tensor type: {type(tensor)}")


class ToTensor:
    """Convert numpy arrays to torch tensors."""
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Convert reference image
        if inputs['ref_image'] is not None and isinstance(inputs['ref_image'], np.ndarray):
            inputs['ref_image'] = torch.from_numpy(inputs['ref_image'].transpose(2, 0, 1)).float()
        
        # Convert source images
        if inputs['src_images'] is not None:
            inputs['src_images'] = [
                torch.from_numpy(img.transpose(2, 0, 1)).float() if isinstance(img, np.ndarray) else img
                for img in inputs['src_images']
            ]
        
        # Convert reference depth
        if inputs['ref_depth'] is not None and isinstance(inputs['ref_depth'], np.ndarray):
            inputs['ref_depth'] = torch.from_numpy(inputs['ref_depth']).float().unsqueeze(0)
        
        # Convert source depths
        if inputs['src_depths'] is not None:
            inputs['src_depths'] = [
                torch.from_numpy(depth).float().unsqueeze(0) if isinstance(depth, np.ndarray) else depth
                for depth in inputs['src_depths']
            ]
        
        # Convert intrinsics
        if inputs['intrinsics'] is not None and isinstance(inputs['intrinsics'], np.ndarray):
            inputs['intrinsics'] = torch.from_numpy(inputs['intrinsics']).float()
        
        # Convert relative poses
        if inputs['relative_poses'] is not None:
            inputs['relative_poses'] = [
                torch.from_numpy(pose).float() if isinstance(pose, np.ndarray) else pose
                for pose in inputs['relative_poses']
            ]
        
        return inputs


class Resize:
    """Resize images and depth maps to target size."""
    
    def __init__(self, size: Tuple[int, int]):
        """
        Args:
            size: Target size (height, width)
        """
        self.size = size
    
    def __call__(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        # Resize all inputs to target size
        inputs = self._resize_inputs(inputs)
        return inputs
    
    def _resize_inputs(self, inputs: Dict[str, Any]) -> Dict[str, Any]:
        """Resize all inputs to target size."""
        # Resize reference image
        if inputs['ref_image'] is not None:
            inputs['ref_image'] = self._resize_tensor(inputs['ref_image'], is_image=True)
        
        # Resize source images
        if inputs['src_images'] is not None:
            inputs['src_images'] = [
                self._resize_tensor(img, is_image=True) for img in inputs['src_images']
            ]
        
        # Resize reference depth
        if inputs['ref_depth'] is not None:
            inputs['ref_depth'] = self._resize_tensor(inputs['ref_depth'], is_image=False)
        
        # Resize source depths
        if inputs['src_depths'] is not None:
            inputs['src_depths'] = [
                self._resize_tensor(depth, is_image=False) for depth in inputs['src_depths']
            ]
        
        # Adjust intrinsics matrix for resize
        if inputs['intrinsics'] is not None:
            intrinsics = inputs['intrinsics'].copy()
            
            # Get original size from reference image
            if isinstance(inputs['ref_image'], np.ndarray):
                h, w = inputs['ref_image'].shape[:2]
            elif isinstance(inputs['ref_image'], torch.Tensor):
                h, w = inputs['ref_image'].shape[-2:]
            else:
                # Assume original size based on target size and scale
                h, w = self.size
            
            th, tw = self.size
            
            # Scale factors
            scale_h = th / h
            scale_w = tw / w
            
            intrinsics[0, :] *= scale_w  # fx, cx
            intrinsics[1, :] *= scale_h  # fy, cy
            
            inputs['intrinsics'] = intrinsics
        
        return inputs
    
    def _resize_tensor(self, tensor: Any, is_image: bool = True) -> Any:
        """Resize a tensor or numpy array."""
        import torch.nn.functional as F
        import cv2
        
        th, tw = self.size
        
        if isinstance(tensor, np.ndarray):
            if is_image:
                return cv2.resize(tensor, (tw, th), interpolation=cv2.INTER_LINEAR)
            else:
                return cv2.resize(tensor, (tw, th), interpolation=cv2.INTER_NEAREST)
        elif isinstance(tensor, torch.Tensor):
            if tensor.dim() == 3:
                tensor = tensor.unsqueeze(0)
                mode = 'bilinear' if is_image else 'nearest'
                tensor = F.interpolate(tensor, size=(th, tw), mode=mode, align_corners=False if is_image else None)
                return tensor.squeeze(0)
            elif tensor.dim() == 2:
                tensor = tensor.unsqueeze(0).unsqueeze(0)
                tensor = F.interpolate(tensor, size=(th, tw), mode='nearest')
                return tensor.squeeze(0).squeeze(0)
            else:
                raise ValueError(f"Unsupported tensor dimension: {tensor.dim()}")
        else:
            raise TypeError(f"Unsupported tensor type: {type(tensor)}")


def create_train_transforms(
    image_size: Tuple[int, int] = (384, 512),
    scale_range: Tuple[float, float] = (0.8, 1.2),
    use_color_jitter: bool = True,
) -> Compose:
    """Create transformations for training."""
    transforms = [
        RandomScale(scale_range),
        RandomCrop(image_size),
    ]
    
    if use_color_jitter:
        transforms.append(RandomColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1,
        ))
    
    transforms.extend([
        Normalize(),
        ToTensor(),
    ])
    
    return Compose(transforms)


def create_val_transforms(
    image_size: Tuple[int, int] = (384, 512),
) -> Compose:
    """Create transformations for validation."""
    transforms = [
        Resize(image_size),
        Normalize(),
        ToTensor(),
    ]
    
    return Compose(transforms)