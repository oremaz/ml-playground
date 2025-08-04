import cv2
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple, Union
from dataclasses import dataclass
import random
from PIL import Image, ImageEnhance, ImageFilter
import albumentations as A
from albumentations.pytorch import ToTensorV2

@dataclass
class ImageAugmentationConfig:
    """Configuration for image augmentation."""
    rotation_range: Tuple[int, int] = (-30, 30)
    width_shift_range: float = 0.1
    height_shift_range: float = 0.1
    zoom_range: Tuple[float, float] = (0.8, 1.2)
    horizontal_flip: bool = True
    vertical_flip: bool = False
    brightness_range: Tuple[float, float] = (0.8, 1.2)
    contrast_range: Tuple[float, float] = (0.8, 1.2)
    saturation_range: Tuple[float, float] = (0.8, 1.2)
    hue_range: Tuple[float, float] = (-0.1, 0.1)
    noise_factor: float = 0.05
    blur_factor: float = 0.5
    elastic_alpha: float = 1.0
    elastic_sigma: float = 50.0
    grid_distortion: float = 0.1
    optical_distortion: float = 0.1
    random_state: Optional[int] = None

class ClassicalImageAugmenter:
    """
    Classical image augmentation techniques using OpenCV and PIL.
    """
    
    def __init__(self, config: Optional[ImageAugmentationConfig] = None):
        """Initialize the classical image augmenter."""
        self.config = config or ImageAugmentationConfig()
        if self.config.random_state is not None:
            random.seed(self.config.random_state)
            np.random.seed(self.config.random_state)
    
    def _ensure_numpy(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Ensure image is in numpy format."""
        if isinstance(image, Image.Image):
            return np.array(image)
        return image
    
    def _ensure_pil(self, image: Union[np.ndarray, Image.Image]) -> Image.Image:
        """Ensure image is in PIL format."""
        if isinstance(image, np.ndarray):
            return Image.fromarray(image)
        return image
    
    def rotate(self, image: Union[np.ndarray, Image.Image], angle: Optional[float] = None) -> np.ndarray:
        """
        Rotate image by a random angle within the specified range.
        
        Args:
            image: Input image
            angle: Rotation angle in degrees. If None, uses random angle from config.
            
        Returns:
            Rotated image
        """
        image = self._ensure_numpy(image)
        if angle is None:
            angle = random.uniform(self.config.rotation_range[0], self.config.rotation_range[1])
        
        height, width = image.shape[:2]
        center = (width // 2, height // 2)
        
        rotation_matrix = cv2.getRotationMatrix2D(center, angle, 1.0)
        rotated = cv2.warpAffine(image, rotation_matrix, (width, height), 
                                flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        
        return rotated
    
    def translate(self, image: Union[np.ndarray, Image.Image], 
                 dx: Optional[float] = None, dy: Optional[float] = None) -> np.ndarray:
        """
        Translate image by random amounts.
        
        Args:
            image: Input image
            dx: Horizontal translation. If None, uses random value from config.
            dy: Vertical translation. If None, uses random value from config.
            
        Returns:
            Translated image
        """
        image = self._ensure_numpy(image)
        height, width = image.shape[:2]
        
        if dx is None:
            dx = random.uniform(-self.config.width_shift_range, self.config.width_shift_range) * width
        if dy is None:
            dy = random.uniform(-self.config.height_shift_range, self.config.height_shift_range) * height
        
        translation_matrix = np.float32([[1, 0, dx], [0, 1, dy]])
        translated = cv2.warpAffine(image, translation_matrix, (width, height),
                                   borderMode=cv2.BORDER_REFLECT)
        
        return translated
    
    def scale(self, image: Union[np.ndarray, Image.Image], scale_factor: Optional[float] = None) -> np.ndarray:
        """
        Scale image by a random factor.
        
        Args:
            image: Input image
            scale_factor: Scale factor. If None, uses random value from config.
            
        Returns:
            Scaled image
        """
        image = self._ensure_numpy(image)
        if scale_factor is None:
            scale_factor = random.uniform(self.config.zoom_range[0], self.config.zoom_range[1])
        
        height, width = image.shape[:2]
        new_height, new_width = int(height * scale_factor), int(width * scale_factor)
        
        scaled = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        
        # If scaled image is smaller, pad it; if larger, crop it
        if scale_factor < 1:
            # Pad with reflection
            pad_top = (height - new_height) // 2
            pad_bottom = height - new_height - pad_top
            pad_left = (width - new_width) // 2
            pad_right = width - new_width - pad_left
            
            scaled = cv2.copyMakeBorder(scaled, pad_top, pad_bottom, pad_left, pad_right,
                                       cv2.BORDER_REFLECT)
        else:
            # Crop from center
            start_y = (new_height - height) // 2
            start_x = (new_width - width) // 2
            scaled = scaled[start_y:start_y + height, start_x:start_x + width]
        
        return scaled
    
    def flip_horizontal(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Flip image horizontally."""
        image = self._ensure_numpy(image)
        return cv2.flip(image, 1)
    
    def flip_vertical(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """Flip image vertically."""
        image = self._ensure_numpy(image)
        return cv2.flip(image, 0)
    
    def adjust_brightness(self, image: Union[np.ndarray, Image.Image], 
                         factor: Optional[float] = None) -> np.ndarray:
        """
        Adjust image brightness.
        
        Args:
            image: Input image
            factor: Brightness factor. If None, uses random value from config.
            
        Returns:
            Brightness-adjusted image
        """
        image = self._ensure_pil(image)
        if factor is None:
            factor = random.uniform(self.config.brightness_range[0], self.config.brightness_range[1])
        
        enhancer = ImageEnhance.Brightness(image)
        adjusted = enhancer.enhance(factor)
        return np.array(adjusted)
    
    def adjust_contrast(self, image: Union[np.ndarray, Image.Image], 
                       factor: Optional[float] = None) -> np.ndarray:
        """
        Adjust image contrast.
        
        Args:
            image: Input image
            factor: Contrast factor. If None, uses random value from config.
            
        Returns:
            Contrast-adjusted image
        """
        image = self._ensure_pil(image)
        if factor is None:
            factor = random.uniform(self.config.contrast_range[0], self.config.contrast_range[1])
        
        enhancer = ImageEnhance.Contrast(image)
        adjusted = enhancer.enhance(factor)
        return np.array(adjusted)
    
    def adjust_saturation(self, image: Union[np.ndarray, Image.Image], 
                         factor: Optional[float] = None) -> np.ndarray:
        """
        Adjust image saturation.
        
        Args:
            image: Input image
            factor: Saturation factor. If None, uses random value from config.
            
        Returns:
            Saturation-adjusted image
        """
        image = self._ensure_pil(image)
        if factor is None:
            factor = random.uniform(self.config.saturation_range[0], self.config.saturation_range[1])
        
        enhancer = ImageEnhance.Color(image)
        adjusted = enhancer.enhance(factor)
        return np.array(adjusted)
    
    def add_noise(self, image: Union[np.ndarray, Image.Image], 
                  noise_factor: Optional[float] = None) -> np.ndarray:
        """
        Add Gaussian noise to image.
        
        Args:
            image: Input image
            noise_factor: Noise standard deviation. If None, uses config value.
            
        Returns:
            Noisy image
        """
        image = self._ensure_numpy(image)
        if noise_factor is None:
            noise_factor = self.config.noise_factor
        
        noise = np.random.normal(0, noise_factor, image.shape).astype(np.float32)
        noisy = image.astype(np.float32) + noise
        noisy = np.clip(noisy, 0, 255).astype(np.uint8)
        
        return noisy
    
    def blur(self, image: Union[np.ndarray, Image.Image], 
             blur_factor: Optional[float] = None) -> np.ndarray:
        """
        Apply Gaussian blur to image.
        
        Args:
            image: Input image
            blur_factor: Blur kernel size. If None, uses config value.
            
        Returns:
            Blurred image
        """
        image = self._ensure_numpy(image)
        if blur_factor is None:
            blur_factor = self.config.blur_factor
        
        kernel_size = int(blur_factor * 10) * 2 + 1  # Ensure odd kernel size
        kernel_size = max(3, min(kernel_size, 21))  # Limit kernel size
        
        blurred = cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)
        return blurred
    
    def elastic_transform(self, image: Union[np.ndarray, Image.Image]) -> np.ndarray:
        """
        Apply elastic transformation to image.
        
        Args:
            image: Input image
            
        Returns:
            Elastic-transformed image
        """
        image = self._ensure_numpy(image)
        height, width = image.shape[:2]
        
        # Create random displacement fields
        dx = np.random.rand(height, width) * 2 - 1
        dy = np.random.rand(height, width) * 2 - 1
        
        # Smooth the displacement fields
        dx = cv2.GaussianBlur(dx, (21, 21), self.config.elastic_sigma)
        dy = cv2.GaussianBlur(dy, (21, 21), self.config.elastic_sigma)
        
        # Scale the displacement fields
        dx = dx * self.config.elastic_alpha
        dy = dy * self.config.elastic_alpha
        
        # Create coordinate grids
        x, y = np.meshgrid(np.arange(width), np.arange(height))
        
        # Apply displacement
        x_new = np.clip(x + dx, 0, width - 1).astype(np.float32)
        y_new = np.clip(y + dy, 0, height - 1).astype(np.float32)
        
        # Remap the image
        transformed = cv2.remap(image, x_new, y_new, cv2.INTER_LINEAR, 
                               borderMode=cv2.BORDER_REFLECT)
        
        return transformed
    
    def augment_image(self, image: Union[np.ndarray, Image.Image], 
                     techniques: List[str] = None) -> List[np.ndarray]:
        """
        Apply multiple augmentation techniques to an image.
        
        Args:
            image: Input image
            techniques: List of techniques to apply. If None, applies all techniques.
            
        Returns:
            List of augmented images
        """
        if techniques is None:
            techniques = [
                "rotate", "translate", "scale", "flip_horizontal", 
                "adjust_brightness", "adjust_contrast", "adjust_saturation",
                "add_noise", "blur", "elastic_transform"
            ]
        
        augmented_images = []
        
        for technique in techniques:
            if hasattr(self, technique):
                method = getattr(self, technique)
                try:
                    augmented = method(image)
                    if augmented is not None and not np.array_equal(augmented, image):
                        augmented_images.append(augmented)
                except Exception as e:
                    print(f"Error applying {technique}: {e}")
                    continue
        
        return augmented_images

class AlbumentationsAugmenter:
    """
    Image augmentation using Albumentations library for more advanced techniques.
    """
    
    def __init__(self, config: Optional[ImageAugmentationConfig] = None):
        """Initialize the Albumentations augmenter."""
        self.config = config or ImageAugmentationConfig()
        if self.config.random_state is not None:
            random.seed(self.config.random_state)
            np.random.seed(self.config.random_state)
    
    def get_transforms(self, technique: str) -> A.Compose:
        """Get Albumentations transforms for a specific technique."""
        
        transforms = {
            "basic": A.Compose([
                A.Rotate(limit=self.config.rotation_range, p=1.0),
                A.HorizontalFlip(p=0.5),
                A.VerticalFlip(p=0.5),
                A.RandomBrightnessContrast(p=1.0),
                A.HueSaturationValue(p=1.0),
            ]),
            
            "geometric": A.Compose([
                A.ShiftScaleRotate(shift_limit=0.1, scale_limit=0.2, rotate_limit=30, p=1.0),
                A.ElasticTransform(alpha=1, sigma=50, alpha_affine=50, p=1.0),
                A.GridDistortion(num_steps=5, distort_limit=0.3, p=1.0),
                A.OpticalDistortion(distort_limit=0.2, shift_limit=0.15, p=1.0),
            ]),
            
            "noise_blur": A.Compose([
                A.GaussNoise(var_limit=(10.0, 50.0), p=1.0),
                A.GaussianBlur(blur_limit=(3, 7), p=1.0),
                A.MotionBlur(blur_limit=7, p=1.0),
                A.MedianBlur(blur_limit=7, p=1.0),
            ]),
            
            "color": A.Compose([
                A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2, p=1.0),
                A.HueSaturationValue(hue_shift_limit=20, sat_shift_limit=30, val_shift_limit=20, p=1.0),
                A.RGBShift(r_shift_limit=15, g_shift_limit=15, b_shift_limit=15, p=1.0),
                A.ChannelShuffle(p=1.0),
            ]),
            
            "weather": A.Compose([
                A.RandomRain(slant_lower=-10, slant_upper=10, drop_length=20, drop_width=1, drop_color=(200, 200, 200), p=1.0),
                A.RandomFog(fog_coef_lower=0.1, fog_coef_upper=0.3, alpha_coef=0.08, p=1.0),
                A.RandomSunFlare(flare_roi=(0, 0, 1, 0.5), angle_lower=0, angle_upper=1, num_flare_circles_lower=6, num_flare_circles_upper=10, p=1.0),
            ]),
            
            "perspective": A.Compose([
                A.Perspective(scale=(0.05, 0.1), p=1.0),
                A.PiecewiseAffine(scale=(0.03, 0.05), nb_rows=4, nb_cols=4, p=1.0),
            ]),
        }
        
        return transforms.get(technique, transforms["basic"])
    
    def augment_image(self, image: np.ndarray, technique: str = "basic") -> np.ndarray:
        """
        Apply Albumentations augmentation to an image.
        
        Args:
            image: Input image as numpy array
            technique: Augmentation technique to apply
            
        Returns:
            Augmented image
        """
        transforms = self.get_transforms(technique)
        augmented = transforms(image=image)
        return augmented["image"]
    
    def augment_batch(self, images: List[np.ndarray], technique: str = "basic") -> List[np.ndarray]:
        """
        Apply Albumentations augmentation to a batch of images.
        
        Args:
            images: List of input images
            technique: Augmentation technique to apply
            
        Returns:
            List of augmented images
        """
        return [self.augment_image(img, technique) for img in images]

# --- Augmentation Functions for Registry ---

def classical_rotation_augmentation(images: List[np.ndarray], **kwargs) -> List[List[np.ndarray]]:
    """Classical rotation augmentation."""
    augmenter = ClassicalImageAugmenter()
    return [augmenter.augment_image(img, ["rotate"]) for img in images]

def classical_geometric_augmentation(images: List[np.ndarray], **kwargs) -> List[List[np.ndarray]]:
    """Classical geometric augmentation (rotation, translation, scale, flip)."""
    augmenter = ClassicalImageAugmenter()
    return [augmenter.augment_image(img, ["rotate", "translate", "scale", "flip_horizontal"]) for img in images]

def classical_color_augmentation(images: List[np.ndarray], **kwargs) -> List[List[np.ndarray]]:
    """Classical color augmentation (brightness, contrast, saturation)."""
    augmenter = ClassicalImageAugmenter()
    return [augmenter.augment_image(img, ["adjust_brightness", "adjust_contrast", "adjust_saturation"]) for img in images]

def classical_noise_blur_augmentation(images: List[np.ndarray], **kwargs) -> List[List[np.ndarray]]:
    """Classical noise and blur augmentation."""
    augmenter = ClassicalImageAugmenter()
    return [augmenter.augment_image(img, ["add_noise", "blur"]) for img in images]

def classical_elastic_augmentation(images: List[np.ndarray], **kwargs) -> List[List[np.ndarray]]:
    """Classical elastic transformation augmentation."""
    augmenter = ClassicalImageAugmenter()
    return [augmenter.augment_image(img, ["elastic_transform"]) for img in images]

def classical_mixed_augmentation(images: List[np.ndarray], **kwargs) -> List[List[np.ndarray]]:
    """Classical mixed augmentation using all techniques."""
    augmenter = ClassicalImageAugmenter()
    return [augmenter.augment_image(img) for img in images]

def albumentations_basic_augmentation(images: List[np.ndarray], **kwargs) -> List[List[np.ndarray]]:
    """Albumentations basic augmentation."""
    augmenter = AlbumentationsAugmenter()
    return [[augmenter.augment_image(img, "basic")] for img in images]

def albumentations_geometric_augmentation(images: List[np.ndarray], **kwargs) -> List[List[np.ndarray]]:
    """Albumentations geometric augmentation."""
    augmenter = AlbumentationsAugmenter()
    return [[augmenter.augment_image(img, "geometric")] for img in images]

def albumentations_noise_blur_augmentation(images: List[np.ndarray], **kwargs) -> List[List[np.ndarray]]:
    """Albumentations noise and blur augmentation."""
    augmenter = AlbumentationsAugmenter()
    return [[augmenter.augment_image(img, "noise_blur")] for img in images]

def albumentations_color_augmentation(images: List[np.ndarray], **kwargs) -> List[List[np.ndarray]]:
    """Albumentations color augmentation."""
    augmenter = AlbumentationsAugmenter()
    return [[augmenter.augment_image(img, "color")] for img in images]

def albumentations_weather_augmentation(images: List[np.ndarray], **kwargs) -> List[List[np.ndarray]]:
    """Albumentations weather effects augmentation."""
    augmenter = AlbumentationsAugmenter()
    return [[augmenter.augment_image(img, "weather")] for img in images]

def albumentations_perspective_augmentation(images: List[np.ndarray], **kwargs) -> List[List[np.ndarray]]:
    """Albumentations perspective transformation augmentation."""
    augmenter = AlbumentationsAugmenter()
    return [[augmenter.augment_image(img, "perspective")] for img in images]

# --- Registry ---
IMAGE_AUGMENTATION_REGISTRY: Dict[str, Callable] = {
    # Classical augmentations
    "classical_rotation": classical_rotation_augmentation,
    "classical_geometric": classical_geometric_augmentation,
    "classical_color": classical_color_augmentation,
    "classical_noise_blur": classical_noise_blur_augmentation,
    "classical_elastic": classical_elastic_augmentation,
    "classical_mixed": classical_mixed_augmentation,
    
    # Albumentations augmentations
    "albumentations_basic": albumentations_basic_augmentation,
    "albumentations_geometric": albumentations_geometric_augmentation,
    "albumentations_noise_blur": albumentations_noise_blur_augmentation,
    "albumentations_color": albumentations_color_augmentation,
    "albumentations_weather": albumentations_weather_augmentation,
    "albumentations_perspective": albumentations_perspective_augmentation,
}

# --- Documentation ---
"""
IMAGE_AUGMENTATION_REGISTRY keys:
- 'classical_rotation': Classical rotation augmentation
- 'classical_geometric': Classical geometric transformations (rotation, translation, scale, flip)
- 'classical_color': Classical color adjustments (brightness, contrast, saturation)
- 'classical_noise_blur': Classical noise and blur effects
- 'classical_elastic': Classical elastic transformation
- 'classical_mixed': Classical mixed techniques
- 'albumentations_basic': Albumentations basic augmentation
- 'albumentations_geometric': Albumentations geometric transformations
- 'albumentations_noise_blur': Albumentations noise and blur effects
- 'albumentations_color': Albumentations color adjustments
- 'albumentations_weather': Albumentations weather effects
- 'albumentations_perspective': Albumentations perspective transformations

Usage:
    from image_aug import IMAGE_AUGMENTATION_REGISTRY
    
    # Classical augmentation
    augmented = IMAGE_AUGMENTATION_REGISTRY["classical_geometric"](images)
    
    # Albumentations augmentation
    augmented = IMAGE_AUGMENTATION_REGISTRY["albumentations_basic"](images)
""" 