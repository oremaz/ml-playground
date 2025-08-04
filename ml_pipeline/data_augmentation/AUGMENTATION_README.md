# Data Augmentation Modules

This repository contains comprehensive augmentation modules for various data types, organized in the `data_augmentation/` folder:

- **`text_aug.py`**: Text augmentation using LLM prompting and classical techniques
- **`image_aug.py`**: Image augmentation using classical computer vision and Albumentations
- **`augmentations.py`**: Traditional data augmentation techniques for numerical/feature data (SMOTE, Mixup, etc.)

## Features

### Text Augmentation (`text_aug.py`)

#### LLM-based Augmentation
- **Paraphrasing**: Generate semantic variations using Google's Gemini 
- **Synonym Substitution**: Replace words with contextually appropriate synonyms
- **Style Variation**: Rewrite text in different writing styles
- **Context Expansion**: Add relevant context and details
- **Simplification**: Simplify complex text while preserving meaning

#### Classical Augmentation
- **Synonym Replacement**: Replace words with predefined synonyms
- **Random Insertion**: Insert random words at random positions
- **Random Deletion**: Remove random words from text
- **Random Swap**: Swap adjacent words randomly

### Image Augmentation (`image_aug.py`)

#### Classical Techniques (OpenCV + PIL)
- **Geometric Transformations**: Rotation, translation, scaling, flipping
- **Color Adjustments**: Brightness, contrast, saturation modifications
- **Noise and Blur**: Gaussian noise, various blur effects
- **Elastic Transformations**: Non-linear deformations

#### Albumentations Library
- **Advanced Geometric**: Shift-scale-rotate, elastic transform, grid distortion
- **Noise and Blur**: Multiple blur types, noise variations
- **Color Manipulation**: RGB shifts, channel shuffling, hue/saturation
- **Weather Effects**: Rain, fog, sun flare effects
- **Perspective Transformations**: Perspective and piecewise affine transforms

### Traditional Data Augmentation (`augmentations.py`)

#### SMOTE Variants
- **SMOTE**: Synthetic Minority Over-sampling Technique
- **BorderlineSMOTE**: SMOTE with borderline focus
- **SVMSMOTE**: SMOTE using SVM support vectors
- **KMeansSMOTE**: SMOTE with K-means clustering
- **ADASYN**: Adaptive Synthetic Sampling

#### Custom Techniques
- **Mixup**: Linear interpolation between samples
- **MixupSMOTE**: Custom combination of Mixup and SMOTE
- **SMOTE with Gaussian Noise**: SMOTE with added noise

#### Hybrid Samplers
- **SMOTEENN**: SMOTE + Edited Nearest Neighbors
- **SMOTETomek**: SMOTE + Tomek Links

#### Data Cleaners
- **Tomek Links**: Remove borderline samples
- **Edited Nearest Neighbors**: Remove noisy samples
- **Repeated ENN**: Iterative ENN cleaning
- **AllKNN**: All K-Nearest Neighbors cleaning

## Installation

### Dependencies

The modules require the following packages (already included in `requirements (1).txt`):

```bash
# Core dependencies
numpy>=1.21.0
pandas>=1.3.0

# Text augmentation
google-generativeai>=0.3.0  # For LLM-based augmentation

# Image augmentation
opencv-python>=4.5.0
pillow>=8.0.0
albumentations>=1.0.0

# Traditional data augmentation
imbalanced-learn>=0.8.0

# Optional: for advanced image effects
scipy>=1.7.0
```

### Setup

1. Install dependencies:
```bash
pip install -r "requirements (1).txt"
```

2. For LLM-based text augmentation, set your Google AI API key:
```bash
export GOOGLE_AI_API_KEY='your_api_key_here'
```

## Usage

### Quick Start

```python
from text_aug import TEXT_AUGMENTATION_REGISTRY
from image_aug import IMAGE_AUGMENTATION_REGISTRY
from augmentations import AUGMENTATION_REGISTRY

# Text augmentation
texts = ["Hello world", "This is a test"]
augmented_texts = TEXT_AUGMENTATION_REGISTRY["classical_synonym"](texts)

# Image augmentation
images = [your_image_array]  # numpy array
augmented_images = IMAGE_AUGMENTATION_REGISTRY["classical_geometric"](images)

# Traditional data augmentation
X, y = your_features, your_labels  # numpy arrays
X_aug, y_aug = AUGMENTATION_REGISTRY["smote"](X, y)
```

### Text Augmentation Examples

#### LLM-based Augmentation

```python
from text_aug import LLMTextAugmenter

# Initialize with API key
augmenter = LLMTextAugmenter(api_key="your_api_key")

# Single text augmentation
text = "This is a delicious recipe for chocolate cake."
augmented = augmenter.augment_text(text, "paraphrase")
print(augmented)  # "Here's a tasty chocolate cake recipe."

# Batch augmentation
texts = ["Hello world", "Good morning"]
augmented_batch = augmenter.augment_batch(texts, "synonym")
```

#### Classical Augmentation

```python
from text_aug import ClassicalTextAugmenter

augmenter = ClassicalTextAugmenter(random_state=42)

# Apply specific techniques
text = "This is a test sentence."
augmented = augmenter.augment_text(text, ["synonym_replacement", "random_insertion"])
```

### Image Augmentation Examples

#### Classical Techniques

```python
from image_aug import ClassicalImageAugmenter, ImageAugmentationConfig

# Custom configuration
config = ImageAugmentationConfig(
    rotation_range=(-30, 30),
    brightness_range=(0.8, 1.2),
    noise_factor=0.05
)

augmenter = ClassicalImageAugmenter(config=config)

# Apply specific techniques
image = your_image  # numpy array or PIL Image
augmented = augmenter.augment_image(image, ["rotate", "adjust_brightness", "add_noise"])
```

#### Albumentations

```python
from image_aug import AlbumentationsAugmenter

augmenter = AlbumentationsAugmenter()

# Apply predefined transforms
image = your_image  # numpy array
augmented = augmenter.augment_image(image, "geometric")
```

### Traditional Data Augmentation Examples

#### SMOTE and Variants

```python
from augmentations import AUGMENTATION_REGISTRY

# Basic SMOTE
X_aug, y_aug = AUGMENTATION_REGISTRY["smote"](X, y)

# Borderline SMOTE
X_aug, y_aug = AUGMENTATION_REGISTRY["borderline_smote"](X, y)

# SMOTE with Gaussian noise
X_aug, y_aug = AUGMENTATION_REGISTRY["smote_gaussian"](X, y, noise_std=0.05)
```

#### Mixup Techniques

```python
# Basic Mixup
X_aug, y_aug = AUGMENTATION_REGISTRY["mixup"](X, y, alpha=0.2)

# Custom MixupSMOTE
X_aug, y_aug = AUGMENTATION_REGISTRY["mixup_smote"](X, y, n_samples=100, alpha=0.2)
```

#### Data Cleaning

```python
# Tomek Links cleaning
X_clean, y_clean = AUGMENTATION_REGISTRY["tomeklinks"](X, y)

# Compose SMOTE with cleaning
from augmentations import compose_augmentation
smote_with_cleaning = compose_augmentation(
    AUGMENTATION_REGISTRY["smote"], 
    AUGMENTATION_REGISTRY["tomeklinks"]
)
X_aug, y_aug = smote_with_cleaning(X, y)
```

## Registry System

Both modules use a registry system for easy access to augmentation techniques:

### Text Augmentation Registry

```python
TEXT_AUGMENTATION_REGISTRY = {
    # LLM-based
    "llm_paraphrase": llm_paraphrase_augmentation,
    "llm_synonym": llm_synonym_augmentation,
    "llm_style": llm_style_augmentation,
    
    # Classical
    "classical_synonym": classical_synonym_augmentation,
    "classical_insertion": classical_insertion_augmentation,
    "classical_deletion": classical_deletion_augmentation,
    "classical_swap": classical_swap_augmentation,
    "classical_mixed": classical_mixed_augmentation,
}
```

### Image Augmentation Registry

```python
IMAGE_AUGMENTATION_REGISTRY = {
    # Classical
    "classical_rotation": classical_rotation_augmentation,
    "classical_geometric": classical_geometric_augmentation,
    "classical_color": classical_color_augmentation,
    "classical_noise_blur": classical_noise_blur_augmentation,
    "classical_elastic": classical_elastic_augmentation,
    "classical_mixed": classical_mixed_augmentation,
    
    # Albumentations
    "albumentations_basic": albumentations_basic_augmentation,
    "albumentations_geometric": albumentations_geometric_augmentation,
    "albumentations_noise_blur": albumentations_noise_blur_augmentation,
    "albumentations_color": albumentations_color_augmentation,
    "albumentations_weather": albumentations_weather_augmentation,
    "albumentations_perspective": albumentations_perspective_augmentation,
}
```

### Traditional Data Augmentation Registry

```python
AUGMENTATION_REGISTRY = {
    # No augmentation
    "none": none_augmentation,
    
    # SMOTE variants
    "smote": smote_augmentation,
    "borderline_smote": borderline_smote_augmentation,
    "svm_smote": svm_smote_augmentation,
    "kmeans_smote": kmeans_smote_augmentation,
    "adasyn": adasyn_augmentation,
    "random_oversampler": random_oversampler_augmentation,
    
    # Custom techniques
    "mixup": mixup_augmentation,
    "mixup_smote": mixup_smote_augmentation,
    "smote_gaussian": smote_gaussian_augmentation,
    
    # Hybrid samplers
    "smoteenn": smoteenn_augmentation,
    "smotetomek": smotetomek_augmentation,
    
    # Data cleaners
    "tomeklinks": tomeklinks_cleaner,
    "enn": enn_cleaner,
    "repeated_enn": repeated_enn_cleaner,
    "allknn": allknn_cleaner,
}
```

## Configuration

### Text Augmentation Configuration

```python
from text_aug import TextAugmentationConfig

config = TextAugmentationConfig(
    model_name="gemini-pro",
    temperature=0.7,
    max_tokens=1000,
    max_retries=3,
    batch_size=10,
    max_workers=4
)
```

### Image Augmentation Configuration

```python
from image_aug import ImageAugmentationConfig

config = ImageAugmentationConfig(
    rotation_range=(-30, 30),
    width_shift_range=0.1,
    height_shift_range=0.1,
    zoom_range=(0.8, 1.2),
    brightness_range=(0.8, 1.2),
    contrast_range=(0.8, 1.2),
    saturation_range=(0.8, 1.2),
    noise_factor=0.05,
    blur_factor=0.5,
    elastic_alpha=1.0,
    elastic_sigma=50.0,
    random_state=42
)
```

## Integration with Existing Pipeline

The augmentation modules can be easily integrated with the existing PyTorch pipeline:

```python
from pipelines_torch.base import GeneralPipeline
from pre_aug.text_aug import TEXT_AUGMENTATION_REGISTRY
from pre_aug.image_aug import IMAGE_AUGMENTATION_REGISTRY

# For text data
text_augmentation = TEXT_AUGMENTATION_REGISTRY["classical_mixed"]

# For image data
image_augmentation = IMAGE_AUGMENTATION_REGISTRY["classical_geometric"]

# For traditional numerical data
data_augmentation = AUGMENTATION_REGISTRY["smote"]

# Create pipeline with augmentation
pipeline = GeneralPipeline(
    model=your_model,
    loss_fn=your_loss_fn,
    optimizer_cls=your_optimizer,
    optimizer_params=your_params,
    augmentations=data_augmentation,  # or text_augmentation, image_augmentation
    task_type="classification"
)
```

## Running Examples

Run the example script to see all techniques in action:

```bash
python augmentation_example.py
```

This will demonstrate:
- Text augmentation with classical and LLM techniques
- Image augmentation with classical and Albumentations techniques
- Traditional data augmentation with SMOTE and Mixup techniques
- Custom configuration examples
- Error handling and API key management

## Best Practices

### Text Augmentation
1. **LLM-based**: Use for high-quality semantic variations, but be mindful of API costs
2. **Classical**: Use for quick, rule-based augmentations
3. **Combination**: Mix both approaches for comprehensive augmentation
4. **Validation**: Always validate augmented text maintains original meaning

### Image Augmentation
1. **Classical**: Good for basic transformations and when you need fine control
2. **Albumentations**: Better for advanced effects and production pipelines
3. **Domain-specific**: Choose augmentations appropriate for your image domain
4. **Parameter tuning**: Adjust augmentation strength based on your dataset

### Traditional Data Augmentation
1. **SMOTE variants**: Use for imbalanced classification datasets
2. **Mixup**: Effective for regularization and improving generalization
3. **Data cleaning**: Combine with oversampling for better quality synthetic samples
4. **Composition**: Use `compose_augmentation()` to combine multiple techniques

### General Tips
1. **Reproducibility**: Set random seeds for consistent results
2. **Batch processing**: Use batch augmentation for efficiency
3. **Error handling**: Implement proper error handling for API calls
4. **Memory management**: Be mindful of memory usage with large image datasets

## Troubleshooting

### Common Issues

1. **Google AI API Key Error**:
   ```
   ValueError: Google AI API key is required
   ```
   Solution: Set the `GOOGLE_AI_API_KEY` environment variable

2. **Import Errors**:
   ```
   ModuleNotFoundError: No module named 'albumentations'
   ```
   Solution: Install missing dependencies with `pip install albumentations opencv-python pillow imbalanced-learn`

3. **Memory Issues with Large Images**:
   Solution: Process images in smaller batches or reduce image resolution

4. **API Rate Limiting**:
   Solution: Implement retry logic and reduce batch sizes for LLM augmentation


## Module Structure

```
data_augmentation/
├── text_aug.py              # Text augmentation (LLM + classical)
├── image_aug.py             # Image augmentation (OpenCV + Albumentations)
├── augmentations.py         # Traditional data augmentation (SMOTE, Mixup, etc.)
├── augmentation_example.py  # Example usage script
└── AUGMENTATION_README.md   # This documentation
```

## Contributing

To add new augmentation techniques:

1. Implement the technique in the appropriate class/module
2. Add a registry function
3. Update the registry dictionary
4. Add documentation and examples
5. Test with the example script

## License

This code is part of the food project and follows the same licensing terms. 