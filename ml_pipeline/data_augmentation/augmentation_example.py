#!/usr/bin/env python3
"""
Example script demonstrating the usage of text_aug.py and image_aug.py modules.
This script shows how to use both LLM-based and classical augmentation techniques.
"""

import numpy as np
from PIL import Image
import os

# Import our augmentation modules
from text_aug import TEXT_AUGMENTATION_REGISTRY, LLMTextAugmenter, ClassicalTextAugmenter
from image_aug import IMAGE_AUGMENTATION_REGISTRY, ClassicalImageAugmenter, AlbumentationsAugmenter

def text_augmentation_example():
    """Demonstrate text augmentation techniques."""
    print("=== Text Augmentation Examples ===\n")
    
    # Sample texts
    sample_texts = [
        "This is a delicious recipe for chocolate cake.",
        "The weather is beautiful today.",
        "Machine learning models require good data."
    ]
    
    print("Original texts:")
    for i, text in enumerate(sample_texts, 1):
        print(f"{i}. {text}")
    print()
    
    # Classical text augmentation (no API key required)
    print("1. Classical Synonym Replacement:")
    try:
        augmented = TEXT_AUGMENTATION_REGISTRY["classical_synonym"](sample_texts)
        for i, (original, variations) in enumerate(zip(sample_texts, augmented), 1):
            print(f"Original {i}: {original}")
            for j, variation in enumerate(variations, 1):
                print(f"  Variation {j}: {variation}")
            print()
    except Exception as e:
        print(f"Error in classical synonym augmentation: {e}\n")
    
    # Classical mixed augmentation
    print("2. Classical Mixed Augmentation:")
    try:
        augmented = TEXT_AUGMENTATION_REGISTRY["classical_mixed"](sample_texts)
        for i, (original, variations) in enumerate(zip(sample_texts, augmented), 1):
            print(f"Original {i}: {original}")
            for j, variation in enumerate(variations, 1):
                print(f"  Variation {j}: {variation}")
            print()
    except Exception as e:
        print(f"Error in classical mixed augmentation: {e}\n")
    
    # LLM-based augmentation (requires API key)
    api_key = os.getenv("GOOGLE_AI_API_KEY")
    if api_key:
        print("3. LLM-based Paraphrase Augmentation:")
        try:
            augmented = TEXT_AUGMENTATION_REGISTRY["llm_paraphrase"](
                sample_texts, api_key=api_key
            )
            for i, (original, variations) in enumerate(zip(sample_texts, augmented), 1):
                print(f"Original {i}: {original}")
                for j, variation in enumerate(variations, 1):
                    print(f"  Variation {j}: {variation}")
                print()
        except Exception as e:
            print(f"Error in LLM paraphrase augmentation: {e}\n")
    else:
        print("3. LLM-based augmentation skipped (GOOGLE_AI_API_KEY not set)\n")

def image_augmentation_example():
    """Demonstrate image augmentation techniques."""
    print("=== Image Augmentation Examples ===\n")
    
    # Create a simple test image (gradient)
    height, width = 100, 100
    test_image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Create a gradient
    for i in range(height):
        for j in range(width):
            test_image[i, j] = [
                int(255 * i / height),  # Red gradient
                int(255 * j / width),   # Green gradient
                128                     # Blue constant
            ]
    
    # Convert to PIL Image for display
    pil_image = Image.fromarray(test_image)
    print(f"Created test image: {pil_image.size} pixels")
    print()
    
    # Classical geometric augmentation
    print("1. Classical Geometric Augmentation:")
    try:
        augmented = IMAGE_AUGMENTATION_REGISTRY["classical_geometric"]([test_image])
        print(f"Generated {len(augmented[0])} geometric variations")
        for i, variation in enumerate(augmented[0], 1):
            print(f"  Variation {i}: shape {variation.shape}")
        print()
    except Exception as e:
        print(f"Error in classical geometric augmentation: {e}\n")
    
    # Classical color augmentation
    print("2. Classical Color Augmentation:")
    try:
        augmented = IMAGE_AUGMENTATION_REGISTRY["classical_color"]([test_image])
        print(f"Generated {len(augmented[0])} color variations")
        for i, variation in enumerate(augmented[0], 1):
            print(f"  Variation {i}: shape {variation.shape}")
        print()
    except Exception as e:
        print(f"Error in classical color augmentation: {e}\n")
    
    # Classical mixed augmentation
    print("3. Classical Mixed Augmentation:")
    try:
        augmented = IMAGE_AUGMENTATION_REGISTRY["classical_mixed"]([test_image])
        print(f"Generated {len(augmented[0])} mixed variations")
        for i, variation in enumerate(augmented[0], 1):
            print(f"  Variation {i}: shape {variation.shape}")
        print()
    except Exception as e:
        print(f"Error in classical mixed augmentation: {e}\n")
    
    # Albumentations augmentation (if available)
    try:
        print("4. Albumentations Basic Augmentation:")
        augmented = IMAGE_AUGMENTATION_REGISTRY["albumentations_basic"]([test_image])
        print(f"Generated {len(augmented[0])} Albumentations variations")
        for i, variation in enumerate(augmented[0], 1):
            print(f"  Variation {i}: shape {variation.shape}")
        print()
    except Exception as e:
        print(f"Error in Albumentations augmentation: {e}\n")

def advanced_usage_example():
    """Demonstrate advanced usage with custom configurations."""
    print("=== Advanced Usage Examples ===\n")
    
    # Custom text augmentation configuration
    print("1. Custom Text Augmentation Configuration:")
    try:
        # Create custom text augmenter
        text_augmenter = ClassicalTextAugmenter(random_state=42)
        
        sample_text = "This is a test sentence for augmentation."
        print(f"Original: {sample_text}")
        
        # Apply specific techniques
        techniques = ["synonym_replacement", "random_insertion"]
        augmented = text_augmenter.augment_text(sample_text, techniques)
        
        for i, variation in enumerate(augmented, 1):
            print(f"Variation {i}: {variation}")
        print()
    except Exception as e:
        print(f"Error in custom text augmentation: {e}\n")
    
    # Custom image augmentation configuration
    print("2. Custom Image Augmentation Configuration:")
    try:
        from image_aug import ImageAugmentationConfig
        
        # Create custom configuration
        config = ImageAugmentationConfig(
            rotation_range=(-45, 45),
            brightness_range=(0.7, 1.3),
            noise_factor=0.1,
            random_state=42
        )
        
        # Create test image
        test_image = np.random.randint(0, 255, (50, 50, 3), dtype=np.uint8)
        
        # Create custom image augmenter
        image_augmenter = ClassicalImageAugmenter(config=config)
        
        # Apply specific techniques
        techniques = ["rotate", "adjust_brightness", "add_noise"]
        augmented = image_augmenter.augment_image(test_image, techniques)
        
        print(f"Original image shape: {test_image.shape}")
        print(f"Generated {len(augmented)} variations")
        for i, variation in enumerate(augmented, 1):
            print(f"Variation {i}: shape {variation.shape}")
        print()
    except Exception as e:
        print(f"Error in custom image augmentation: {e}\n")

def main():
    """Run all augmentation examples."""
    print("Augmentation Module Examples")
    print("=" * 50)
    print()
    
    # Run examples
    text_augmentation_example()
    image_augmentation_example()
    advanced_usage_example()
    
    print("=" * 50)
    print("Examples completed!")
    print("\nTo use LLM-based text augmentation, set the GOOGLE_AI_API_KEY environment variable:")
    print("export GOOGLE_AI_API_KEY='your_api_key_here'")
    print("\nTo install additional dependencies for image augmentation:")
    print("pip install opencv-python albumentations pillow")

if __name__ == "__main__":
    main() 