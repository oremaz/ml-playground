import os
import random
import json
from typing import List, Dict, Any, Optional, Callable, Union
import numpy as np
from google import genai
from google.genai import types
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

@dataclass
class TextAugmentationConfig:
    """Configuration for text augmentation."""
    model_name: str = "gemini-2.5-flash-lite"
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    top_k: int = 40
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 10
    max_workers: int = 4

class LLMTextAugmenter:
    """
    Text augmentation using LLM prompting for generating variations of input text.
    """
    
    def __init__(self, api_key: Optional[str] = None, config: Optional[TextAugmentationConfig] = None):
        self.config = config or TextAugmentationConfig()
        # Set up Google AI
        if api_key is None:
            api_key = os.getenv("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError("Google API key is required. Set GOOGLE_API_KEY environment variable or pass api_key parameter.")
        self.client = genai.Client(api_key=api_key)
        self.model_name = self.config.model_name
        self.generation_config = types.GenerateContentConfig(
            temperature=self.config.temperature,
            top_p=self.config.top_p,
            top_k=self.config.top_k,
            max_output_tokens=self.config.max_tokens,
        )

    def _create_prompt(self, text: str, augmentation_type: str) -> str:
        """Create a prompt for the specified augmentation type."""
        
        prompts = {
            "paraphrase": f"Paraphrase: {text}",
            "synonym": f"Replace words with synonyms: {text}",
            "style": f"Rewrite in different style: {text}",
            "expand": f"Expand with context: {text}",
            "simplify": f"Simplify: {text}"
        }
        
        return prompts.get(augmentation_type, prompts["paraphrase"])
    
    def _augment_single_text(self, text: str, augmentation_type: str) -> Optional[str]:
        """Augment a single text using the specified augmentation type."""
        
        prompt = self._create_prompt(text, augmentation_type)
        
        for attempt in range(self.config.max_retries):
            try:
                response = self.client.models.generate_content(
                    model=self.model_name,
                    contents=prompt,
                    config=self.generation_config,
                )
                if hasattr(response, 'candidates') and response.candidates:
                    augmented_text = response.candidates[0].content.parts[0].text.strip()
                elif hasattr(response, 'text'):
                    augmented_text = response.text.strip()
                else:
                    augmented_text = None
                if augmented_text:
                    if augmented_text.startswith('"') and augmented_text.endswith('"'):
                        augmented_text = augmented_text[1:-1]
                    return augmented_text
                    
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                    continue
                else:
                    print(f"Failed to augment text after {self.config.max_retries} attempts: {e}")
                    return None
        
        return None
    
    def augment_text(self, text: str, augmentation_type: str = "paraphrase") -> Optional[str]:
        """
        Augment a single text with a single variation.
        
        Args:
            text: Input text to augment
            augmentation_type: Type of augmentation to apply
            
        Returns:
            Augmented text or None if augmentation failed
        """
        return self._augment_single_text(text, augmentation_type)
    
    def augment_batch(self, texts: List[str], augmentation_type: str = "paraphrase") -> List[List[str]]:
        """
        Augment a batch of texts with a single variation.
        
        Args:
            texts: List of input texts to augment
            augmentation_type: Type of augmentation to apply
            
        Returns:
            List of lists, where each inner list contains augmented variations for the corresponding input text
        """
        results = []
        
        # Process in batches to avoid overwhelming the API
        for i in range(0, len(texts), self.config.batch_size):
            batch_texts = texts[i:i + self.config.batch_size]
            batch_results = []
            
            with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                futures = [executor.submit(self.augment_text, text, augmentation_type) for text in batch_texts]
                
                for future in as_completed(futures):
                    try:
                        result = future.result()
                        batch_results.append([result] if result else [])
                    except Exception as e:
                        print(f"Error augmenting text: {e}")
                        batch_results.append([])
            
            results.extend(batch_results)
        
        return results

# --- Classical Text Augmentation Techniques ---

class ClassicalTextAugmenter:
    """
    Classical text augmentation techniques that don't require LLM API calls.
    """
    
    def __init__(self, random_state: Optional[int] = None):
        """Initialize the classical text augmenter."""
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)
    
    def synonym_replacement(self, text: str, ratio: float = 0.1) -> str:
        """
        Replace words with synonyms using a simple dictionary approach.
        
        Args:
            text: Input text
            ratio: Fraction of words to replace
            
        Returns:
            Text with some words replaced by synonyms
        """
        # Simple synonym dictionary (in practice, you'd use a more comprehensive one)
        synonyms = {
            "good": ["great", "excellent", "fine"],
            "bad": ["terrible", "awful", "horrible"],
            "big": ["large", "huge", "enormous"],
            "small": ["tiny", "little", "miniature"]
        }
        
        words = text.split()
        n_replacements = max(1, int(len(words) * ratio))
        
        for _ in range(n_replacements):
            # Find a word that has synonyms
            available_words = [word for word in words if word.lower() in synonyms]
            if not available_words:
                break
            
            word_to_replace = random.choice(available_words)
            replacement = random.choice(synonyms[word_to_replace.lower()])
            
            # Replace the word (case-sensitive)
            for i, word in enumerate(words):
                if word.lower() == word_to_replace.lower():
                    words[i] = replacement
                    break
        
        return " ".join(words)
    
    def random_insertion(self, text: str, ratio: float = 0.1) -> str:
        """
        Randomly insert words into the text.
        
        Args:
            text: Input text
            ratio: Fraction of words to insert
            
        Returns:
            Text with random word insertions
        """
        words = text.split()
        n_insertions = max(1, int(len(words) * ratio))
        
        # Simple word bank for insertions
        insertion_words = ["very", "really", "quite", "extremely"]
        
        for _ in range(n_insertions):
            if len(words) > 0:
                insert_pos = random.randint(0, len(words))
                insert_word = random.choice(insertion_words)
                words.insert(insert_pos, insert_word)
        
        return " ".join(words)
    
    def random_deletion(self, text: str, ratio: float = 0.1) -> str:
        """
        Randomly delete words from the text.
        
        Args:
            text: Input text
            ratio: Fraction of words to delete
            
        Returns:
            Text with random word deletions
        """
        words = text.split()
        n_deletions = max(1, int(len(words) * ratio))
        
        for _ in range(n_deletions):
            if len(words) > 1:  # Keep at least one word
                delete_pos = random.randint(0, len(words) - 1)
                words.pop(delete_pos)
        
        return " ".join(words)
    
    def random_swap(self, text: str, swap_ratio: float = 0.1) -> str:
        """
        Randomly swap adjacent words in the text.
        
        Args:
            text: Input text
            swap_ratio: Fraction of words to swap
            
        Returns:
            Text with random word swaps
        """
        words = text.split()
        n_swaps = max(1, int(len(words) * swap_ratio))
        
        for _ in range(n_swaps):
            if len(words) > 1:
                pos = random.randint(0, len(words) - 2)
                words[pos], words[pos + 1] = words[pos + 1], words[pos]
        
        return " ".join(words)
    
    def augment_text(self, text: str, techniques: List[str] = None) -> List[str]:
        """
        Apply multiple classical augmentation techniques to a text.
        
        Args:
            text: Input text
            techniques: List of techniques to apply
            
        Returns:
            List of augmented text variations
        """
        if techniques is None:
            techniques = ["synonym_replacement", "random_insertion", "random_deletion"]
        
        augmented_texts = []
        
        for technique in techniques:
            if hasattr(self, technique):
                method = getattr(self, technique)
                augmented = method(text)
                if augmented != text:
                    augmented_texts.append(augmented)
        
        return augmented_texts

# --- Augmentation Functions for Registry ---

def llm_paraphrase_augmentation(texts: List[str], api_key: Optional[str] = None, **kwargs) -> List[List[str]]:
    """LLM-based paraphrase augmentation."""
    augmenter = LLMTextAugmenter(api_key=api_key)
    return augmenter.augment_batch(texts, "paraphrase")

def llm_synonym_augmentation(texts: List[str], api_key: Optional[str] = None, **kwargs) -> List[List[str]]:
    """LLM-based synonym substitution augmentation."""
    augmenter = LLMTextAugmenter(api_key=api_key)
    return augmenter.augment_batch(texts, "synonym")

def llm_style_augmentation(texts: List[str], api_key: Optional[str] = None, **kwargs) -> List[List[str]]:
    """LLM-based style variation augmentation."""
    augmenter = LLMTextAugmenter(api_key=api_key)
    return augmenter.augment_batch(texts, "style")

def classical_synonym_augmentation(texts: List[str], **kwargs) -> List[List[str]]:
    """Classical synonym replacement augmentation."""
    augmenter = ClassicalTextAugmenter()
    return [augmenter.augment_text(text, ["synonym_replacement"]) for text in texts]

def classical_insertion_augmentation(texts: List[str], **kwargs) -> List[List[str]]:
    """Classical random insertion augmentation."""
    augmenter = ClassicalTextAugmenter()
    return [augmenter.augment_text(text, ["random_insertion"]) for text in texts]

def classical_deletion_augmentation(texts: List[str], **kwargs) -> List[List[str]]:
    """Classical random deletion augmentation."""
    augmenter = ClassicalTextAugmenter()
    return [augmenter.augment_text(text, ["random_deletion"]) for text in texts]

def classical_swap_augmentation(texts: List[str], **kwargs) -> List[List[str]]:
    """Classical random swap augmentation."""
    augmenter = ClassicalTextAugmenter()
    return [augmenter.augment_text(text, ["random_swap"]) for text in texts]

def classical_mixed_augmentation(texts: List[str], **kwargs) -> List[List[str]]:
    """Classical mixed augmentation using all techniques."""
    augmenter = ClassicalTextAugmenter()
    return [augmenter.augment_text(text) for text in texts]

# --- Registry ---
TEXT_AUGMENTATION_REGISTRY: Dict[str, Callable] = {
    # LLM-based augmentations
    "llm_paraphrase": llm_paraphrase_augmentation,
    "llm_synonym": llm_synonym_augmentation,
    "llm_style": llm_style_augmentation,
    
    # Classical augmentations
    "classical_synonym": classical_synonym_augmentation,
    "classical_insertion": classical_insertion_augmentation,
    "classical_deletion": classical_deletion_augmentation,
    "classical_swap": classical_swap_augmentation,
    "classical_mixed": classical_mixed_augmentation,
}

# --- Documentation ---
"""
TEXT_AUGMENTATION_REGISTRY keys:
- 'llm_paraphrase': LLM-based paraphrase generation
- 'llm_synonym': LLM-based synonym substitution
- 'llm_style': LLM-based style variation
- 'classical_synonym': Classical synonym replacement
- 'classical_insertion': Classical random word insertion
- 'classical_deletion': Classical random word deletion
- 'classical_swap': Classical random word swapping
- 'classical_mixed': Classical mixed techniques

Usage:
    from text_aug import TEXT_AUGMENTATION_REGISTRY
    
    # LLM augmentation (requires API key)
    augmented = TEXT_AUGMENTATION_REGISTRY["llm_paraphrase"](
        texts=["Hello world"], 
        api_key="your_api_key"
    )
    
    # Classical augmentation
    augmented = TEXT_AUGMENTATION_REGISTRY["classical_synonym"](
        texts=["Hello world"]
    )
"""