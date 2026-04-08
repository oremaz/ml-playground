import os
import random
from typing import List, Dict, Optional, Callable
import numpy as np
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, as_completed
import time

# ---------------------------------------------------------------------------
# Default models per backend (one path per provider)
# ---------------------------------------------------------------------------

DEFAULT_GEMINI_MODEL     = "gemini-3.1-flash-lite-preview"
DEFAULT_OPENROUTER_MODEL = "qwen/qwen3.6-plus:free"
DEFAULT_LOCAL_MODEL      = "Qwen/Qwen3.5-9B"

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class TextAugmentationConfig:
    """Configuration for LLM text augmentation.

    backend
    -------
    "gemini"
        Google Gemini 3.1 via google-genai.
        Requires: GOOGLE_API_KEY env var (or api_key argument).
        Default model: gemini-3.1-flash-lite-preview

    "openrouter"
        Any model served by OpenRouter (OpenAI-compatible API).
        Requires: OPENROUTER_API_KEY env var (or api_key argument).
        Default model: qwen/qwen3.6-plus:free

    "local"
        HuggingFace model loaded locally in INT4 via bitsandbytes.
        No API key required. Requires: torch, transformers, bitsandbytes, accelerate.
        Default model: Qwen/Qwen3.5-9B
    """
    backend: str = "gemini"
    model_name: Optional[str] = None  # None → resolved to backend default
    temperature: float = 0.7
    max_tokens: int = 1000
    top_p: float = 0.9
    top_k: int = 40                   # Gemini only; ignored by other backends
    max_retries: int = 3
    retry_delay: float = 1.0
    batch_size: int = 10
    max_workers: int = 4              # API backends only; local is always sequential

# ---------------------------------------------------------------------------
# LLM Augmenter
# ---------------------------------------------------------------------------

class LLMTextAugmenter:
    """Text augmentation backed by Gemini, OpenRouter, or a local INT4 model.

    Examples
    --------
    # Gemini 3.1 Flash Lite (default)
    augmenter = LLMTextAugmenter()

    # OpenRouter – Qwen 3.6 Plus free tier
    cfg = TextAugmentationConfig(backend="openrouter")
    augmenter = LLMTextAugmenter(config=cfg)

    # Local INT4 Qwen 3.5-9B (no API key needed)
    cfg = TextAugmentationConfig(backend="local")
    augmenter = LLMTextAugmenter(config=cfg)
    """

    _OPENROUTER_BASE_URL = "https://openrouter.ai/api/v1"

    def __init__(
        self,
        api_key: Optional[str] = None,
        config: Optional[TextAugmentationConfig] = None,
    ):
        self.config = config or TextAugmentationConfig()

        # Resolve model name defaults
        _defaults = {
            "gemini":     DEFAULT_GEMINI_MODEL,
            "openrouter": DEFAULT_OPENROUTER_MODEL,
            "local":      DEFAULT_LOCAL_MODEL,
        }
        if self.config.backend not in _defaults:
            raise ValueError(
                f"Unknown backend '{self.config.backend}'. "
                "Choose 'gemini', 'openrouter', or 'local'."
            )
        self.model_name = self.config.model_name or _defaults[self.config.backend]

        # ── Gemini ────────────────────────────────────────────────────────
        if self.config.backend == "gemini":
            from google import genai
            from google.genai import types as genai_types

            if api_key is None:
                api_key = os.getenv("GOOGLE_API_KEY")
            if api_key is None:
                raise ValueError(
                    "Google API key required. "
                    "Set GOOGLE_API_KEY or pass api_key."
                )
            self._client = genai.Client(api_key=api_key)
            self._gemini_gen_config = genai_types.GenerateContentConfig(
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                top_k=self.config.top_k,
                max_output_tokens=self.config.max_tokens,
            )

        # ── OpenRouter ────────────────────────────────────────────────────
        elif self.config.backend == "openrouter":
            from openai import OpenAI

            if api_key is None:
                api_key = os.getenv("OPENROUTER_API_KEY")
            if api_key is None:
                raise ValueError(
                    "OpenRouter API key required. "
                    "Set OPENROUTER_API_KEY or pass api_key."
                )
            self._client = OpenAI(
                api_key=api_key,
                base_url=self._OPENROUTER_BASE_URL,
            )

        # ── Local (bitsandbytes INT4) ──────────────────────────────────────
        elif self.config.backend == "local":
            import torch
            from transformers import (
                AutoTokenizer,
                AutoModelForCausalLM,
                BitsAndBytesConfig,
            )

            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_use_double_quant=True,
            )
            print(f"Loading {self.model_name} in INT4 (bitsandbytes NF4) ...")
            self._tokenizer = AutoTokenizer.from_pretrained(
                self.model_name, trust_remote_code=True
            )
            self._local_model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                device_map="auto",
                quantization_config=bnb_config,
                trust_remote_code=True,
            )
            self._local_model.eval()
            self._torch = torch
            print("Local model loaded.")

    # ------------------------------------------------------------------
    # Prompt building
    # ------------------------------------------------------------------

    def _create_prompt(
        self,
        text: str,
        augmentation_type: str,
        custom_instruction: Optional[str] = None,
    ) -> str:
        if custom_instruction is not None:
            return f"{custom_instruction}\n\nOriginal:\n{text}\n\nRewritten:"

        prompts = {
            "paraphrase": f"Paraphrase the following text, returning only the rewritten version:\n\n{text}",
            "synonym":    f"Rewrite the following text replacing key words with synonyms, returning only the rewritten version:\n\n{text}",
            "style":      f"Rewrite the following text in a different style, returning only the rewritten version:\n\n{text}",
            "expand":     f"Expand the following text with more context, returning only the expanded version:\n\n{text}",
            "simplify":   f"Simplify the following text, returning only the simplified version:\n\n{text}",
        }
        return prompts.get(augmentation_type, prompts["paraphrase"])

    # ------------------------------------------------------------------
    # Backend-specific calls
    # ------------------------------------------------------------------

    def _call_gemini(self, prompt: str) -> Optional[str]:
        response = self._client.models.generate_content(
            model=self.model_name,
            contents=prompt,
            config=self._gemini_gen_config,
        )
        if hasattr(response, "candidates") and response.candidates:
            return response.candidates[0].content.parts[0].text.strip()
        if hasattr(response, "text"):
            return response.text.strip()
        return None

    def _call_openrouter(self, prompt: str) -> Optional[str]:
        response = self._client.chat.completions.create(
            model=self.model_name,
            messages=[{"role": "user", "content": prompt}],
            temperature=self.config.temperature,
            max_tokens=self.config.max_tokens,
            top_p=self.config.top_p,
        )
        return response.choices[0].message.content.strip()

    def _call_local(self, prompt: str) -> Optional[str]:
        torch = self._torch
        inputs = self._tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512,
        ).to(self._local_model.device)
        with torch.no_grad():
            out_ids = self._local_model.generate(
                **inputs,
                max_new_tokens=self.config.max_tokens,
                do_sample=True,
                temperature=self.config.temperature,
                top_p=self.config.top_p,
                pad_token_id=self._tokenizer.eos_token_id,
            )
        generated = self._tokenizer.decode(
            out_ids[0][inputs["input_ids"].shape[1]:],
            skip_special_tokens=True,
        )
        return generated.strip() or None

    # ------------------------------------------------------------------
    # Core augmentation
    # ------------------------------------------------------------------

    def _augment_single_text(
        self,
        text: str,
        augmentation_type: str,
        custom_instruction: Optional[str] = None,
    ) -> Optional[str]:
        prompt = self._create_prompt(text, augmentation_type, custom_instruction)

        _dispatch = {
            "gemini":     self._call_gemini,
            "openrouter": self._call_openrouter,
            "local":      self._call_local,
        }
        call = _dispatch[self.config.backend]

        for attempt in range(self.config.max_retries):
            try:
                result = call(prompt)
                if result:
                    if result.startswith('"') and result.endswith('"'):
                        result = result[1:-1]
                    return result
            except Exception as e:
                if attempt < self.config.max_retries - 1:
                    time.sleep(self.config.retry_delay)
                else:
                    print(f"Failed after {self.config.max_retries} attempts: {e}")
                    return None

        return None

    def augment_text(
        self,
        text: str,
        augmentation_type: str = "paraphrase",
        custom_instruction: Optional[str] = None,
    ) -> Optional[str]:
        """Augment a single text.

        Args:
            text: Input text.
            augmentation_type: 'paraphrase' | 'synonym' | 'style' | 'expand' | 'simplify'
            custom_instruction: Overrides the built-in prompt template when provided.

        Returns:
            Augmented text, or None if all retries failed.
        """
        return self._augment_single_text(text, augmentation_type, custom_instruction)

    def augment_batch(
        self,
        texts: List[str],
        augmentation_type: str = "paraphrase",
        custom_instruction: Optional[str] = None,
    ) -> List[List[str]]:
        """Augment a list of texts.

        API backends (gemini, openrouter) process batches in parallel via
        ThreadPoolExecutor. The local backend runs sequentially (GPU is shared).

        Returns:
            List of lists — each inner list contains one augmented variant,
            or is empty if augmentation failed.
        """
        results = []
        for i in range(0, len(texts), self.config.batch_size):
            batch = texts[i : i + self.config.batch_size]

            if self.config.backend == "local":
                # Sequential: no benefit from threading on a single GPU
                batch_results = []
                for text in batch:
                    result = self.augment_text(text, augmentation_type, custom_instruction)
                    batch_results.append([result] if result else [])
            else:
                batch_results = []
                with ThreadPoolExecutor(max_workers=self.config.max_workers) as executor:
                    futures = [
                        executor.submit(
                            self.augment_text, text, augmentation_type, custom_instruction
                        )
                        for text in batch
                    ]
                    for future in as_completed(futures):
                        try:
                            result = future.result()
                            batch_results.append([result] if result else [])
                        except Exception as e:
                            print(f"Error augmenting text: {e}")
                            batch_results.append([])

            results.extend(batch_results)
        return results

# ---------------------------------------------------------------------------
# Classical Text Augmentation (unchanged)
# ---------------------------------------------------------------------------

class ClassicalTextAugmenter:
    """Classical text augmentation techniques that don't require an LLM API."""

    def __init__(self, random_state: Optional[int] = None):
        if random_state is not None:
            random.seed(random_state)
            np.random.seed(random_state)

    def synonym_replacement(self, text: str, ratio: float = 0.1) -> str:
        synonyms = {
            "good":  ["great", "excellent", "fine"],
            "bad":   ["terrible", "awful", "horrible"],
            "big":   ["large", "huge", "enormous"],
            "small": ["tiny", "little", "miniature"],
        }
        words = text.split()
        n = max(1, int(len(words) * ratio))
        for _ in range(n):
            available = [w for w in words if w.lower() in synonyms]
            if not available:
                break
            word = random.choice(available)
            replacement = random.choice(synonyms[word.lower()])
            for i, w in enumerate(words):
                if w.lower() == word.lower():
                    words[i] = replacement
                    break
        return " ".join(words)

    def random_insertion(self, text: str, ratio: float = 0.1) -> str:
        words = text.split()
        n = max(1, int(len(words) * ratio))
        pool = ["very", "really", "quite", "extremely"]
        for _ in range(n):
            if words:
                words.insert(random.randint(0, len(words)), random.choice(pool))
        return " ".join(words)

    def random_deletion(self, text: str, ratio: float = 0.1) -> str:
        words = text.split()
        n = max(1, int(len(words) * ratio))
        for _ in range(n):
            if len(words) > 1:
                words.pop(random.randint(0, len(words) - 1))
        return " ".join(words)

    def random_swap(self, text: str, swap_ratio: float = 0.1) -> str:
        words = text.split()
        n = max(1, int(len(words) * swap_ratio))
        for _ in range(n):
            if len(words) > 1:
                pos = random.randint(0, len(words) - 2)
                words[pos], words[pos + 1] = words[pos + 1], words[pos]
        return " ".join(words)

    def augment_text(
        self, text: str, techniques: Optional[List[str]] = None
    ) -> List[str]:
        if techniques is None:
            techniques = ["synonym_replacement", "random_insertion", "random_deletion"]
        augmented = []
        for technique in techniques:
            if hasattr(self, technique):
                result = getattr(self, technique)(text)
                if result != text:
                    augmented.append(result)
        return augmented

# ---------------------------------------------------------------------------
# Registry helper functions
# ---------------------------------------------------------------------------

def _make_llm_augmenter(backend: str, api_key: Optional[str], **kwargs) -> LLMTextAugmenter:
    cfg = TextAugmentationConfig(backend=backend)
    return LLMTextAugmenter(api_key=api_key, config=cfg)

def llm_paraphrase_augmentation(
    texts: List[str], api_key: Optional[str] = None, backend: str = "gemini", **kwargs
) -> List[List[str]]:
    """LLM-based paraphrase augmentation."""
    return _make_llm_augmenter(backend, api_key).augment_batch(texts, "paraphrase")

def llm_synonym_augmentation(
    texts: List[str], api_key: Optional[str] = None, backend: str = "gemini", **kwargs
) -> List[List[str]]:
    """LLM-based synonym substitution augmentation."""
    return _make_llm_augmenter(backend, api_key).augment_batch(texts, "synonym")

def llm_style_augmentation(
    texts: List[str], api_key: Optional[str] = None, backend: str = "gemini", **kwargs
) -> List[List[str]]:
    """LLM-based style variation augmentation."""
    return _make_llm_augmenter(backend, api_key).augment_batch(texts, "style")

def classical_synonym_augmentation(texts: List[str], **kwargs) -> List[List[str]]:
    a = ClassicalTextAugmenter()
    return [a.augment_text(t, ["synonym_replacement"]) for t in texts]

def classical_insertion_augmentation(texts: List[str], **kwargs) -> List[List[str]]:
    a = ClassicalTextAugmenter()
    return [a.augment_text(t, ["random_insertion"]) for t in texts]

def classical_deletion_augmentation(texts: List[str], **kwargs) -> List[List[str]]:
    a = ClassicalTextAugmenter()
    return [a.augment_text(t, ["random_deletion"]) for t in texts]

def classical_swap_augmentation(texts: List[str], **kwargs) -> List[List[str]]:
    a = ClassicalTextAugmenter()
    return [a.augment_text(t, ["random_swap"]) for t in texts]

def classical_mixed_augmentation(texts: List[str], **kwargs) -> List[List[str]]:
    a = ClassicalTextAugmenter()
    return [a.augment_text(t) for t in texts]

# ---------------------------------------------------------------------------
# Registry
# ---------------------------------------------------------------------------

TEXT_AUGMENTATION_REGISTRY: Dict[str, Callable] = {
    # LLM-based (pass backend="openrouter" or backend="local" to switch)
    "llm_paraphrase": llm_paraphrase_augmentation,
    "llm_synonym":    llm_synonym_augmentation,
    "llm_style":      llm_style_augmentation,
    # Classical
    "classical_synonym":   classical_synonym_augmentation,
    "classical_insertion": classical_insertion_augmentation,
    "classical_deletion":  classical_deletion_augmentation,
    "classical_swap":      classical_swap_augmentation,
    "classical_mixed":     classical_mixed_augmentation,
}

"""
TEXT_AUGMENTATION_REGISTRY usage
---------------------------------
from text_aug import TEXT_AUGMENTATION_REGISTRY

# Gemini 3.1 Flash Lite (default)
result = TEXT_AUGMENTATION_REGISTRY["llm_paraphrase"](texts=["Hello world"])

# OpenRouter – Qwen 3.6 Plus free tier
result = TEXT_AUGMENTATION_REGISTRY["llm_paraphrase"](
    texts=["Hello world"], backend="openrouter"
)

# Local INT4 (no API key, GPU required)
result = TEXT_AUGMENTATION_REGISTRY["llm_paraphrase"](
    texts=["Hello world"], backend="local"
)

# Classical (no API, no GPU)
result = TEXT_AUGMENTATION_REGISTRY["classical_mixed"](texts=["Hello world"])
"""
