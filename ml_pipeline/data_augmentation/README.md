# Data augmentation

Training-data transformations for tabular arrays, text and images. This module
provides callables; the training pipeline decides which rows they can see.
Apply augmentation only after splitting, to the training portion of each fold.
Validation and final evaluation inputs retain their original distribution.

Install the relevant optional dependencies from the repository root: `.[tabular]`
for the tabular example, `.[vision]` for images and `.[text]` for text backends.
Hosted OpenRouter augmentation also needs the OpenAI-compatible client from
`.[research]`; local quantized inference needs the GPU dependencies.

## Choose an interface

| Input | Entry point | Return value | Dependencies |
|---|---|---|---|
| Numeric rows | [AUGMENTATION_REGISTRY](augmentations.py) | `(X_aug, y_aug)` | NumPy and imbalanced-learn |
| Texts | [TEXT_AUGMENTATION_REGISTRY](text_aug.py) | One list of variants per input text | Classical methods need no provider; LLM methods need their backend |
| Images | [IMAGE_AUGMENTATION_REGISTRY](image_aug.py) | One list of image arrays per input image | OpenCV/Pillow; Albumentations for its registry entries |

The text and image return values are nested lists. A text with no applicable
classical replacement can have an empty variants list. These APIs are not
interchangeable with the tabular `(X, y)` interface.

## Tabular sampling

The registered names are `none`, `smote`, `borderline_smote`, `svm_smote`,
`kmeans_smote`, `adasyn`, `mixup`, `mixup_smote`, `smoteenn` and `smotetomek`.
The registry in [augmentations.py](augmentations.py) is the authoritative list.
MGS-GRF and TabEBM are not implemented in this module.

For the oversamplers, `max_factor` means majority count divided by the target
minority count **before cleaning**. With 100 majority and 20 minority examples,
`max_factor=2` requests 50 minority examples; it does not reduce an existing
class count. The SMOTE/ADASYN variants and `mixup_smote` then apply TomekLinks.
`smoteenn` uses its own ENN cleaning; `smotetomek` uses its own Tomek cleaning.
Cleaning can change the final counts, so the requested ratio is not a guarantee.

`mixup_smote` currently assumes binary labels 0 and 1, with 1 as the minority.
Plain `mixup` interpolates targets as well as features. For integer class labels,
its fractional targets need a compatible training objective; do not silently
treat them as hard class IDs. `none` leaves inputs unchanged.

This complete example needs no data files, provider or model weights:

```python
import numpy as np
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ml_pipeline.data_augmentation.augmentations import AUGMENTATION_REGISTRY

X, y = make_classification(
    n_samples=200, n_features=8, weights=[0.8, 0.2], random_state=42,
)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, stratify=y, random_state=42,
)
X_aug, y_aug = AUGMENTATION_REGISTRY["smote"](
    X_train, y_train, max_factor=2.0, random_state=42,
)
assert len(X_aug) == len(y_aug)
print(np.bincount(y_train), np.bincount(y_aug))
```

## Text and image variants

Text augmentation creates rewritten training examples, such as paraphrases or
style variants. It is an optional library utility: no experiment or application
in these three project checkouts currently calls it. It is separate from the CSV
assistant and from the detection/evasion experiment implementations.

Classical text keys are `classical_synonym`, `classical_insertion`,
`classical_deletion`, `classical_swap` and `classical_mixed`. Synonym replacement
uses a small built-in word list, rather than a language-wide lexical resource.
For example:

```python
from ml_pipeline.data_augmentation.text_aug import TEXT_AUGMENTATION_REGISTRY

variants = TEXT_AUGMENTATION_REGISTRY["classical_synonym"](["A good small meal"])
assert len(variants) == 1
print(variants[0])
```

`llm_paraphrase`, `llm_synonym` and `llm_style` accept `backend="gemini"`,
`"openrouter"` or `"local"`. Hosted backends use `GOOGLE_API_KEY` or
`OPENROUTER_API_KEY`; local inference needs its model weights and GPU packages.
Use `LLMTextAugmenter` with `TextAugmentationConfig` for explicit model and
sampling settings. Gemini remains the default backend; selecting OpenRouter uses
its free router. Registry convenience functions accept `model_name`; OpenRouter augmentation
defaults to `openrouter/free` because the previous free Plus default could not be
verified. The router can select different model sizes, as authorized for this
free tier. The CSV assistant's separate `openrouter/free` setting is unchanged.

The image registry includes classical geometry, color, noise/blur, elastic and
mixed transformations, plus Albumentations basic, geometric, noise/blur, color,
weather and perspective transforms. Inputs are image arrays; inspect
[image_aug.py](image_aug.py) for expected shapes and transformation settings.
A transformation is valid only if it preserves the task's label.

## Reproducibility and integration

Tabular samplers accept `random_state`; some reset NumPy's global random stream.
`ClassicalTextAugmenter(random_state=...)` seeds Python and NumPy. Registry text
wrappers use the existing random streams rather than exposing all class options.
Preserve the original seed, call order and augmentation configuration when
replaying an experiment.

See [the pipeline guide](../pipelines_torch/README.md) for fitting and checkpoint
behavior, and [the experiment index](../../experiments/README.md) for recorded
results. Extra synthetic rows alone do not establish improved generalization.
