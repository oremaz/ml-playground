# Model and provider audit (2026-10-27)

This audit preserves provider and capacity class. It distinguishes current
checkpoint families, useful controls and manual adapters. "SOTA" is specific to
a dataset, task and compute budget; upgrading a package or checkpoint does not
establish that every registered model is best in its class. Recorded results
remain attached to the original source, weights and environment.

## Tabular prediction

| Slot | Decision | Evidence and scope |
|---|---|---|
| NVIDIA Kumo Tabular | Add classification and regression; choose `size="small"`, `"medium"` or `"large"` | [Official checkpoint](https://huggingface.co/nvidia/Kumo-Tabular), [NVIDIA reference implementation](https://github.com/NVIDIA/structured-data-models): approximately 28M, 62M and 215M parameters |
| TabICLv2 | Update package pin from 2.0.3 to 2.2.0; explicitly retain the existing v2 checkpoint filenames | [Official release](https://github.com/soda-inria/tabicl/releases/tag/v2.2.0) and [checkpoint list](https://github.com/soda-inria/tabicl#available-models); a software refresh, not a new accuracy claim |
| Google TabFM | Keep the pinned official implementation and existing model family | No verified replacement within this slot's provider/capacity contract was selected |
| TabPFN dependency | Pin the previously unbounded optional dependency to 9.2.0 | This is dependency reproducibility; TabPFN is not currently a model-registry entry |
| Constant, linear, trees, MLP and stacking | Retain as controls and trainable model families | Their performance depends on fitting and tuning; they are not universally SOTA pretrained checkpoints |

## Assistant and augmentation models

| Slot | Decision | Evidence and scope |
|---|---|---|
| Groq Qwen 27B assistant | Retain `qwen/qwen3.8-27b` | [Groq model card](https://console.groq.com/docs/model/qwen/qwen3.8-27b) and [deprecation history](https://console.groq.com/docs/deprecations) identify it as the direct 27B successor |
| Local Qwen 27B assistant | Retain `Qwen/Qwen3.8-27B` at `1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0` | [Pinned model](https://huggingface.co/Qwen/Qwen3.8-27B/tree/1d4bf0f2ff6012fd82039f2fa52739d0dd7c60c0); current public revision matches the existing pin |
| OpenRouter free assistant | Retain `openrouter/free` | A free routing service, not a fixed model size or a single reproducible pretrained checkpoint; [public catalog](https://openrouter.ai/api/v1/models) |
| Gemini augmentation | Replace retired 3.1 Flash-Lite preview with 3.5 Flash-Lite | Same published Lite product tier; parameter counts are not disclosed. [Google deprecations](https://ai.google.dev/gemini-api/docs/deprecations) and [release notes](https://ai.google.dev/gemini-api/docs/changelog) |
| OpenRouter augmentation | Use `openrouter/free` in place of the unavailable `qwen/qwen3.6-plus:free` | User-authorized exception to the size constraint: keep free routing while allowing different model sizes. Registry helpers also forward an explicit `model_name` override |
| Local 9B augmentation | Retain its explicit Qwen3.5-9B slot | No verified same-provider 9B successor was selected; changing it to 27B would violate the size constraint |

Gemini 3.5+ requests omit deprecated temperature/top-p/top-k parameters. Explicit
older model overrides retain their previous sampling configuration. Hosted API
availability and alias routing are provider-controlled; stored conversations do
not guarantee that retired endpoints can be called again.

## Vision and forecasting

| Slot | Decision | Evidence and scope |
|---|---|---|
| Qwen 2B vision-language adapter | Use `Qwen/Qwen3.5-2B`, pinned at `15852e8c16360a2fea060d615a32b45270f8a8fc`, through the multimodal auto loader | [Official 2B card](https://huggingface.co/Qwen/Qwen3.5-2B); the old default `Qwen2.5-VL-2B-Instruct` could not be verified. Historical class/registry names and explicit model overrides remain available; the new default uses image-aware chat formatting, while older explicit model IDs keep the prior processor path |
| Meta ViT-S and ViT-B image backbones | Add DINOv3 small/base choices, alongside the existing large DINOv3 | [Meta small](https://huggingface.co/facebook/dinov3-vits16-pretrain-lvd1689m), [base](https://huggingface.co/facebook/dinov3-vitb16-pretrain-lvd1689m), [large](https://huggingface.co/facebook/dinov3-vitl16-pretrain-lvd1689m); approximately 21.6M, 85.7M and 303M. Each has an explicit revision pin |
| Older DINOv2/timm models and scratch CNNs | Retain their named historical models as baselines | A DINOv2 key is not silently redirected to DINOv3; classification heads still need training |
| OpenAI CLIP ViT-B/32 | Retain its provider and size | No newer OpenAI checkpoint in the same slot was selected; a different provider's encoder would change the comparison |
| Forecasting adapters | Retain Timestral-209M, Toto 2.0 sizes, TimesFM 3.0 and Chronos-2 | Their [Mistral](https://huggingface.co/mistralai/Timestral-209M), [Datadog](https://huggingface.co/Datadog/Toto-2.0-313m), [Google](https://huggingface.co/google/timesfm-3.0-pytorch) and [Amazon](https://huggingface.co/amazon/chronos-2) cards remain the relevant releases; no verified same-size successor was substituted |
| User-supplied Hugging Face and GGUF models | Keep explicit model/file selection | There is no single baked-in size or provider to upgrade automatically |

## Kumo interface and environment

The registry exposes `kumo_classifier` and `kumo_regressor`. Both default to the
small model; choose the size explicitly in a recorded comparison. The pinned
`structured-data-models==0.2.1` uses checkpoint release `v1.0.1`. Its default
recipe uses only training context to fit transformations and can select up to
500 feature columns per ensemble member. Classification probabilities follow
the fitted label order. Regression returns the median (`q500`), not a mean.

`fit` retains labeled training context rather than optimizing model weights.
By default all training rows are used. `max_context_rows` enables an explicit,
seeded context subset; its selected indices are stored. Query batching and
estimator batching bound working memory. Sample weights are unsupported and
are rejected. Checkpoint serialization retains context and parameters, rather
than copying frozen GPU model weights; reload requires the recorded SDM version
and its cached/downloadable checkpoint.

Install `uv pip install -e ".[kumo]"` in a dedicated environment: SDM needs Torch
2.7+, while the existing research stack is Torch 2.6. PyArrow is pinned to 21.0.0
in that extra because newer wheels require NumPy 2, whereas this project retains
NumPy 1.26.4. Do not synchronize a running experiment's environment to new extras.

For a small classification smoke run in that environment:

```python
from sklearn.datasets import make_classification
from sklearn.model_selection import train_test_split
from ml_pipeline.pipelines_torch.models import MODEL_REGISTRY

X, y = make_classification(n_samples=32, n_features=4, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.25, stratify=y, random_state=42,
)
model = MODEL_REGISTRY["kumo_classifier"](
    size="small", device="cpu", num_estimators=1, random_state=42,
)
model.fit(X_train, y_train)
probabilities = model.predict_proba(X_test)
assert probabilities.shape == (len(X_test), len(model.classes_))
```

The first prediction loads the public weights. This tiny example is a smoke
check; use fixed development folds and a reserved test set for model comparisons.

The pretrained **small** classifier and regressor were exercised on tiny CPU
examples, including serialization/reload and batching. This validates adapter
behavior, not comparative accuracy. Medium/large inference, gated DINOv3 weights,
real Qwen vision inference, and hosted APIs were not run in this cleanup.
