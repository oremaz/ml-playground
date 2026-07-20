import torch
import torch.nn as nn
from typing import Dict, Type, Optional, Callable

import timm

class SimpleCNN(nn.Module):
    """A very small CNN for quick experiments on vision datasets."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
            nn.Conv2d(16, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2),
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(32 * 8 * 8, 64),  # assumes input images of size 32x32
            nn.ReLU(),
            nn.Linear(64, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class AdaptiveCNN(nn.Module):
    """CNN that adapts to different input sizes using adaptive pooling with modern GELU activations."""

    def __init__(self, num_classes: int = 2, input_channels: int = 3):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(input_channels, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(32, 32, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.GELU(),
            nn.MaxPool2d(2),
            nn.Dropout(0.25),
            
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.GELU(),
            nn.AdaptiveAvgPool2d((4, 4)),  # Adaptive pooling to fixed size
            nn.Dropout(0.25),
        )
        
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(128 * 4 * 4, 256),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(256, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, channels: int):
        super().__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(channels)

    def forward(self, x):
        residual = x
        out = nn.functional.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        return nn.functional.relu(out)


class ResidualCNN(nn.Module):
    """A small residual network inspired by ResNet architecture."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.prep = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3, stride=2, padding=1),
        )
        self.layer1 = ResidualBlock(64)
        self.layer2 = ResidualBlock(64)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.prep(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.pool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x


class ResNet50(nn.Module):
    """Transfer learning using a ResNet-50 backbone pretrained on ImageNet with adaptive input size."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True, input_size: int = 224, freeze_backbone: bool = True):
        super().__init__()
        from torchvision import models

        self.input_size = input_size
        
        try:  # torchvision >= 0.13
            self.model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
        except AttributeError:  # older torchvision versions
            self.model = models.resnet50(pretrained=pretrained)
        
        # Optionally freeze backbone for faster training
        if freeze_backbone:
            for name, param in self.model.named_parameters():
                if 'fc' not in name:  # Don't freeze the final classification layer
                    param.requires_grad = False
        
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        # If input size doesn't match expected size (224), resize it
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        return self.model(x)


class BinaryClassifier(nn.Module):
    """EfficientNet-based binary classifier for AI-generated image detection (DiffusionFake architecture)."""
    
    # Encoder configurations mapping encoder names to features and initialization
    ENCODER_PARAMS = {
        "xception": {
            "features": 2048,
            "init_op": lambda **kwargs: timm.create_model('xception', pretrained=True, **kwargs)
        },
        "tf_efficientnet_b2_ns": {
            "features": 1408,
            "init_op": lambda **kwargs: timm.create_model('tf_efficientnet_b2_ns', pretrained=True, **kwargs)
        },
        "tf_efficientnet_b3_ns": {
            "features": 1536,
            "init_op": lambda **kwargs: timm.create_model('tf_efficientnet_b3_ns', pretrained=True, **kwargs)
        },
        "tf_efficientnet_b4_ns": {
            "features": 1792,
            "init_op": lambda **kwargs: timm.create_model('tf_efficientnet_b4_ns', pretrained=True, **kwargs)
        },
        "tf_efficientnet_b5_ns": {
            "features": 2048,
            "init_op": lambda **kwargs: timm.create_model('tf_efficientnet_b5_ns', pretrained=True, **kwargs)
        },
        "tf_efficientnet_b6_ns": {
            "features": 2304,
            "init_op": lambda **kwargs: timm.create_model('tf_efficientnet_b6_ns', pretrained=True, **kwargs)
        },
        "tf_efficientnet_b7_ns": {
            "features": 2560,
            "init_op": lambda **kwargs: timm.create_model('tf_efficientnet_b7_ns', pretrained=True, **kwargs)
        },
    }

    def __init__(self, encoder: str = "tf_efficientnet_b4_ns", num_classes: int = 1, drop_rate: float = 0.2, has_feature: bool = True, feature_dim: int = 128) -> None:
        super().__init__()
        if encoder not in self.ENCODER_PARAMS:
            raise ValueError(f"Unknown encoder '{encoder}'. Available: {list(self.ENCODER_PARAMS.keys())}")
        
        self.encoder = self.ENCODER_PARAMS[encoder]["init_op"]()
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(drop_rate)
        self.channel_drop = nn.Dropout2d(drop_rate)
        self.has_feature = has_feature
        self.feature_dim = self.ENCODER_PARAMS[encoder]["features"]
        self.feature = nn.Linear(self.feature_dim, feature_dim)
        self.fc = nn.Linear(self.feature_dim, num_classes)

    def forward(self, x, return_features: bool = False):
        featuremap = self.encoder.forward_features(x)
        x = self.global_pool(featuremap).flatten(1)
        output = self.fc(x)

        if self.has_feature and return_features:
            return output, featuremap

        return output


class CLIPClassifier(nn.Module):
    """Fine-tuned CLIP vision encoder with a linear classification head using Hugging Face transformers."""

    def __init__(self, num_classes: int = 2, model_name: str = "openai/clip-vit-base-patch32", input_size: int = 224):
        super().__init__()
        try:
            from transformers import CLIPVisionModel, CLIPProcessor
        except ImportError:
            raise ImportError(
                "transformers is required for CLIPClassifier. Install with: pip install transformers"
            )

        self.input_size = input_size
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.vision_model = CLIPVisionModel.from_pretrained(model_name)
        
        # Freeze all vision model parameters first
        for param in self.vision_model.parameters():
            param.requires_grad = False
        
            
        # Add classification head
        hidden_size = self.vision_model.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Convert tensor to PIL images for processor
        # Assuming x is in range [0, 1] or already normalized
        if x.shape[-1] != 224 or x.shape[-2] != 224:
            x = torch.nn.functional.interpolate(x, size=(224, 224), mode='bilinear', align_corners=False)
        
        # Process through CLIP vision model
        # Note: This expects input in the format expected by CLIP (RGB, normalized)
        vision_outputs = self.vision_model(pixel_values=x)
        pooled_output = vision_outputs.pooler_output  # [batch_size, hidden_size]
        
        # Classification
        logits = self.classifier(pooled_output)
        return logits


class DINOv3Classifier(nn.Module):
    """DINOv3 vision backbone with a linear classification head using Hugging Face transformers.

    DINOv3 models are gated on HuggingFace. Authenticate with ``huggingface-cli login``
    before using this class.
    """

    def __init__(
        self,
        num_classes: int = 2,
        model_name: str = "facebook/dinov3-vitl16-pretrain-lvd1689m",
        input_size: int = 518,
        freeze_backbone: bool = True,
    ):
        super().__init__()
        try:
            from transformers import AutoModel, AutoImageProcessor
        except ImportError:
            raise ImportError(
                "transformers is required for DINOv3Classifier. Install with: pip install transformers"
            )

        self.input_size = input_size
        self.backbone = AutoModel.from_pretrained(model_name)

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        hidden_size = self.backbone.config.hidden_size
        self.classifier = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        if x.shape[-1] != self.input_size or x.shape[-2] != self.input_size:
            x = torch.nn.functional.interpolate(
                x, size=(self.input_size, self.input_size),
                mode='bilinear', align_corners=False,
            )

        outputs = self.backbone(pixel_values=x)
        # Use CLS token output
        pooled = outputs.last_hidden_state[:, 0]
        logits = self.classifier(pooled)
        return logits


class Qwen2VLQLoRA(nn.Module):
    """QLoRA fine-tuning wrapper for the Qwen 2.5 Vision-Language model."""

    def __init__(
        self,
        model_name: str = "Qwen/Qwen2.5-VL-2B-Instruct",
        num_classes: int = 2,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
    ):
        super().__init__()
        from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
        from peft import LoraConfig, get_peft_model

        import torch as _torch

        self.processor = AutoProcessor.from_pretrained(model_name)

        quant_cfg = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=_torch.bfloat16 if _torch.cuda.is_available() else _torch.float16,
        )

        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            quantization_config=quant_cfg,
            device_map="auto",
        )

        lora_cfg = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias="none",
            target_modules="all-linear",
        )
        self.model = get_peft_model(self.model, lora_cfg)
        self.num_classes = num_classes

    def forward(self, images, prompts):
        inputs = self.processor(images=images, text=prompts, return_tensors="pt").to(self.model.device)
        output = self.model(**inputs)
        return output.logits[:, -1, : self.num_classes]


class TimmVisionModel(nn.Module):
    """Thin wrapper around timm backbones for registry-friendly initialization."""

    def __init__(
        self,
        *,
        model_name: str,
        num_classes: int = 2,
        pretrained: bool = True,
        in_chans: int = 3,
        global_pool: Optional[str] = None,
        **kwargs,
    ) -> None:
        super().__init__()

        timm_kwargs = dict(kwargs)
        timm_kwargs.setdefault("num_classes", num_classes)
        timm_kwargs.setdefault("pretrained", pretrained)
        timm_kwargs.setdefault("in_chans", in_chans)
        timm_kwargs.setdefault("img_size", 32)
        if global_pool is not None:
            timm_kwargs.setdefault("global_pool", global_pool)

        attempt_kwargs = dict(timm_kwargs)
        def _create_with_fallbacks(kws: dict):
            try:
                return timm.create_model(model_name, **kws)
            except TypeError as e:
                msg = str(e)
                # Drop unsupported kwargs one by one and retry
                retried = False
                if "unexpected keyword argument 'img_size'" in msg and 'img_size' in kws:
                    print(f"Dropping unsupported 'img_size' argument for model {model_name}")
                    kws = dict(kws)
                    kws.pop('img_size', None)
                    retried = True
                if "unexpected keyword argument 'global_pool'" in msg and 'global_pool' in kws:
                    kws = dict(kws)
                    kws.pop('global_pool', None)
                    retried = True
                if retried:
                    return timm.create_model(model_name, **kws)
                raise
            except RuntimeError as e:
                raise
        
        self.model = _create_with_fallbacks(attempt_kwargs)
        self.expected_img_size = timm_kwargs.get("img_size")

    def forward(self, x):
        size = self.expected_img_size
        if size and (x.shape[-1] != size or x.shape[-2] != size):
            x = torch.nn.functional.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
        return self.model(x)


def _register_timm_model(
    backbone: str,
    *,
    registry_name: str,
    default_kwargs: Optional[Dict[str, object]] = None,
) -> Callable[..., nn.Module]:
    """Create a callable that instantiates :class:`TimmVisionModel`."""

    defaults = dict(default_kwargs or {})

    class _TimmWrapper(TimmVisionModel):
        def __init__(self, num_classes: int = 2, **kwargs) -> None:
            params = dict(defaults)
            params.update(kwargs)
            super().__init__(model_name=backbone, num_classes=num_classes, **params)

    _TimmWrapper.__name__ = f"Timm_{registry_name.title().replace('-', '_')}"
    return _TimmWrapper


def _register_binary_classifier(
    encoder: str,
    *,
    registry_name: str,
    default_kwargs: Optional[Dict[str, object]] = None,
) -> Callable[..., nn.Module]:
    """Create a callable that instantiates :class:`BinaryClassifier` with a specific encoder."""

    defaults = dict(default_kwargs or {})

    class _BinaryClassifierWrapper(BinaryClassifier):
        def __init__(self, num_classes: int = 2, **kwargs) -> None:
            params = dict(defaults)
            params.update(kwargs)
            super().__init__(encoder=encoder, num_classes=num_classes, **params)

    _BinaryClassifierWrapper.__name__ = f"BinaryClassifier_{registry_name.title().replace('-', '_')}"
    return _BinaryClassifierWrapper


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "simple_cnn": SimpleCNN,
    "adaptive_cnn": AdaptiveCNN,
    "residual_cnn": ResidualCNN,
    "resnet50": ResNet50,
    "binary_efficientnet_b4_ns": _register_binary_classifier(
        "tf_efficientnet_b4_ns", registry_name="efficientnet_b4_ns"
    ),
    "binary_xception": _register_binary_classifier(
        "xception", registry_name="xception"
    ),
    "clip_classifier": CLIPClassifier,
    "dinov3_classifier": DINOv3Classifier,
    "qwen2_vl_qlora": Qwen2VLQLoRA,
    # Modern timm-based backbones for vision benchmarks
    "timm_convnextv2_tiny": _register_timm_model(
        "convnextv2_tiny.fcmae_ft_in1k", registry_name="convnextv2_tiny"
    ),
    "timm_efficientnetv2_s": _register_timm_model(
        "tf_efficientnetv2_s.in21k", registry_name="tf_efficientnetv2_s"
    ),
    "timm_vit_base_patch16": _register_timm_model(
        "vit_base_patch16_224.augreg2_in21k_ft_in1k",
        registry_name="vit_base_patch16",
    ),
    "timm_vit_mae_base": _register_timm_model(
        "vit_base_patch16_224.mae", registry_name="vit_mae_base"
    ),
    # DINOv2 self-supervised ViT backbones
    "timm_dinov2_vit_small": _register_timm_model(
        "vit_small_patch14_dinov2", registry_name="dinov2_vit_small"
    ),
    "timm_dinov2_vit_base": _register_timm_model(
        "vit_base_patch14_dinov2", registry_name="dinov2_vit_base"
    ),
    "timm_dinov2_vit_large": _register_timm_model(
        "vit_large_patch14_dinov2", registry_name="dinov2_vit_large"
    ),
}


def get_model(name: str, num_classes: int = 2, **kwargs) -> nn.Module:
    """Retrieve a vision model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)