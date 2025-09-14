import torch
import torch.nn as nn
from typing import Dict, Type


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


class DropoutCNN(nn.Module):
    """A deeper CNN with GELU activations and dropout for regularisation."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 32, kernel_size=3, padding=1),
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
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(64 * 8 * 8, 256),
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
    """Transfer learning using a ResNet-50 backbone pretrained on ImageNet."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        from torchvision import models

        try:  # torchvision >= 0.13
            self.model = models.resnet50(
                weights=models.ResNet50_Weights.DEFAULT if pretrained else None
            )
        except AttributeError:  # older torchvision versions
            self.model = models.resnet50(pretrained=pretrained)
        in_features = self.model.fc.in_features
        self.model.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)


class VisionTransformer(nn.Module):
    """Vision Transformer (ViT-B/16) with optional ImageNet pretraining."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        from torchvision import models

        try:
            self.model = models.vit_b_16(
                weights=models.ViT_B_16_Weights.DEFAULT if pretrained else None
            )
        except AttributeError:
            self.model = models.vit_b_16(pretrained=pretrained)
        in_features = self.model.heads.head.in_features
        self.model.heads.head = nn.Linear(in_features, num_classes)

    def forward(self, x):
        return self.model(x)

class CLIPClassifier(nn.Module):
    """Fine-tuned CLIP vision encoder with a linear classification head."""

    def __init__(self, num_classes: int = 2, model_name: str = "ViT-B/32"):
        super().__init__()
        import clip  # type: ignore

        clip_model, _ = clip.load(model_name, device="cpu", jit=False)
        self.visual = clip_model.visual
        for param in self.visual.parameters():
            param.requires_grad = False
        self.classifier = nn.Linear(self.visual.output_dim, num_classes)

    def forward(self, x):
        x = self.visual(x)
        x = self.classifier(x)
        return x


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


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "simple_cnn": SimpleCNN,
    "dropout_cnn": DropoutCNN,
    "residual_cnn": ResidualCNN,
    "resnet50": ResNet50,
    "vision_transformer": VisionTransformer,
    "clip_classifier": CLIPClassifier,
    "qwen2_vl_qlora": Qwen2VLQLoRA,
}


def get_model(name: str, num_classes: int = 2, **kwargs) -> nn.Module:
    """Retrieve a vision model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)
