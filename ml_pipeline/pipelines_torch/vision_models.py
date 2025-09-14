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


class YOLOv5(nn.Module):
    """YOLOv5 model adapted for image classification tasks."""

    def __init__(self, num_classes: int = 2, pretrained: bool = True):
        super().__init__()
        import torch as _torch

        # Use the classification variant provided by Ultralytics
        self.model = _torch.hub.load(
            "ultralytics/yolov5", "yolov5s-cls", pretrained=pretrained
        )
        if self.model.model[-1].out_features != num_classes:
            self.model.model[-1] = nn.Linear(
                self.model.model[-1].in_features, num_classes
            )

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


MODEL_REGISTRY: Dict[str, Type[nn.Module]] = {
    "simple_cnn": SimpleCNN,
    "dropout_cnn": DropoutCNN,
    "residual_cnn": ResidualCNN,
    "resnet50": ResNet50,
    "vision_transformer": VisionTransformer,
    "yolov5": YOLOv5,
    "clip_classifier": CLIPClassifier,
}


def get_model(name: str, num_classes: int = 2, **kwargs) -> nn.Module:
    """Retrieve a vision model by name."""
    if name not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model '{name}'. Available: {list(MODEL_REGISTRY)}")
    return MODEL_REGISTRY[name](num_classes=num_classes, **kwargs)
