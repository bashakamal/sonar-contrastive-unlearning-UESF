from typing import Tuple

import torch
import torch.nn as nn
import torchvision.models as models

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def create_model(model_name: str, num_classes: int, pretrained: bool = True) -> nn.Module:
    """
    Wrapper to create different backbones.
    Paper uses EfficientNet-B0 as baseline & unlearned backbone.
    """
    if model_name == "resnet18":
        model = models.resnet18(pretrained=pretrained)
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif model_name == "efficientnet_b0":
        model = models.efficientnet_b0(pretrained=pretrained)
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)

    elif model_name == "vit_b_16":
        model = models.vit_b_16(pretrained=pretrained)
        in_features = model.heads.head.in_features
        model.heads.head = nn.Linear(in_features, num_classes)

    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    return model.to(DEVICE)


def extract_features(model: nn.Module, x: torch.Tensor) -> torch.Tensor:
    """
    Extract intermediate features for triplet loss (TCU).
    EfficientNet/Vision-Transformer use forward_features;
    ResNet path uses conv + layers.
    """
    model.eval()
    with torch.no_grad():
        # EfficientNet / ViT
        if hasattr(model, "forward_features"):
            feats = model.forward_features(x)

        # MobileNet-like
        elif hasattr(model, "features"):
            feats = model.features(x)

        # ResNet-style
        else:
            x = model.conv1(x)
            x = model.bn1(x)
            x = model.relu(x)
            x = model.layer1(x)
            x = model.layer2(x)
            x = model.layer3(x)
            x = model.layer4(x)
            x = model.avgpool(x)
            feats = x

        return feats.view(feats.size(0), -1)
