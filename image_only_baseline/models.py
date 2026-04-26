import torch.nn as nn
from torchvision import models


class BinaryClassifierHead(nn.Module):
    def __init__(self, in_features: int, dropout: float = 0.3):
        super().__init__()

        self.head = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.head(x)


# ------------------------------------------------------------
# ResNet18
# ------------------------------------------------------------
def create_resnet18(
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    weights = models.ResNet18_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.resnet18(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = BinaryClassifierHead(
        in_features=in_features,
        dropout=dropout,
    )

    return model


# ------------------------------------------------------------
# ResNet50
# ------------------------------------------------------------
def create_resnet50(
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    weights = models.ResNet50_Weights.IMAGENET1K_V2 if pretrained else None
    model = models.resnet50(weights=weights)

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.fc = BinaryClassifierHead(
        in_features=in_features,
        dropout=dropout,
    )

    return model


# ------------------------------------------------------------
# VGG16
# ------------------------------------------------------------
def create_vgg16(
    pretrained: bool = True,
    dropout: float = 0.5,
    freeze_backbone: bool = True,
) -> nn.Module:
    weights = models.VGG16_Weights.IMAGENET1K_V1 if pretrained else None
    model = models.vgg16(weights=weights)

    if freeze_backbone:
        for param in model.features.parameters():
            param.requires_grad = False

    # VGG classifier:
    # classifier[0] = Linear
    # classifier[1] = ReLU
    # classifier[2] = Dropout
    # classifier[3] = Linear
    # classifier[4] = ReLU
    # classifier[5] = Dropout
    # classifier[6] = Linear
    in_features = model.classifier[-1].in_features

    model.classifier[2] = nn.Dropout(p=dropout)
    model.classifier[5] = nn.Dropout(p=dropout)
    model.classifier[-1] = nn.Linear(in_features, 1)

    return model


# ------------------------------------------------------------
# GoogLeNet
# ------------------------------------------------------------
def create_googlenet(
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    weights = models.GoogLeNet_Weights.IMAGENET1K_V1 if pretrained else None

    model = models.googlenet(
        weights=weights,
        aux_logits=True,
    )

    model.aux_logits = False
    model.aux1 = None
    model.aux2 = None

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    in_features = model.fc.in_features
    model.dropout = nn.Dropout(p=dropout)
    model.fc = nn.Linear(in_features, 1)

    return model


# ------------------------------------------------------------
# Model factory
# ------------------------------------------------------------
def create_model(
    model_name: str,
    pretrained: bool = True,
    dropout: float = 0.3,
    freeze_backbone: bool = False,
) -> nn.Module:
    model_name = model_name.lower()

    if model_name == "resnet18":
        return create_resnet18(
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

    elif model_name == "resnet50":
        return create_resnet50(
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

    elif model_name == "vgg16":
        return create_vgg16(
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

    elif model_name == "googlenet":
        return create_googlenet(
            pretrained=pretrained,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

    else:
        raise ValueError(
            f"Unknown model_name: {model_name}. "
            f"Choose from: resnet18, resnet50, vgg16, googlenet."
        )