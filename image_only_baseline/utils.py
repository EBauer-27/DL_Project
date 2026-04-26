import os
import json
import random
from typing import Dict, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from torchvision import transforms


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)

    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.deterministic = False
    torch.backends.cudnn.benchmark = True


# ------------------------------------------------------------
# File helpers
# ------------------------------------------------------------
def ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def save_json(obj: Dict, path: str) -> None:
    with open(path, "w") as f:
        json.dump(obj, f, indent=4)


# ------------------------------------------------------------
# Image transforms
# ------------------------------------------------------------
def get_image_transforms(
    image_size: int = 224,
    augment: bool = True,
) -> Tuple[transforms.Compose, transforms.Compose]:
    """
    Returns train and validation transforms.

    ImageNet normalization is used because all models are ImageNet-pretrained.
    """

    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if augment:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        train_transform = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

    val_transform = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_transform, val_transform


# ------------------------------------------------------------
# Class imbalance helper
# ------------------------------------------------------------
def compute_pos_weight_from_dataset(dataset) -> Optional[torch.Tensor]:
    """
    Computes pos_weight for BCEWithLogitsLoss.

    pos_weight = num_negative / num_positive

    This is useful when malignant/benign classes are imbalanced.
    """
    labels = dataset.df["label"].values.astype(np.float32)

    num_pos = labels.sum()
    num_neg = len(labels) - num_pos

    if num_pos == 0:
        return None

    pos_weight = num_neg / num_pos
    return torch.tensor([pos_weight], dtype=torch.float32)


# ------------------------------------------------------------
# Metrics
# ------------------------------------------------------------
def compute_binary_metrics(
    y_true: np.ndarray,
    y_prob: np.ndarray,
    threshold: float = 0.5,
) -> Dict[str, float]:
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
    }

    try:
        metrics["auc"] = roc_auc_score(y_true, y_prob)
    except ValueError:
        metrics["auc"] = float("nan")

    return metrics


# ------------------------------------------------------------
# Train / validation loops
# ------------------------------------------------------------
def train_one_epoch(
    model: nn.Module,
    dataloader,
    criterion,
    optimizer,
    device: torch.device,
) -> float:
    model.train()

    total_loss = 0.0
    total_samples = 0

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].float().to(device)

        optimizer.zero_grad()

        logits = model(images).view(-1)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

    return total_loss / max(total_samples, 1)


@torch.no_grad()
def evaluate(
    model: nn.Module,
    dataloader,
    criterion,
    device: torch.device,
    threshold: float = 0.5,
) -> Dict[str, float]:
    model.eval()

    total_loss = 0.0
    total_samples = 0

    all_labels = []
    all_probs = []

    for batch in dataloader:
        images = batch["image"].to(device)
        labels = batch["label"].float().to(device)

        logits = model(images).view(-1)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)

        batch_size = images.size(0)
        total_loss += loss.item() * batch_size
        total_samples += batch_size

        all_labels.extend(labels.detach().cpu().numpy())
        all_probs.extend(probs.detach().cpu().numpy())

    y_true = np.array(all_labels).astype(int)
    y_prob = np.array(all_probs).astype(float)

    metrics = compute_binary_metrics(
        y_true=y_true,
        y_prob=y_prob,
        threshold=threshold,
    )

    metrics["loss"] = total_loss / max(total_samples, 1)

    return metrics


# ------------------------------------------------------------
# Checkpointing
# ------------------------------------------------------------
def save_checkpoint(
    model: nn.Module,
    config: Dict,
    metrics: Dict,
    path: str,
) -> None:
    checkpoint = {
        "model_state_dict": model.state_dict(),
        "config": config,
        "metrics": metrics,
    }

    torch.save(checkpoint, path)