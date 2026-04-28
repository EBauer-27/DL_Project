import os
import json
import argparse
import random
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)


# ------------------------------------------------------------
# Reproducibility
# ------------------------------------------------------------
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ------------------------------------------------------------
# Dataset: image only
# ------------------------------------------------------------
class MIDASImageOnlyDataset(Dataset):
    def __init__(self, csv_path, image_root, transform=None):
        self.df = pd.read_csv(csv_path)
        self.image_root = image_root
        self.transform = transform

        if "label" not in self.df.columns:
            self.df = self._create_label_column(self.df)

    def _create_label_column(self, df):
        df = df.copy()
        danger_words = ["melanoma", "bcc", "scc", "carcinoma", "malignant", "mct", "ak"]

        if "midas_path" not in df.columns:
            raise ValueError("No label column and no midas_path column found.")

        path_text = df["midas_path"].fillna("").astype(str).str.lower()
        df["label"] = path_text.apply(
            lambda x: 1 if any(word in x for word in danger_words) else 0
        )
        return df

    def _image_path(self, row):
        fname = row["midas_file_name"]
        return os.path.join(self.image_root, os.path.basename(str(fname)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        img_path = self._image_path(row)

        if os.path.exists(img_path):
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.float32)

        return image, label


# ------------------------------------------------------------
# Transforms
# ------------------------------------------------------------
def get_transforms(augment=True, image_size=224):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.02,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((image_size, image_size)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

    val_tf = transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_tf, val_tf


# ------------------------------------------------------------
# Image-only pretrained ResNet classifier
# ------------------------------------------------------------
class ResNetImageClassifier(nn.Module):
    def __init__(
        self,
        model_name="resnet50",
        pretrained=True,
        dropout=0.3,
        freeze_backbone=False,
    ):
        super().__init__()

        if model_name == "resnet18":
            weights = models.ResNet18_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet18(weights=weights)
        elif model_name == "resnet50":
            weights = models.ResNet50_Weights.DEFAULT if pretrained else None
            self.backbone = models.resnet50(weights=weights)
        else:
            raise ValueError(f"Unsupported model_name: {model_name}")

        in_features = self.backbone.fc.in_features

        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False

        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.backbone(x).squeeze(1)


# ------------------------------------------------------------
# Train/evaluate helpers
# ------------------------------------------------------------
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    n = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()
        logits = model(images)
        loss = criterion(logits, labels)

        loss.backward()
        optimizer.step()

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n += batch_size

    return running_loss / max(n, 1)


@torch.no_grad()
def evaluate(model, loader, criterion, device, threshold=0.5):
    model.eval()

    all_labels = []
    all_probs = []
    running_loss = 0.0
    n = 0

    for images, labels in loader:
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)
        loss = criterion(logits, labels)

        probs = torch.sigmoid(logits)

        batch_size = images.size(0)
        running_loss += loss.item() * batch_size
        n += batch_size

        all_labels.extend(labels.detach().cpu().numpy().tolist())
        all_probs.extend(probs.detach().cpu().numpy().tolist())

    y_true = np.array(all_labels).astype(int)
    y_prob = np.array(all_probs)
    y_pred = (y_prob >= threshold).astype(int)

    metrics = {
        "loss": running_loss / max(n, 1),
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
    }

    return metrics


# ------------------------------------------------------------
# Main training
# ------------------------------------------------------------
def main(args):
    set_seed(args.seed)

    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")
    print("Using device:", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    train_tf, val_tf = get_transforms(
        augment=args.augment,
        image_size=args.image_size,
    )

    train_ds = MIDASImageOnlyDataset(
        csv_path=args.train_csv,
        image_root=args.image_root,
        transform=train_tf,
    )

    val_ds = MIDASImageOnlyDataset(
        csv_path=args.val_csv,
        image_root=args.image_root,
        transform=val_tf,
    )

    print(f"Train samples: {len(train_ds)}")
    print(f"Val samples:   {len(val_ds)}")

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = ResNetImageClassifier(
        model_name=args.model_name,
        pretrained=True,
        dropout=args.dropout,
        freeze_backbone=args.freeze_backbone,
    ).to(device)

    criterion = nn.BCEWithLogitsLoss()

    optimizer = torch.optim.AdamW(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=args.lr,
        weight_decay=args.weight_decay,
    )

    best_auc = -1.0
    best_epoch = -1
    best_metrics = None
    patience_counter = 0

    config = vars(args)

    print("\nConfig:")
    print(json.dumps(config, indent=4))
    print()

    for epoch in range(1, args.epochs + 1):
        train_loss = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            loader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
        )

        print(
            f"Epoch {epoch:03d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val acc: {val_metrics['accuracy']:.4f} | "
            f"Val precision: {val_metrics['precision']:.4f} | "
            f"Val recall: {val_metrics['recall']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f}",
            flush=True,
        )

        current_auc = val_metrics["auc"]

        if current_auc > best_auc:
            best_auc = current_auc
            best_epoch = epoch
            best_metrics = val_metrics
            patience_counter = 0

            torch.save(
                {
                    "model_state_dict": model.state_dict(),
                    "config": config,
                    "best_epoch": best_epoch,
                    "best_metrics": best_metrics,
                },
                output_dir / "best_resnet_image_only.pth",
            )

            with open(output_dir / "best_config_and_metrics.json", "w") as f:
                json.dump(
                    {
                        "best_epoch": best_epoch,
                        "config": config,
                        "metrics": best_metrics,
                        "checkpoint_path": str(output_dir / "best_resnet_image_only.pth"),
                    },
                    f,
                    indent=4,
                )
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping at epoch {epoch}")
            break

    print("\nTraining finished.")
    print(f"Best epoch: {best_epoch}")
    print(f"Best validation AUC: {best_auc:.4f}")
    print(f"Best model saved to: {output_dir / 'best_resnet_image_only.pth'}")
    print(f"Metrics saved to: {output_dir / 'best_config_and_metrics.json'}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--train_csv", type=str, required=True)
    parser.add_argument("--val_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--output_dir", type=str, default="image_only_baseline/results/resnet_only")

    parser.add_argument("--model_name", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--image_size", type=int, default=224)

    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--epochs", type=int, default=35)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--num_workers", type=int, default=2)

    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--threshold", type=float, default=0.5)

    parser.add_argument("--freeze_backbone", action="store_true")
    parser.add_argument("--no_augment", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()
    args.augment = not args.no_augment

    main(args)