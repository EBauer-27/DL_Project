import os
import json
import argparse
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
    confusion_matrix,
    classification_report,
)


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

        danger_words = [
            "melanoma",
            "bcc",
            "scc",
            "carcinoma",
            "malignant",
            "mct",
            "ak",
        ]

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
        filename = str(row["midas_file_name"])

        return image, label, filename


# ------------------------------------------------------------
# Transform
# ------------------------------------------------------------
def get_test_transform(image_size=224):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    return transforms.Compose([
        transforms.Resize((image_size, image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])


# ------------------------------------------------------------
# Model definitions supporting both checkpoint formats
# ------------------------------------------------------------
class ClassificationHead(nn.Module):
    """
    HPO checkpoint format:
        fc.head.1.weight
        fc.head.1.bias
    """
    def __init__(self, in_features, dropout=0.3):
        super().__init__()
        self.head = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1),
        )

    def forward(self, x):
        return self.head(x)


class WrappedResNetImageClassifier(nn.Module):
    """
    Longer-training checkpoint format:
        backbone.conv1.weight
        backbone.fc.1.weight
        backbone.fc.1.bias
    """
    def __init__(
        self,
        model_name="resnet50",
        dropout=0.3,
        freeze_backbone=False,
    ):
        super().__init__()

        if model_name == "resnet18":
            self.backbone = models.resnet18(weights=None)
        elif model_name == "resnet50":
            self.backbone = models.resnet50(weights=None)
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
        logits = self.backbone(x)
        return logits.view(-1)


def build_hpo_resnet_model(
    model_name="resnet50",
    dropout=0.3,
    freeze_backbone=False,
):
    """
    HPO checkpoint format:
        conv1.weight
        layer1...
        fc.head.1.weight
        fc.head.1.bias
    """
    if model_name == "resnet18":
        model = models.resnet18(weights=None)
    elif model_name == "resnet50":
        model = models.resnet50(weights=None)
    else:
        raise ValueError(f"Unsupported model_name: {model_name}")

    in_features = model.fc.in_features

    if freeze_backbone:
        for param in model.parameters():
            param.requires_grad = False

    model.fc = ClassificationHead(in_features, dropout=dropout)

    return model


def build_model_for_checkpoint(
    state_dict,
    model_name="resnet50",
    dropout=0.3,
    freeze_backbone=False,
):
    """
    Automatically chooses the correct model wrapper depending on checkpoint keys.

    Supports:
    1. HPO checkpoint:
        conv1.weight
        layer1...
        fc.head.1.weight

    2. Longer-training checkpoint:
        backbone.conv1.weight
        backbone.layer1...
        backbone.fc.1.weight
    """
    keys = list(state_dict.keys())

    if any(k.startswith("backbone.") for k in keys):
        print("Detected checkpoint format: wrapped ResNet with backbone.* keys")
        model = WrappedResNetImageClassifier(
            model_name=model_name,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )
    else:
        print("Detected checkpoint format: raw HPO ResNet keys")
        model = build_hpo_resnet_model(
            model_name=model_name,
            dropout=dropout,
            freeze_backbone=freeze_backbone,
        )

    return model


# ------------------------------------------------------------
# Evaluation
# ------------------------------------------------------------
@torch.no_grad()
def evaluate(model, loader, device, threshold=0.5):
    model.eval()

    all_labels = []
    all_probs = []
    all_preds = []
    all_files = []

    for images, labels, filenames in loader:
        images = images.to(device)

        logits = model(images)
        logits = logits.view(-1)

        probs = torch.sigmoid(logits).detach().cpu().numpy()

        labels_np = labels.numpy()
        preds_np = (probs >= threshold).astype(int)

        all_labels.extend(labels_np.tolist())
        all_probs.extend(probs.tolist())
        all_preds.extend(preds_np.tolist())
        all_files.extend(list(filenames))

    y_true = np.array(all_labels).astype(int)
    y_prob = np.array(all_probs)
    y_pred = np.array(all_preds).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred, zero_division=0),
        "f1": f1_score(y_true, y_pred, zero_division=0),
        "auc": roc_auc_score(y_true, y_prob) if len(np.unique(y_true)) > 1 else float("nan"),
        "threshold": threshold,
        "n_samples": int(len(y_true)),
        "n_negative": int((y_true == 0).sum()),
        "n_positive": int((y_true == 1).sum()),
    }

    cm = confusion_matrix(y_true, y_pred)

    predictions = pd.DataFrame({
        "midas_file_name": all_files,
        "true_label": y_true,
        "pred_label": y_pred,
        "prob_positive": y_prob,
    })

    report = classification_report(
        y_true,
        y_pred,
        target_names=["class_0", "class_1"],
        zero_division=0,
    )

    return metrics, cm, predictions, report


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    # --------------------------------------------------------
    # Device selection
    # --------------------------------------------------------
    if torch.cuda.is_available():
        device = torch.device("cuda")
    elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        device = torch.device("mps")
    else:
        device = torch.device("cpu")

    print("Using device:", device)

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # --------------------------------------------------------
    # Load checkpoint
    # --------------------------------------------------------
    checkpoint = torch.load(args.checkpoint_path, map_location=device)

    if isinstance(checkpoint, dict) and "config" in checkpoint:
        train_config = checkpoint["config"]
        print("Loaded config from checkpoint:")
        print(json.dumps(train_config, indent=4, default=str))
    else:
        train_config = {}
        print("No config found in checkpoint. Using CLI/default values.")

    if isinstance(checkpoint, dict) and "model_state_dict" in checkpoint:
        state_dict = checkpoint["model_state_dict"]
    else:
        state_dict = checkpoint

    # --------------------------------------------------------
    # Model settings
    # --------------------------------------------------------
    model_name = args.model_name or train_config.get("model_name", "resnet50")
    dropout = args.dropout if args.dropout is not None else train_config.get("dropout", 0.3)
    freeze_backbone = args.freeze_backbone or bool(train_config.get("freeze_backbone", False))
    image_size = args.image_size if args.image_size is not None else int(train_config.get("image_size", 224))

    print("\nModel settings:")
    print(f"model_name: {model_name}")
    print(f"dropout: {dropout}")
    print(f"freeze_backbone: {freeze_backbone}")
    print(f"image_size: {image_size}")

    # --------------------------------------------------------
    # Build matching model and load weights
    # --------------------------------------------------------
    model = build_model_for_checkpoint(
        state_dict=state_dict,
        model_name=model_name,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    ).to(device)

    model.load_state_dict(state_dict)
    print("\nCheckpoint loaded successfully.")

    # --------------------------------------------------------
    # Dataset / loader
    # --------------------------------------------------------
    test_transform = get_test_transform(image_size=image_size)

    test_ds = MIDASImageOnlyDataset(
        csv_path=args.test_csv,
        image_root=args.image_root,
        transform=test_transform,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    print(f"Test samples: {len(test_ds)}")

    # --------------------------------------------------------
    # Evaluate
    # --------------------------------------------------------
    metrics, cm, predictions, report = evaluate(
        model=model,
        loader=test_loader,
        device=device,
        threshold=args.threshold,
    )

    print("\n" + "=" * 80)
    print("TEST METRICS")
    print("=" * 80)

    for k, v in metrics.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")

    print("\nConfusion matrix:")
    print(cm)

    print("\nClassification report:")
    print(report)

    # --------------------------------------------------------
    # Save outputs
    # --------------------------------------------------------
    metrics_path = output_dir / "test_metrics.json"
    cm_path = output_dir / "test_confusion_matrix.csv"
    pred_path = output_dir / "test_predictions.csv"
    report_path = output_dir / "test_classification_report.txt"

    with open(metrics_path, "w") as f:
        json.dump(metrics, f, indent=4)

    pd.DataFrame(
        cm,
        index=["true_0", "true_1"],
        columns=["pred_0", "pred_1"],
    ).to_csv(cm_path)

    predictions.to_csv(pred_path, index=False)

    with open(report_path, "w") as f:
        f.write(report)

    print("\nSaved:")
    print(metrics_path)
    print(cm_path)
    print(pred_path)
    print(report_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--test_csv", type=str, required=True)
    parser.add_argument("--image_root", type=str, required=True)
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)

    parser.add_argument("--model_name", type=str, default="resnet18", choices=["resnet18", "resnet50"])
    parser.add_argument("--image_size", type=int, default=None)
    parser.add_argument("--dropout", type=float, default=None)
    parser.add_argument("--freeze_backbone", action="store_true")

    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--threshold", type=float, default=0.5)

    args = parser.parse_args()
    main(args)