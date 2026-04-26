import os
import copy
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data_processing.data import MIDASMultimodalDataset
from .multimodal import MultimodalSkinClassifier
from metadata_only_baseline.utils import compute_metrics, set_seed 

# ============================================================
# Config
# ============================================================
IMG_ROOT = "data/MRA-MIDAS/midasmultimodalimagedatasetforaibasedskincancer/"
TRAIN_PATH = "manifests_record_split/train.csv"
VAL_PATH = "manifests_record_split/val.csv"

SAVE_DIR = "multimodal_pipeline/model"
SAVE_PATH = os.path.join(SAVE_DIR, "best_multimodal_model.pth")

BATCH_SIZE = 32
EPOCHS = 20
LR = 4.377982870865037e-05
WEIGHT_DECAY = 4.967600193908121e-05
SEED = 42
NUM_WORKERS = 0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")



# ============================================================
# Train / Validate
# ============================================================
def train_one_epoch(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_targets = []
    all_probs = []

    for batch in loader:
        image = batch["image"].to(device)
        x_categ = batch["x_categ"].to(device)
        x_cont = batch["x_cont"].to(device)
        y = batch["label"].to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        logits = model(image, x_categ, x_cont)
        loss = criterion(logits, y)

        loss.backward()
        optimizer.step()

        running_loss += loss.item() * y.size(0)

        probs = torch.sigmoid(logits).detach().cpu().numpy().flatten()
        targets = y.detach().cpu().numpy().flatten()

        all_probs.extend(probs.tolist())
        all_targets.extend(targets.tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_probs)
    return epoch_loss, metrics


@torch.no_grad()
def validate_one_epoch(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    all_targets = []
    all_probs = []

    for batch in loader:
        image = batch["image"].to(device)
        x_categ = batch["x_categ"].to(device)
        x_cont = batch["x_cont"].to(device)
        y = batch["label"].to(device).float().unsqueeze(1)

        logits = model(image, x_categ, x_cont)
        loss = criterion(logits, y)

        running_loss += loss.item() * y.size(0)

        probs = torch.sigmoid(logits).cpu().numpy().flatten()
        targets = y.cpu().numpy().flatten()

        all_probs.extend(probs.tolist())
        all_targets.extend(targets.tolist())

    epoch_loss = running_loss / len(loader.dataset)
    metrics = compute_metrics(all_targets, all_probs)
    return epoch_loss, metrics


# ============================================================
# Main
# ============================================================
def main():
    set_seed(SEED)
    os.makedirs(SAVE_DIR, exist_ok=True)

    print(f"Using device: {DEVICE}")

    # --------------------------------------------------------
    # Image transforms
    # --------------------------------------------------------
    train_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    val_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])

    # --------------------------------------------------------
    # Datasets
    # IMPORTANT:
    # - train fits scaler / category maps
    # - val reuses them
    # --------------------------------------------------------
    train_ds = MIDASMultimodalDataset(
        file_path=TRAIN_PATH,
        image_root=IMG_ROOT,
        transform=train_transforms,
        is_training=True,
        fit_scaler=True,
    )

    val_ds = MIDASMultimodalDataset(
        file_path=VAL_PATH,
        image_root=IMG_ROOT,
        transform=val_transforms,
        is_training=False,
        scaler=train_ds.scaler,
        cat_maps=train_ds.cat_maps,
    )

    print("categorical columns:", train_ds.categorical_cols)
    print("continuous columns:", train_ds.continuous_cols)
    print("categories:", train_ds.categories)
    print("num_continuous:", train_ds.num_continuous)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS,
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = MultimodalSkinClassifier(
        categories=train_ds.categories,
        num_continuous=train_ds.num_continuous,
        img_backbone="resnet18",
        pretrained=True,
        hidden_dim=256,
        tab_depth=2,
        tab_heads=8,
        fusion_heads=8,
        dropout=0.2,
    ).to(DEVICE)

    # --------------------------------------------------------
    # Loss / optimizer
    # --------------------------------------------------------
    train_labels = train_ds.df["label"].values
    pos_count = (train_labels == 1).sum()
    neg_count = (train_labels == 0).sum()

    if pos_count > 0:
        pos_weight = torch.tensor([neg_count / pos_count], dtype=torch.float32).to(DEVICE)
    else:
        pos_weight = torch.tensor([1.0], dtype=torch.float32).to(DEVICE)

    print(f"Positive samples: {pos_count}")
    print(f"Negative samples: {neg_count}")
    print(f"pos_weight: {pos_weight.item():.4f}")

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=LR,
        weight_decay=WEIGHT_DECAY,
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    best_val_auc = -float("inf")
    best_state = None
    patience = 5
    epochs_without_improvement = 0

    for epoch in range(EPOCHS):
        train_loss, train_metrics = train_one_epoch(model, train_loader, criterion, optimizer, DEVICE)
        val_loss, val_metrics = validate_one_epoch(model, val_loader, criterion, DEVICE)

        print(
            f"Epoch [{epoch + 1:03d}/{EPOCHS}] | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_metrics['accuracy']:.4f} | "
            f"Train Prec: {train_metrics['precision']:.4f} | "
            f"Train Rec: {train_metrics['recall']:.4f} | "
            f"Train AUC: {train_metrics['auc']:.4f} || "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_metrics['accuracy']:.4f} | "
            f"Val Prec: {val_metrics['precision']:.4f} | "
            f"Val Rec: {val_metrics['recall']:.4f} | "
            f"Val AUC: {val_metrics['auc']:.4f}"
        )

        if val_metrics["auc"] > best_val_auc:
            best_val_auc = val_metrics["auc"]
            best_state = copy.deepcopy(model.state_dict())
            epochs_without_improvement = 0

            torch.save(
                {
                    "model_state_dict": best_state,
                    "categories": tuple(train_ds.categories),
                    "num_continuous": int(train_ds.num_continuous),
                    "categorical_cols": train_ds.categorical_cols,
                    "continuous_cols": train_ds.continuous_cols,
                    "scaler": train_ds.scaler,
                    "cat_maps": train_ds.cat_maps,
                    "best_val_auc": best_val_auc,
                },
                SAVE_PATH,
            )
            print(f"Saved best model to {SAVE_PATH}")
        else:
            epochs_without_improvement += 1

        if epochs_without_improvement >= patience:
            print(f"Early stopping triggered after {epoch + 1} epochs.")
            break

    print(f"\nBest validation AUC: {best_val_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)


if __name__ == "__main__":
    main()