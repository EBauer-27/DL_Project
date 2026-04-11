import os
import copy
import random
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from tab_transformer_pytorch import TabTransformer
from .utils import compute_metrics, set_seed

from data_processing.data import MIDASTabularDataset

#python -m metadata_only_baseline.train
# ============================================================
# Config
# ============================================================
IMG_ROOT = "data/MRA-MIDAS/midasmultimodalimagedatasetforaibasedskincancer/"
TRAIN_PATH = "manifests/train.csv"
VAL_PATH = "manifests/val.csv"

BATCH_SIZE = 32
EPOCHS = 13
LR = 1e-4
WEIGHT_DECAY = 1e-5
SEED = 42
SAVE_PATH = "metadata_only_baseline/best_tabtransformer.pth"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ============================================================
# Train / Validate
# ============================================================
def train(model, loader, criterion, optimizer, device):
    model.train()

    running_loss = 0.0
    all_targets = []
    all_probs = []

    for x_categ, x_cont, y in loader:
        x_categ = x_categ.to(device)
        x_cont = x_cont.to(device)
        y = y.to(device).float().unsqueeze(1)

        optimizer.zero_grad()

        logits = model(x_categ, x_cont)
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

    for x_categ, x_cont, y in loader:
        x_categ = x_categ.to(device)
        x_cont = x_cont.to(device)
        y = y.to(device).float().unsqueeze(1)

        logits = model(x_categ, x_cont)
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

    print(f"Using device: {DEVICE}")

    # --------------------------------------------------------
    # Datasets
    # --------------------------------------------------------
    train_ds = MIDASTabularDataset(
        file_path=TRAIN_PATH,
        is_training=True,
        fit_scaler=True
    )

    val_ds = MIDASTabularDataset(
        file_path=VAL_PATH,
        is_training=False,
        scaler=train_ds.scaler,
        cat_maps=train_ds.cat_maps
    )

    print("categorical columns:", train_ds.categorical_cols)
    print("continuous columns:", train_ds.continuous_cols)
    print("categories:", train_ds.categories)
    print("num_continuous:", train_ds.num_continuous)

    train_loader = DataLoader(
        train_ds,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0
    )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = TabTransformer(
        categories=train_ds.categories,
        num_continuous=train_ds.num_continuous,
        dim=32,
        dim_out=1,
        depth=6,
        heads=8,
        attn_dropout=0.1,
        ff_dropout=0.1,
        mlp_hidden_mults=(4, 2),
        mlp_act=nn.ReLU(),
    ).to(DEVICE)

    # --------------------------------------------------------
    # Loss with class weighting
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
        weight_decay=WEIGHT_DECAY
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    best_val_auc = -float("inf")
    best_state = None

    for epoch in range(EPOCHS):
        train_loss, train_metrics = train(model, train_loader, criterion, optimizer, DEVICE)
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

            os.makedirs(os.path.dirname(SAVE_PATH), exist_ok=True)
            torch.save(
                {
                    "model_state_dict": best_state,
                    "categories": train_ds.categories,
                    "num_continuous": train_ds.num_continuous,
                    "categorical_cols": train_ds.categorical_cols,
                    "continuous_cols": train_ds.continuous_cols,
                    "scaler": train_ds.scaler,
                    "cat_maps": train_ds.cat_maps,
                    "best_val_auc": best_val_auc,
                },
                SAVE_PATH,
            )
            print(f"Saved best model to {SAVE_PATH}")

    print(f"\nBest validation AUC: {best_val_auc:.4f}")

    if best_state is not None:
        model.load_state_dict(best_state)


if __name__ == "__main__":
    main()