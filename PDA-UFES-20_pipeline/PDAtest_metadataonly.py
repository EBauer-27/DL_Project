import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from preprocessing_PDA import PADDataset
from tab_transformer_pytorch import TabTransformer
import torch.nn as nn


device = "cuda" if torch.cuda.is_available() else "cpu"


checkpoint = torch.load(
    PROJECT_ROOT / "metadata_only_baseline" / "model" / "best_tabtransformer.pth",
    map_location=device,
    weights_only=False,
)

scaler = checkpoint["scaler"]
cat_maps = checkpoint["cat_maps"]


model = TabTransformer(
    categories=checkpoint["categories"],
    num_continuous=checkpoint["num_continuous"],
    dim=32,
    dim_out=1,
    depth=6,
    heads=8,
    attn_dropout=0.1,
    ff_dropout=0.1,
    mlp_hidden_mults=(4, 2),
    mlp_act=nn.ReLU(),
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()


pad_ds = PADDataset(
    file_path=str(PROJECT_ROOT / "PDA-UFES-20" / "metadata.csv"),
    image_root=str(PROJECT_ROOT / "PDA-UFES-20" / "images"),
    transform=None,
    use_images=False,
    scaler=scaler,
    cat_maps=cat_maps,
    drop_incomplete=True,
    extract_zips=False
)

import pandas as pd

N_PER_CLASS = 100
RANDOM_STATE = 42

df = pad_ds.df.copy()

benign_df = df[df["label"] == 0]
malignant_df = df[df["label"] == 1]

print("\nOriginal PAD label counts:")
print(df["label"].value_counts())

if len(benign_df) < N_PER_CLASS or len(malignant_df) < N_PER_CLASS:
    raise ValueError(
        f"Not enough samples. "
        f"Benign: {len(benign_df)}, malignant: {len(malignant_df)}, "
        f"requested per class: {N_PER_CLASS}"
    )

balanced_df = pd.concat([
    benign_df.sample(n=N_PER_CLASS, random_state=RANDOM_STATE),
    malignant_df.sample(n=N_PER_CLASS, random_state=RANDOM_STATE),
])

balanced_df = balanced_df.sample(frac=1, random_state=RANDOM_STATE).reset_index(drop=True)

pad_ds.df = balanced_df

print("\nBalanced PAD label counts:")
print(pad_ds.df["label"].value_counts())
print("Balanced PAD samples:", len(pad_ds))

pad_loader = DataLoader(
    pad_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)


all_probs = []
all_labels = []

with torch.no_grad():
    for batch in pad_loader:
        x_categ = batch["x_categ"].to(device)
        x_cont = batch["x_cont"].to(device)
        labels = batch["label"].to(device)

        logits = model(x_categ, x_cont)
        probs = torch.sigmoid(logits).view(-1)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())


preds = [1 if p >= 0.5 else 0 for p in all_probs]

print("PAD samples:", len(all_labels))
print("AUC:", roc_auc_score(all_labels, all_probs))
print("Accuracy:", accuracy_score(all_labels, preds))
print("Balanced accuracy:", balanced_accuracy_score(all_labels, preds))
print("F1:", f1_score(all_labels, preds))

from sklearn.metrics import confusion_matrix, precision_score, recall_score

cm = confusion_matrix(all_labels, preds)
tn, fp, fn, tp = cm.ravel()

precision = precision_score(all_labels, preds, zero_division=0)
recall = recall_score(all_labels, preds, zero_division=0)

print("\nConfusion matrix:")
print(cm)

print("\nConfusion counts:")
print("TP:", tp)
print("TN:", tn)
print("FP:", fp)
print("FN:", fn)

print("\nExtra metrics:")
print("Precision:", precision)
print("Recall / Sensitivity:", recall)