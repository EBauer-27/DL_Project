import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

# --- PATHS ---
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from preprocessing_PDA import PADDataset
from multimodal_pipeline.multimodal import MultimodalSkinClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

DATA_ROOT = PROJECT_ROOT / "PDA-UFES-20"
METADATA_PATH = DATA_ROOT / "metadata.csv"
IMAGE_ROOT = DATA_ROOT / "images"
MODEL_PATH = PROJECT_ROOT / "multimodal_pipeline" / "model" / "best_multimodal_model_resnet18.pth"

print("Project root:", PROJECT_ROOT)
print("Metadata:", METADATA_PATH, METADATA_PATH.exists())
print("Images:", IMAGE_ROOT, IMAGE_ROOT.exists())
print("Model:", MODEL_PATH, MODEL_PATH.exists())

# --- SETTINGS ---
N_PER_CLASS = 100
RANDOM_STATE = 42
BATCH_SIZE = 32
THRESHOLD = 0.5

# --- TRANSFORMS ---
pad_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

# --- LOAD CHECKPOINT ---
checkpoint = torch.load(
    MODEL_PATH,
    map_location=device,
    weights_only=False,
)

scaler = checkpoint["scaler"]
cat_maps = checkpoint["cat_maps"]

# --- BUILD MODEL ---
model = MultimodalSkinClassifier(
    categories=checkpoint["categories"],
    num_continuous=checkpoint["num_continuous"],
)

model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# --- LOAD FULL PAD DATASET ---
pad_ds = PADDataset(
    file_path=str(METADATA_PATH),
    image_root=str(IMAGE_ROOT),
    transform=pad_transform,
    scaler=scaler,
    cat_maps=cat_maps,
    drop_incomplete=True,
    extract_zips=False,
)

# --- CREATE BALANCED SUBSET: 200 BENIGN + 200 MALIGNANT ---
df = pad_ds.df.copy()

benign_df = df[df["label"] == 0]
malignant_df = df[df["label"] == 1]

print("\nOriginal PAD label counts:")
print(df["label"].value_counts())

if len(benign_df) < N_PER_CLASS or len(malignant_df) < N_PER_CLASS:
    raise ValueError(
        f"Not enough samples for balanced subset. "
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

# --- DATA LOADER ---
pad_loader = DataLoader(
    pad_ds,
    batch_size=BATCH_SIZE,
    shuffle=False,
    num_workers=0,
)

# --- EVAL ---
all_probs = []
all_labels = []

with torch.no_grad():
    for batch in pad_loader:
        image = batch["image"].to(device)
        x_categ = batch["x_categ"].to(device)
        x_cont = batch["x_cont"].to(device)
        labels = batch["label"].to(device)

        logits = model(
            image=image,
            x_categ=x_categ,
            x_cont=x_cont,
        )

        probs = torch.sigmoid(logits).view(-1)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

all_probs = np.array(all_probs)
all_labels = np.array(all_labels)

preds = (all_probs >= THRESHOLD).astype(int)

# --- RESULTS ---
print("\n--- Balanced PAD Evaluation ---")
print("Threshold:", THRESHOLD)
print("PAD samples:", len(all_labels))

print("True class counts:", dict(zip(*np.unique(all_labels.astype(int), return_counts=True))))
print("Predicted class counts:", dict(zip(*np.unique(preds, return_counts=True))))

print("Probability min:", all_probs.min())
print("Probability max:", all_probs.max())
print("Probability mean:", all_probs.mean())

print("AUC:", roc_auc_score(all_labels, all_probs))
print("Accuracy:", accuracy_score(all_labels, preds))
print("Balanced accuracy:", balanced_accuracy_score(all_labels, preds))
print("F1:", f1_score(all_labels, preds))

print("\nConfusion matrix:")
print(confusion_matrix(all_labels, preds))