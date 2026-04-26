import torch
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score

from preprocessing_PDA import PADDataset
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

from preprocessing_PDA import PADDataset
from multimodal_pipeline.multimodal import MultimodalSkinClassifier

device = "cuda" if torch.cuda.is_available() else "cpu"

pad_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
])

# --- LOAD CHECKPOINT ---
checkpoint = torch.load(
    r"multimodal_pipeline\model\best_multimodal_model.pth",
    map_location=device,
    weights_only=False
)

scaler = checkpoint["scaler"]
cat_maps = checkpoint["cat_maps"]

# --- BUILD MODEL (WICHTIG!) ---
model = MultimodalSkinClassifier(
    categories=checkpoint["categories"],
    num_continuous=checkpoint["num_continuous"],
)

# --- LOAD WEIGHTS ---
model.load_state_dict(checkpoint["model_state_dict"])
model = model.to(device)
model.eval()

# --- DATA ---
pad_ds = PADDataset(
    file_path=r"PDA-UFES-20\metadata.csv",
    image_root=r"PDA-UFES-20\images",
    transform=pad_transform,
    scaler=scaler,
    cat_maps=cat_maps,
    drop_incomplete=True,
)

pad_loader = DataLoader(
    pad_ds,
    batch_size=32,
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

preds = [1 if p >= 0.5 else 0 for p in all_probs]

print("PAD samples:", len(all_labels))
print("AUC:", roc_auc_score(all_labels, all_probs))
print("Accuracy:", accuracy_score(all_labels, preds))
print("Balanced accuracy:", balanced_accuracy_score(all_labels, preds))
print("F1:", f1_score(all_labels, preds))