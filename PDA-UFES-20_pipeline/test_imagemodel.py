import sys
from pathlib import Path
from collections import OrderedDict

import torch
import torch.nn as nn
import pandas as pd
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
import torchvision.models as models
from sklearn.metrics import roc_auc_score, accuracy_score, balanced_accuracy_score, f1_score


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.append(str(PROJECT_ROOT))

device = "cuda" if torch.cuda.is_available() else "cpu"


class ImageOnlyModel(nn.Module):
    def __init__(self):
        super().__init__()

        self.resnet = models.resnet18(weights=None)

        self.hidden = nn.Sequential(
            *list(self.resnet.children())[:-1]
        )

        self.out = nn.Sequential(
            nn.Dropout(0.0),
            nn.Linear(512, 1)
        )

    def forward(self, x):
        x = self.hidden(x)
        x = torch.flatten(x, 1)
        x = self.out(x)
        return x


class PADImageOnlyDataset(Dataset):
    LABEL_MAP = {
        "ACK": 1,
        "BCC": 1,
        "SCC": 1,
        "MEL": 1,
        "NEV": 0,
        "SEK": 0,
    }

    def __init__(self, file_path, image_root, transform=None):
        self.image_root = Path(image_root)
        self.transform = transform

        df = pd.read_csv(file_path)
        df = df[["img_id", "diagnostic"]].copy()

        df["diagnostic"] = df["diagnostic"].astype(str).str.upper().str.strip()
        df["label"] = df["diagnostic"].map(self.LABEL_MAP)

        df = df.dropna(subset=["img_id", "diagnostic", "label"]).reset_index(drop=True)

        self.df = df

        print("PAD image-only samples:", len(self.df))
        print("PAD label balance:", self.df["label"].value_counts().to_dict())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        img_path = self.image_root / Path(str(row["img_id"])).name

        if img_path.exists():
            image = Image.open(img_path).convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform is not None:
            image = self.transform(image)

        label = torch.tensor(row["label"], dtype=torch.float32)

        return image, label


transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

checkpoint_path = PROJECT_ROOT / "Image_Eva" / "checkpoints_opt" / "trial_10.pt"

print("Loading checkpoint from:", checkpoint_path)

state_dict = torch.load(
    checkpoint_path,
    map_location=device,
    weights_only=False
)

model = ImageOnlyModel()

missing, unexpected = model.load_state_dict(state_dict, strict=False)

print("Missing keys:", missing)
print("Unexpected keys:", unexpected)

model = model.to(device)
model.eval()

pad_ds = PADImageOnlyDataset(
    file_path=PROJECT_ROOT / "PDA-UFES-20" / "metadata.csv",
    image_root=PROJECT_ROOT / "PDA-UFES-20" / "images",
    transform=transform,
)

pad_loader = DataLoader(
    pad_ds,
    batch_size=32,
    shuffle=False,
    num_workers=0,
)

all_probs = []
all_labels = []

print("Starting evaluation...")

with torch.no_grad():
    for i, (images, labels) in enumerate(pad_loader):
        images = images.to(device)
        labels = labels.to(device)

        logits = model(images)

        if isinstance(logits, tuple):
            logits = logits[0]

        probs = torch.sigmoid(logits).view(-1)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())

        if i % 10 == 0:
            print(f"Processed batch {i}/{len(pad_loader)}")

preds = [1 if p >= 0.5 else 0 for p in all_probs]

print("PAD samples:", len(all_labels))
print("AUC:", roc_auc_score(all_labels, all_probs))
print("Accuracy:", accuracy_score(all_labels, preds))
print("Balanced accuracy:", balanced_accuracy_score(all_labels, preds))
print("F1:", f1_score(all_labels, preds))