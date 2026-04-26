import torch
import numpy as np
from torch.utils.data import DataLoader
from torchvision import transforms
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score, recall_score
from data_processing.data import MIDASDataset
from image_only_baseline.Models import VggNet, ResNet18, GoogLeNet

IMG_ROOT = "data/MRA-MIDAS/midas_224_cache/"
TEST_PATH = "manifests/test.csv"
MODEL_PATH = "trial_10.pt"

test_tf = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

test_ds = MIDASDataset(
    file_path=TEST_PATH,
    image_root=IMG_ROOT,
    transform=test_tf,
    is_training=False
)

test_loader = DataLoader(test_ds, batch_size=32, shuffle=False)

model = VggNet()  # nur richtig, falls trial_10 wirklich VGG war
state = torch.load(MODEL_PATH, map_location="cpu")
model.load_state_dict(state)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

all_labels = []
all_probs = []

with torch.no_grad():
    for batch in test_loader:
        img = batch["image"].to(device)
        label = batch["label"].float()

        logits = model(img).squeeze(1)
        probs = torch.sigmoid(logits)

        all_probs.extend(probs.cpu().numpy())
        all_labels.extend(label.numpy())

all_labels = np.array(all_labels).astype(int)
all_probs = np.array(all_probs)
all_preds = (all_probs >= 0.5).astype(int)

metrics = {
    "AUC": roc_auc_score(all_labels, all_probs),
    "F1": f1_score(all_labels, all_preds),
    "Accuracy": accuracy_score(all_labels, all_preds),
    "Precision": precision_score(all_labels, all_preds),
    "Recall": recall_score(all_labels, all_preds),
}

print("\nTest Metrics")
print("=" * 30)
print(f"{'Metric':<12} {'Value':>10}")
print("-" * 30)

for name, value in metrics.items():
    print(f"{name:<12} {value:>10.4f}")

print("=" * 30)