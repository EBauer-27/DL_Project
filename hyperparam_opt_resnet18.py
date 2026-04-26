import os
import copy
import optuna
import pandas as pd

from PIL import Image

import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as T

from torch.utils.data import Dataset, DataLoader
from torchvision.models import resnet18, ResNet18_Weights


# =========================
# CONFIG
# =========================

TRAIN_MANIFEST = "manifests/train.csv"
VAL_MANIFEST = "manifests/val.csv"
TEST_MANIFEST = "manifests/test.csv"

IMAGE_PATH_COLUMN = "midas_file_name"
LABEL_COLUMN = "label"
IMAGE_ROOT_DIR = "MRA-MIDAS/cached_images"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

IMG_SIZE = 224
NUM_WORKERS = 4

NUM_EPOCHS = 15
N_TRIALS = 30

BEST_MODEL_PATH = "image_only_baseline/best_resnet18_optuna_checkpoint.pt"


# =========================
# DATASET
# =========================

class ImageModelHyperparamoptimization(Dataset):
    def __init__(self, manifest_path, transform=None, class_to_idx=None):
        self.df = pd.read_csv(manifest_path)
        self.transform = transform

        if IMAGE_PATH_COLUMN not in self.df.columns:
            raise ValueError(f"Missing column: {IMAGE_PATH_COLUMN}")

        if LABEL_COLUMN not in self.df.columns:
            raise ValueError(f"Missing column: {LABEL_COLUMN}")

        if class_to_idx is None:
            classes = sorted(self.df[LABEL_COLUMN].unique())
            self.class_to_idx = {cls_name: idx for idx, cls_name in enumerate(classes)}
        else:
            self.class_to_idx = class_to_idx

        self.classes = list(self.class_to_idx.keys())

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        image_path = row[IMAGE_PATH_COLUMN]

        if not os.path.isabs(image_path):
            image_path = os.path.join(IMAGE_ROOT_DIR, image_path)
            
        label_name = row[LABEL_COLUMN]

        image = Image.open(image_path).convert("RGB")
        label = self.class_to_idx[label_name]

        if self.transform is not None:
            image = self.transform(image)

        return image, label


# =========================
# TRANSFORMS
# =========================

train_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.RandomHorizontalFlip(),
    T.RandomVerticalFlip(),
    T.RandomRotation(20),
    T.ColorJitter(
        brightness=0.15,
        contrast=0.15,
        saturation=0.10,
        hue=0.03,
    ),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])

eval_transform = T.Compose([
    T.Resize((IMG_SIZE, IMG_SIZE)),
    T.ToTensor(),
    T.Normalize(
        mean=[0.485, 0.456, 0.406],
        std=[0.229, 0.224, 0.225],
    ),
])


# =========================
# DATASETS
# =========================

train_dataset = ImageModelHyperparamoptimization(
    TRAIN_MANIFEST,
    transform=train_transform,
)

val_dataset = ImageModelHyperparamoptimization(
    VAL_MANIFEST,
    transform=eval_transform,
    class_to_idx=train_dataset.class_to_idx,
)

test_dataset = ImageModelHyperparamoptimization(
    TEST_MANIFEST,
    transform=eval_transform,
    class_to_idx=train_dataset.class_to_idx,
)

NUM_CLASSES = len(train_dataset.classes)

print("Classes:", train_dataset.classes)
print("Class mapping:", train_dataset.class_to_idx)
print("Number of classes:", NUM_CLASSES)


# =========================
# MODEL
# =========================

def create_model(dropout):
    model = resnet18(weights=ResNet18_Weights.IMAGENET1K_V1)

    in_features = model.fc.in_features

    model.fc = nn.Sequential(
        nn.Dropout(dropout),
        nn.Linear(in_features, NUM_CLASSES),
    )

    return model


# =========================
# TRAINING
# =========================

def train_one_epoch(model, loader, optimizer, criterion):
    model.train()

    total_loss = 0.0
    correct = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)

    return epoch_loss, epoch_acc


@torch.no_grad()
def evaluate(model, loader, criterion):
    model.eval()

    total_loss = 0.0
    correct = 0

    for images, labels in loader:
        images = images.to(DEVICE)
        labels = labels.to(DEVICE)

        outputs = model(images)
        loss = criterion(outputs, labels)

        total_loss += loss.item() * images.size(0)
        correct += (outputs.argmax(dim=1) == labels).sum().item()

    epoch_loss = total_loss / len(loader.dataset)
    epoch_acc = correct / len(loader.dataset)

    return epoch_loss, epoch_acc


# =========================
# OPTUNA OBJECTIVE
# =========================

def objective(trial):
    batch_size = trial.suggest_categorical("batch_size", [16, 32, 64])

    lr = trial.suggest_float(
        "lr",
        1e-5,
        1e-3,
        log=True,
    )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-2,
        log=True,
    )

    dropout = trial.suggest_float(
        "dropout",
        0.0,
        0.5,
    )

    label_smoothing = trial.suggest_float(
        "label_smoothing",
        0.0,
        0.15,
    )

    optimizer_name = trial.suggest_categorical(
        "optimizer",
        ["AdamW", "SGD"],
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True,
    )

    model = create_model(dropout).to(DEVICE)

    criterion = nn.CrossEntropyLoss(
        label_smoothing=label_smoothing,
    )

    if optimizer_name == "AdamW":
        optimizer = optim.AdamW(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
        )
    else:
        momentum = trial.suggest_float("momentum", 0.8, 0.99)

        optimizer = optim.SGD(
            model.parameters(),
            lr=lr,
            momentum=momentum,
            weight_decay=weight_decay,
            nesterov=True,
        )

    scheduler = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=NUM_EPOCHS,
    )

    best_val_acc = 0.0

    for epoch in range(NUM_EPOCHS):
        train_loss, train_acc = train_one_epoch(
            model,
            train_loader,
            optimizer,
            criterion,
        )

        val_loss, val_acc = evaluate(
            model,
            val_loader,
            criterion,
        )

        scheduler.step()

        trial.report(val_acc, epoch)

        if trial.should_prune():
            raise optuna.exceptions.TrialPruned()

        best_val_acc = max(best_val_acc, val_acc)

        print(
            f"Trial {trial.number} | "
            f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Val Loss: {val_loss:.4f} | "
            f"Val Acc: {val_acc:.4f}"
        )

    return best_val_acc


# =========================
# RUN OPTUNA
# =========================

study = optuna.create_study(
    direction="maximize",
    pruner=optuna.pruners.MedianPruner(
        n_startup_trials=5,
        n_warmup_steps=3,
    ),
)

study.optimize(
    objective,
    n_trials=N_TRIALS,
)

print("\nBest Optuna trial:")
print("Best validation accuracy:", study.best_value)
print("Best hyperparameters:")

for key, value in study.best_params.items():
    print(f"{key}: {value}")


# =========================
# FINAL TRAINING WITH BEST PARAMS
# =========================

best_params = study.best_params

final_batch_size = best_params["batch_size"]

train_loader = DataLoader(
    train_dataset,
    batch_size=final_batch_size,
    shuffle=True,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

val_loader = DataLoader(
    val_dataset,
    batch_size=final_batch_size,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

test_loader = DataLoader(
    test_dataset,
    batch_size=final_batch_size,
    shuffle=False,
    num_workers=NUM_WORKERS,
    pin_memory=True,
)

final_model = create_model(
    dropout=best_params["dropout"],
).to(DEVICE)

criterion = nn.CrossEntropyLoss(
    label_smoothing=best_params["label_smoothing"],
)

if best_params["optimizer"] == "AdamW":
    optimizer = optim.AdamW(
        final_model.parameters(),
        lr=best_params["lr"],
        weight_decay=best_params["weight_decay"],
    )
else:
    optimizer = optim.SGD(
        final_model.parameters(),
        lr=best_params["lr"],
        momentum=best_params["momentum"],
        weight_decay=best_params["weight_decay"],
        nesterov=True,
    )

scheduler = optim.lr_scheduler.CosineAnnealingLR(
    optimizer,
    T_max=NUM_EPOCHS,
)

best_val_acc = 0.0
best_state_dict = None

for epoch in range(NUM_EPOCHS):
    train_loss, train_acc = train_one_epoch(
        final_model,
        train_loader,
        optimizer,
        criterion,
    )

    val_loss, val_acc = evaluate(
        final_model,
        val_loader,
        criterion,
    )

    scheduler.step()

    if val_acc > best_val_acc:
        best_val_acc = val_acc
        best_state_dict = copy.deepcopy(final_model.state_dict())

        torch.save(
            {
                "model_state_dict": best_state_dict,
                "best_params": best_params,
                "classes": train_dataset.classes,
                "class_to_idx": train_dataset.class_to_idx,
                "val_accuracy": best_val_acc,
                "image_size": IMG_SIZE,
                "model_name": "resnet18",
            },
            BEST_MODEL_PATH,
        )

    print(
        f"Final training | "
        f"Epoch {epoch + 1}/{NUM_EPOCHS} | "
        f"Train Loss: {train_loss:.4f} | "
        f"Train Acc: {train_acc:.4f} | "
        f"Val Loss: {val_loss:.4f} | "
        f"Val Acc: {val_acc:.4f}"
    )


# =========================
# FINAL TEST EVALUATION
# =========================

checkpoint = torch.load(
    BEST_MODEL_PATH,
    map_location=DEVICE,
)

final_model.load_state_dict(
    checkpoint["model_state_dict"]
)

test_loss, test_acc = evaluate(
    final_model,
    test_loader,
    criterion,
)

print("\nFinal test result:")
print("Test loss:", test_loss)
print("Test accuracy:", test_acc)
print("Saved checkpoint to:", BEST_MODEL_PATH)