import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from data_processing.data import MIDASMultimodalDataset
from metadata_only_baseline.utils import compute_metrics
from .multimodal import MultimodalSkinClassifier

# ============================================================
# Config
# ============================================================
TEST_PATH = "manifests_record_split/test.csv"
IMG_ROOT = "data/MRA-MIDAS/midasmultimodalimagedatasetforaibasedskincancer/"

# ResNet18 multimodal checkpoint
MODEL_PATH = "multimodal_pipeline/model/best_multimodal_model_resnet18.pth"

BATCH_SIZE = 32
NUM_WORKERS = 0
IMAGE_SIZE = 224

if torch.cuda.is_available():
    DEVICE = torch.device("cuda")
elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
    DEVICE = torch.device("mps")
else:
    DEVICE = torch.device("cpu")


def detect_backbone_from_checkpoint(state_dict):
    """
    Detect image backbone from projection layer shape.

    ResNet18 outputs 512 features before projection.
    ResNet50 outputs 2048 features before projection.
    """
    proj_shape = state_dict["image_encoder.proj.weight"].shape

    if proj_shape[1] == 2048:
        return "resnet50"
    elif proj_shape[1] == 512:
        return "resnet18"
    else:
        raise ValueError(f"Unknown image encoder projection shape: {proj_shape}")


def main():
    print("Using device:", DEVICE)
    print("Loading checkpoint:", MODEL_PATH)

    checkpoint = torch.load(MODEL_PATH, map_location=DEVICE, weights_only=False)
    state_dict = checkpoint["model_state_dict"]

    categories = tuple(checkpoint["categories"])
    num_continuous = int(checkpoint["num_continuous"])
    scaler = checkpoint["scaler"]
    cat_maps = checkpoint["cat_maps"]

    img_backbone = detect_backbone_from_checkpoint(state_dict)

    print("Detected image backbone:", img_backbone)
    print("Best validation AUC:", checkpoint.get("best_val_auc"))
    print("Categories:", categories)
    print("Num continuous:", num_continuous)

    model = MultimodalSkinClassifier(
        categories=categories,
        num_continuous=num_continuous,
        img_backbone=img_backbone,
        pretrained=False,
        hidden_dim=256,
        tab_depth=2,
        tab_heads=8,
        fusion_heads=8,
        dropout=0.2,
    ).to(DEVICE)

    model.load_state_dict(state_dict)
    model.eval()

    test_transforms = transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225],
        ),
    ])

    test_ds = MIDASMultimodalDataset(
        file_path=TEST_PATH,
        image_root=IMG_ROOT,
        transform=test_transforms,
        is_training=False,
        scaler=scaler,
        cat_maps=cat_maps,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
    )

    print("Test samples:", len(test_ds))

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(DEVICE)
            x_categ = batch["x_categ"].to(DEVICE)
            x_cont = batch["x_cont"].to(DEVICE)
            labels = batch["label"].to(DEVICE).float().unsqueeze(1)

            logits = model(image, x_categ, x_cont)
            probs = torch.sigmoid(logits).view(-1).cpu().numpy()
            labels = labels.view(-1).cpu().numpy()

            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())

    metrics = compute_metrics(all_labels, all_probs)

    print("\nMultimodal Test Results:")
    print(f"Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Precision: {metrics['precision']:.4f}")
    print(f"Recall:    {metrics['recall']:.4f}")
    print(f"F1:        {metrics['f1']:.4f}")
    print(f"AUC:       {metrics['auc']:.4f}")


if __name__ == "__main__":
    main()