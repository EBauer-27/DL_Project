import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score
from torchvision import transforms

from data_processing.data import MIDASMultimodalDataset
from metadata_only_baseline.utils import compute_metrics
from .multimodal import MultimodalSkinClassifier

TEST_PATH = "manifests/test.csv"
IMG_ROOT = "data/MRA-MIDAS/midasmultimodalimagedatasetforaibasedskincancer/"
MODEL_PATH = "multimodal_pipeline/model/best_multimodal_model.pth"
BATCH_SIZE = 16

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    categories = tuple(checkpoint["categories"])
    num_continuous = int(checkpoint["num_continuous"])
    scaler = checkpoint["scaler"]
    cat_maps = checkpoint["cat_maps"]

    model = MultimodalSkinClassifier(
        categories=categories,
        num_continuous=num_continuous,
        img_backbone="resnet18",
        pretrained=True,
        hidden_dim=256,
        tab_depth=2,
        tab_heads=8,
        fusion_heads=8,
        dropout=0.2,
    ).to(device)

    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    test_transforms = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
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
        num_workers=0,
    )

    all_labels = []
    all_probs = []

    with torch.no_grad():
        for batch in test_loader:
            image = batch["image"].to(device)
            x_categ = batch["x_categ"].to(device)
            x_cont = batch["x_cont"].to(device)
            labels = batch["label"].to(device).float().unsqueeze(1)

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
    print(f"AUC:       {metrics['auc']:.4f}")


if __name__ == "__main__":
    main()