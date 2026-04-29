import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms, models
import cv2


MODEL_PATH = "image_only_baseline/model/best_image_only_model.pth"
MODEL_NAME = "resnet18"

IMAGE_DIR = "MRA-MIDAS/midasmultimodalimagedatasetforaibasedskincancer/"
LABEL_CSV_PATH = "manifests_record_split/test.csv"

OUTPUT_DIR = "image_only_baseline/gradcam_image_only"


class GradCAM:
    def __init__(self, model, target_layer):
        self.model = model
        self.target_layer = target_layer
        self.features = None
        self.gradients = None

        self.fwd_handle = target_layer.register_forward_hook(self.forward_hook)
        self.bwd_handle = target_layer.register_full_backward_hook(self.backward_hook)

    def remove_hooks(self):
        self.fwd_handle.remove()
        self.bwd_handle.remove()

    def forward_hook(self, module, input, output):
        self.features = output.detach()

    def backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0].detach()

    def generate(self, input_tensor, class_idx):
        self.model.zero_grad()

        logits = self.model(input_tensor)

        if logits.ndim == 1:
            logits = logits.unsqueeze(1)

        if logits.shape[1] == 1:
            target_score = logits[0, 0] if class_idx == 1 else -logits[0, 0]
        else:
            target_score = logits[0, class_idx]

        target_score.backward()

        if self.gradients is None or self.features is None:
            return None

        weights = self.gradients.mean(dim=(2, 3), keepdim=True)
        gradcam = (weights * self.features).sum(dim=1, keepdim=True)
        gradcam = F.relu(gradcam).squeeze().cpu().numpy()

        denom = gradcam.max() - gradcam.min()
        if denom > 1e-8:
            gradcam = (gradcam - gradcam.min()) / denom
        else:
            gradcam = np.zeros_like(gradcam)

        return gradcam


class WrappedResNet(nn.Module):
    def __init__(self, dropout=0.2):
        super().__init__()

        self.backbone = models.resnet18(weights=None)
        in_features = self.backbone.fc.in_features

        self.backbone.fc = nn.Sequential(
            nn.Dropout(dropout),
            nn.Linear(in_features, 1)
        )

    def forward(self, x):
        return self.backbone(x)


def load_model(model_path, model_name, device):
    checkpoint = torch.load(model_path, map_location=device)

    config = checkpoint.get("config", {})
    dropout = config.get("dropout", 0.2)

    if model_name == "resnet18":
        model = WrappedResNet(dropout=dropout)
    else:
        raise ValueError(f"Unknown model: {model_name}")

    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.to(device)
    model.eval()

    return model


def get_target_layer(model, model_name):
    if model_name == "resnet18":
        return model.backbone.layer4[-1].conv2
    else:
        raise ValueError(f"Unknown model: {model_name}")


def load_manifest(csv_path):
    df = pd.read_csv(csv_path)

    df["midas_file_name"] = df["midas_file_name"].astype(str).str.strip()

    label_map = {
        0: "benign",
        1: "malignant"
    }

    df["label_str"] = df["label"].map(label_map)

    return df


def get_manifest_image_paths(df_manifest, image_dir):
    image_dir = Path(image_dir)

    image_paths = []

    for _, row in df_manifest.iterrows():
        file_name = row["midas_file_name"]
        image_path = image_dir / file_name

        if image_path.exists():
            image_paths.append((image_path, row["label_str"]))
            continue

        stem = Path(file_name).stem
        suffix = Path(file_name).suffix

        cropped_file_name = f"{stem}_cropped{suffix}"
        cropped_image_path = image_dir / cropped_file_name

        if cropped_image_path.exists():
            image_paths.append((cropped_image_path, row["label_str"]))
        else:
            print(f"[MISSING] {image_path} or {cropped_image_path}")

    return image_paths


def get_confusion_type(ground_truth_label, pred_label):
    if ground_truth_label == "malignant" and pred_label == "malignant":
        return "TP"
    elif ground_truth_label == "benign" and pred_label == "benign":
        return "TN"
    elif ground_truth_label == "benign" and pred_label == "malignant":
        return "FP"
    elif ground_truth_label == "malignant" and pred_label == "benign":
        return "FN"
    else:
        return "UNK"


def create_visualization(
    original_img,
    gradcam_heatmap,
    ground_truth,
    pred_label,
    prob_malignant,
    confusion_type,
    save_path
):
    heatmap_colored = cv2.applyColorMap(
        (gradcam_heatmap * 255).astype(np.uint8),
        cv2.COLORMAP_JET
    )
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)

    overlay = (0.6 * original_img + 0.4 * heatmap_colored).astype(np.uint8)

    fig, axes = plt.subplots(1, 2, figsize=(12, 5))

    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")

    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Overlay", fontsize=12)
    axes[1].axis("off")

    title = (
        f"{confusion_type} | "
        f"GT: {ground_truth} | "
        f"Pred: {pred_label} | "
        f"P(malignant): {prob_malignant:.4f}"
    )

    fig.suptitle(title, fontsize=13, fontweight="bold")

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def process_image(model, gradcam_handler, image_path, ground_truth_label, device):
    class_names = ["benign", "malignant"]

    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)

    pil_img = Image.open(image_path).convert("RGB")
    pil_resized = pil_img.resize((224, 224))
    original_array = np.array(pil_resized)

    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        ),
    ])

    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)

    with torch.no_grad():
        logits = model(input_tensor)

        if logits.ndim == 1:
            logits = logits.unsqueeze(1)

    if logits.shape[1] == 1:
        prob_malignant = torch.sigmoid(logits[0, 0]).item()
        pred_idx = 1 if prob_malignant > 0.5 else 0
    else:
        probs = torch.softmax(logits, dim=1)
        prob_malignant = probs[0, 1].item()
        pred_idx = logits.argmax(dim=1).item()

    pred_label = class_names[pred_idx]
    confusion_type = get_confusion_type(ground_truth_label, pred_label)

    gradcam_heatmap = gradcam_handler.generate(input_tensor, pred_idx)

    if gradcam_heatmap is None:
        return None

    gradcam_heatmap = cv2.resize(gradcam_heatmap, (224, 224))

    output_filename = f"{confusion_type}_{image_path.stem}_gradcam.png"
    output_path = Path(OUTPUT_DIR) / output_filename

    create_visualization(
        original_array,
        gradcam_heatmap,
        ground_truth_label,
        pred_label,
        prob_malignant,
        confusion_type,
        str(output_path)
    )

    return output_path, pred_label, prob_malignant, confusion_type


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, MODEL_NAME, device)

    print("Setting up Grad-CAM...")
    target_layer = get_target_layer(model, MODEL_NAME)
    gradcam = GradCAM(model, target_layer)

    print(f"Loading manifest from {LABEL_CSV_PATH}...")
    df_manifest = load_manifest(LABEL_CSV_PATH)

    manifest_image_paths = get_manifest_image_paths(df_manifest, IMAGE_DIR)

    print(f"Images listed in manifest: {len(df_manifest)}")
    print(f"Existing images found: {len(manifest_image_paths)}")

    success_count = 0

    counts = {
        "TP": 0,
        "TN": 0,
        "FP": 0,
        "FN": 0,
        "UNK": 0
    }

    for i, (image_path, ground_truth_label) in enumerate(manifest_image_paths):
        try:
            result = process_image(
                model,
                gradcam,
                image_path,
                ground_truth_label,
                device
            )

            if result:
                success_count += 1

                output_path, pred_label, prob_malignant, confusion_type = result
                counts[confusion_type] += 1

                status = "✓" if ground_truth_label == pred_label else "✗"

                print(
                    f"[{status} {success_count:4d}] "
                    f"{image_path.name}: "
                    f"GT={ground_truth_label:9s} "
                    f"Pred={pred_label:9s} "
                    f"Type={confusion_type:3s} "
                    f"P(mal)={prob_malignant:.4f}"
                )

        except Exception as e:
            print(f"[ERROR] {image_path.name}: {str(e)[:100]}")

    gradcam.remove_hooks()

    print("\nFinished.")
    print(f"Generated {success_count} Grad-CAM visualizations")
    print(f"Saved to: {OUTPUT_DIR}/")

    print("\nConfusion counts:")
    print(f"TP: {counts['TP']}")
    print(f"TN: {counts['TN']}")
    print(f"FP: {counts['FP']}")
    print(f"FN: {counts['FN']}")
    print(f"UNK: {counts['UNK']}")


if __name__ == "__main__":
    main()