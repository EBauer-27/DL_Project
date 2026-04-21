import os
import math
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from matplotlib import cm
from torch.utils.data import DataLoader
from torchvision import transforms
from PIL import Image

from data_processing.data import MIDASMultimodalDataset
from multimodal_pipeline.multimodal import MultimodalSkinClassifier


TEST_PATH = "manifests/test.csv"
IMG_ROOT = "data/MRA-MIDAS/midasmultimodalimagedatasetforaibasedskincancer/"
MODEL_PATH = "multimodal_pipeline/model/best_multimodal_model.pth"
OUTPUT_DIR = "multimodal_pipeline/results/xai_cases"

BATCH_SIZE = 16
THRESHOLD = 0.5
MAX_EXAMPLES_PER_GROUP = 15  # change to 2 or 3 if you want more per TP/TN/FP/FN

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


# ------------------------------------------------------------
# Image normalization helpers
# ------------------------------------------------------------
IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def denormalize_image(img_tensor: torch.Tensor) -> torch.Tensor:
    img = img_tensor.detach().cpu() * IMAGENET_STD + IMAGENET_MEAN
    return img.clamp(0, 1)


# ------------------------------------------------------------
# Optional rough subtype text from midas_path
# ------------------------------------------------------------
def infer_subtype_from_path(path_value: str) -> str:
    if not isinstance(path_value, str):
        return "unknown"
    text = path_value.lower()
    if "melanoma" in text:
        return "melanoma"
    if "bcc" in text:
        return "bcc"
    if "scc" in text:
        return "scc"
    if "carcinoma" in text:
        return "carcinoma"
    if "ak" in text:
        return "ak"
    if "mct" in text:
        return "mct"
    return "benign/other"


# ------------------------------------------------------------
# Find last conv layer in image encoder
# ------------------------------------------------------------
def find_last_conv_layer(module: nn.Module):
    last_conv = None
    for _, m in module.named_modules():
        if isinstance(m, nn.Conv2d):
            last_conv = m
    if last_conv is None:
        raise RuntimeError("No Conv2d layer found for Grad-CAM.")
    return last_conv


# ------------------------------------------------------------
# Grad-CAM
# ------------------------------------------------------------
class GradCAM:
    def __init__(self, model: nn.Module, target_layer: nn.Module):
        self.model = model
        self.target_layer = target_layer
        self.activations = None
        self.gradients = None
        self.handles = []

        self.handles.append(
            target_layer.register_forward_hook(self._forward_hook)
        )
        self.handles.append(
            target_layer.register_full_backward_hook(self._backward_hook)
        )

    def _forward_hook(self, module, inputs, output):
        self.activations = output

    def _backward_hook(self, module, grad_input, grad_output):
        self.gradients = grad_output[0]

    def remove(self):
        for h in self.handles:
            h.remove()

    def generate(self, image, x_categ, x_cont):
        self.model.zero_grad()

        logits = self.model(image, x_categ, x_cont)
        score = logits.view(-1)[0]
        score.backward(retain_graph=True)

        grads = self.gradients[0]          # [C, H, W]
        acts = self.activations[0]         # [C, H, W]

        weights = grads.mean(dim=(1, 2))   # [C]
        cam = torch.zeros_like(acts[0])

        for c, w in enumerate(weights):
            cam += w * acts[c]

        cam = torch.relu(cam)
        cam = cam.detach().cpu().numpy()

        if cam.max() > cam.min():
            cam = (cam - cam.min()) / (cam.max() - cam.min())
        else:
            cam = np.zeros_like(cam)

        return cam


# ------------------------------------------------------------
# Overlay helper
# ------------------------------------------------------------
def overlay_cam_on_image(img_tensor: torch.Tensor, cam: np.ndarray, alpha: float = 0.4):
    img = denormalize_image(img_tensor).permute(1, 2, 0).numpy()

    h, w = img.shape[:2]
    cam_img = Image.fromarray((cam * 255).astype(np.uint8)).resize((w, h), Image.BILINEAR)
    cam_resized = np.array(cam_img).astype(np.float32) / 255.0

    heatmap = cm.jet(cam_resized)[..., :3]
    overlay = (1 - alpha) * img + alpha * heatmap
    overlay = np.clip(overlay, 0, 1)

    return img, overlay, cam_resized


# ------------------------------------------------------------
# Local metadata importance by single-feature perturbation
# For categorical features -> set to 0 (unknown token)
# For continuous features -> set to 0.0 (mean after standardization)
# ------------------------------------------------------------
@torch.no_grad()
def local_metadata_importance(model, image, x_categ, x_cont, feature_names, num_cat, device):
    model.eval()

    image_b = image.unsqueeze(0).to(device)
    x_categ_b = x_categ.unsqueeze(0).to(device)
    x_cont_b = x_cont.unsqueeze(0).to(device)

    baseline_prob = torch.sigmoid(model(image_b, x_categ_b, x_cont_b)).view(-1).item()

    deltas = []

    # categorical
    for i in range(num_cat):
        x_cat_mod = x_categ_b.clone()
        x_cont_mod = x_cont_b.clone()
        x_cat_mod[:, i] = 0  # unknown / masked
        prob = torch.sigmoid(model(image_b, x_cat_mod, x_cont_mod)).view(-1).item()
        deltas.append(baseline_prob - prob)

    # continuous
    for j in range(x_cont.shape[0]):
        x_cat_mod = x_categ_b.clone()
        x_cont_mod = x_cont_b.clone()
        x_cont_mod[:, j] = 0.0  # roughly mean after scaling
        prob = torch.sigmoid(model(image_b, x_cat_mod, x_cont_mod)).view(-1).item()
        deltas.append(baseline_prob - prob)

    df = pd.DataFrame({
        "feature": feature_names,
        "delta_prob": deltas,
        "abs_delta_prob": np.abs(deltas),
    }).sort_values("abs_delta_prob", ascending=False)

    return baseline_prob, df


# ------------------------------------------------------------
# Collect predictions on test set
# ------------------------------------------------------------
@torch.no_grad()
def collect_test_predictions(model, loader, device, threshold=0.5):
    model.eval()

    records = []
    running_idx = 0

    for batch in loader:
        images = batch["image"].to(device)
        x_categ = batch["x_categ"].to(device)
        x_cont = batch["x_cont"].to(device)
        labels = batch["label"].to(device).float()

        logits = model(images, x_categ, x_cont)
        probs = torch.sigmoid(logits).view(-1)
        preds = (probs >= threshold).float()

        batch_size = labels.shape[0]
        for i in range(batch_size):
            records.append({
                "dataset_idx": running_idx + i,
                "label": int(labels[i].item()),
                "pred": int(preds[i].item()),
                "prob": float(probs[i].item()),
            })

        running_idx += batch_size

    return pd.DataFrame(records)


# ------------------------------------------------------------
# Select representative cases
# ------------------------------------------------------------
def select_cases(pred_df, max_per_group=1):
    groups = {
        "TP": pred_df[(pred_df["label"] == 1) & (pred_df["pred"] == 1)].copy(),
        "TN": pred_df[(pred_df["label"] == 0) & (pred_df["pred"] == 0)].copy(),
        "FP": pred_df[(pred_df["label"] == 0) & (pred_df["pred"] == 1)].copy(),
        "FN": pred_df[(pred_df["label"] == 1) & (pred_df["pred"] == 0)].copy(),
    }

    selected = []

    for name, df in groups.items():
        if len(df) == 0:
            continue

        # pick most confident examples within each group
        if name in ["TP", "FP"]:
            df = df.sort_values("prob", ascending=False)
        else:
            df = df.sort_values("prob", ascending=True)

        chosen = df.head(max_per_group).copy()
        chosen["case_type"] = name
        selected.append(chosen)

    if not selected:
        return pd.DataFrame(columns=["dataset_idx", "label", "pred", "prob", "case_type"])

    return pd.concat(selected, ignore_index=True)


# ------------------------------------------------------------
# Plot one case
# ------------------------------------------------------------
def plot_case_figure(
    case_row,
    dataset,
    model,
    gradcam,
    output_path,
    device,
):
    idx = int(case_row["dataset_idx"])
    sample = dataset[idx]

    image = sample["image"]
    x_categ = sample["x_categ"]
    x_cont = sample["x_cont"]
    label = int(sample["label"].item())

    image_b = image.unsqueeze(0).to(device)
    x_categ_b = x_categ.unsqueeze(0).to(device)
    x_cont_b = x_cont.unsqueeze(0).to(device)

    model.zero_grad()
    prob = torch.sigmoid(model(image_b, x_categ_b, x_cont_b)).view(-1).item()
    pred = int(prob >= THRESHOLD)

    cam = gradcam.generate(image_b, x_categ_b, x_cont_b)
    img_np, overlay_np, _ = overlay_cam_on_image(image, cam, alpha=0.4)

    feature_names = list(dataset.categorical_cols) + list(dataset.continuous_cols)
    baseline_prob, local_imp_df = local_metadata_importance(
        model=model,
        image=image,
        x_categ=x_categ,
        x_cont=x_cont,
        feature_names=feature_names,
        num_cat=len(dataset.categorical_cols),
        device=device,
    )

    local_imp_top = local_imp_df.head(8).iloc[::-1]

    raw_row = dataset.df.iloc[idx]
    subtype = infer_subtype_from_path(raw_row.get("midas_path", ""))
    case_type = case_row["case_type"]

    fig = plt.figure(figsize=(15, 5))
    gs = fig.add_gridspec(1, 3, width_ratios=[1.0, 1.0, 1.2])

    ax1 = fig.add_subplot(gs[0, 0])
    ax1.imshow(img_np)
    ax1.axis("off")
    ax1.set_title("Original image")

    ax2 = fig.add_subplot(gs[0, 1])
    ax2.imshow(overlay_np)
    ax2.axis("off")
    ax2.set_title("Grad-CAM overlay")

    ax3 = fig.add_subplot(gs[0, 2])
    colors = ["tab:red" if v > 0 else "tab:blue" for v in local_imp_top["delta_prob"]]
    ax3.barh(local_imp_top["feature"], local_imp_top["delta_prob"], color=colors)
    ax3.axvline(0.0, color="black", linewidth=1)
    ax3.set_title("Local metadata importance")
    ax3.set_xlabel("Δ predicted probability\n(remove / neutralize feature)")
    ax3.tick_params(axis="y", labelsize=9)

    true_name = "malignant" if label == 1 else "benign"
    pred_name = "malignant" if pred == 1 else "benign"

    fig.suptitle(
        f"{case_type} | True: {true_name} | Pred: {pred_name} | "
        f"Prob(malignant)={prob:.3f} | subtype={subtype}",
        fontsize=13
    )

    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

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

    pred_df = collect_test_predictions(model, test_loader, device, threshold=THRESHOLD)
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions.csv"), index=False)

    # assign case type for every sample so we can generate figures for the whole test set
    def _case_type(row):
        lab = int(row["label"])
        pr = int(row["pred"])
        if lab == 1 and pr == 1:
            return "TP"
        if lab == 0 and pr == 0:
            return "TN"
        if lab == 0 and pr == 1:
            return "FP"
        if lab == 1 and pr == 0:
            return "FN"
        return "UNK"

    pred_df["case_type"] = pred_df.apply(_case_type, axis=1)
    pred_df.to_csv(os.path.join(OUTPUT_DIR, "test_predictions_with_case_types.csv"), index=False)

    target_layer = find_last_conv_layer(model.image_encoder.feature_extractor)
    gradcam = GradCAM(model, target_layer)

    try:
        total = len(pred_df)
        print(f"Generating figures for {total} test samples (this may take a while)...")
        for i, row in pred_df.iterrows():
            idx = int(row["dataset_idx"])
            case_type = row.get("case_type", "UNK")
            prob = row.get("prob", None)

            out_name = f"{case_type}_idx{idx}" + (f"_p{prob:.3f}" if prob is not None else "") + ".png"
            out_path = os.path.join(OUTPUT_DIR, out_name)

            try:
                # ensure the row passed to plotting contains a case_type key
                row_for_plot = row.copy()
                row_for_plot["case_type"] = case_type

                plot_case_figure(
                    case_row=row_for_plot,
                    dataset=test_ds,
                    model=model,
                    gradcam=gradcam,
                    output_path=out_path,
                    device=device,
                )
                if (i + 1) % 50 == 0 or (i + 1) == total:
                    print(f"Saved {i+1}/{total}: {out_path}")
            except Exception as e:
                # log and continue on individual sample failures
                print(f"Failed to process idx={idx} (row={i}): {e}")
                continue
    finally:
        gradcam.remove()

    print(f"\nAll outputs (attempted) saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()