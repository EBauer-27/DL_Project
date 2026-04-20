
import pandas as pd
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms
import cv2
from Models import ResNet18, GoogLeNet, VggNet


MODEL_PATH = "Image_Models/checkpoints_opt/trial_10.pt"
MODEL_NAME = "resnet"
IMAGE_DIR = "MRA-MIDAS/cache_images/" 
LABEL_CSV_PATH = "manifests/train.csv"
OUTPUT_DIR = "Image_Models/gradcam_image_only"

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

def load_model(model_path, model_name, device):
    if model_name == 'resnet':
        model = ResNet18()
    elif model_name == 'googlenet':
        model = GoogLeNet()
    elif model_name == 'vggnet':
        model = VggNet()
    else:
        raise ValueError(f"Unknown model: {model_name}")
    
    state_dict = torch.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    model = model.to(device)
    model.eval()
    return model


def get_target_layer(model, model_name):
    """Get the target convolutional layer for Grad-CAM"""
    if model_name == 'resnet':
        return model.resnet.layer4[-1].conv2
    elif model_name == 'googlenet':
        return model.googleNet.inception5b.branch2[-1]
    elif model_name == 'vggnet':
        return model.vggNet.features[-1]
    else:
        raise ValueError(f"Unknown model: {model_name}")


def get_image_files(image_dir):
    image_dir = Path(image_dir)
    return sorted([p for p in image_dir.iterdir() if p.is_file() and p.suffix.lower() == ".jpg"])


def load_label_mapping(csv_path):
    """Load labels from training CSV"""
    df = pd.read_csv(csv_path)
    label_map = {0: "benign", 1: "malignant"}
    df["label_str"] = df["label"].map(label_map)
    df["midas_file_name"] = df["midas_file_name"].str.strip()
    return df, dict(zip(df["midas_file_name"], df["label_str"]))


def create_visualization(original_img, gradcam_heatmap, ground_truth, pred_label, confidence, save_path):
    """
    Create comparison figure: Original | Overlay
    """
    
    # colored heatmap
    heatmap_colored = cv2.applyColorMap((gradcam_heatmap * 255).astype(np.uint8), cv2.COLORMAP_JET)
    heatmap_colored = cv2.cvtColor(heatmap_colored, cv2.COLOR_BGR2RGB)
    
    # Overlay: 60% original + 40% heatmap
    overlay = (0.6 * original_img + 0.4 * heatmap_colored).astype(np.uint8)
    
    # create figure
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    axes[0].imshow(original_img)
    axes[0].set_title("Original Image", fontsize=12)
    axes[0].axis("off")
    
    axes[1].imshow(overlay)
    axes[1].set_title("Grad-CAM Overlay", fontsize=12)
    axes[1].axis("off")
    
    # title
    title = f"GT: {ground_truth} | Pred: {pred_label} | Confidence: {confidence:.4f}"
    fig.suptitle(title, fontsize=13, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close(fig)


def process_image(model, gradcam_handler, image_path, ground_truth_label, device):
    """Process single image with Grad-CAM"""
    class_names = ["benign", "malignant"]
    
    Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
    
    # load and preprocess image
    pil_img = Image.open(image_path).convert("RGB")
    pil_resized = pil_img.resize((224, 224))
    original_array = np.array(pil_resized)
    
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225]),
    ])
    
    input_tensor = transform(pil_img).unsqueeze(0).to(device)
    input_tensor.requires_grad_(True)
    
    # prediction
    with torch.no_grad():
        logits = model(input_tensor)
    
    # Handle single output (sigmoid) or two outputs (softmax)
    if logits.shape[1] == 1:
        # Binary classification with single output (sigmoid)
        prob_malignant = torch.sigmoid(logits[0, 0]).item()
        pred_idx = 1 if prob_malignant > 0.5 else 0
        confidence = max(prob_malignant, 1 - prob_malignant)
    else:
        # Two outputs (softmax)
        probs = torch.softmax(logits, dim=1)
        pred_idx = logits.argmax(dim=1).item()
        confidence = probs[0, pred_idx].item()
    
    pred_label = class_names[pred_idx]
    
    # Grad-CAM for prediction class
    gradcam_heatmap = gradcam_handler.generate(input_tensor, pred_idx)
    
    if gradcam_heatmap is None:
        return None
    
    # Upsample to input size
    gradcam_heatmap = cv2.resize(gradcam_heatmap, (224, 224))
    
    # save
    output_filename = f"{image_path.stem}_gradcam.png"
    output_path = Path(OUTPUT_DIR) / output_filename
    
    create_visualization(original_array, gradcam_heatmap, ground_truth_label, pred_label, confidence, str(output_path))
    
    return output_path, pred_label, confidence


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    print(f"Loading model from {MODEL_PATH}...")
    model = load_model(MODEL_PATH, MODEL_NAME, device)
    
    print(f"Setting up Grad-CAM...")
    target_layer = get_target_layer(model, MODEL_NAME)
    gradcam = GradCAM(model, target_layer)
    
    # Load training data and labels
    df_train, label_mapping = load_label_mapping(LABEL_CSV_PATH)
    training_images = set(df_train["midas_file_name"].str.strip().values)
    
    print(f"Found {len(training_images)} images in training set")
    
    # Get all images from disk
    all_image_files = get_image_files(IMAGE_DIR)
    
    # Filter to only training images
    training_image_paths = [p for p in all_image_files if p.name in training_images]
    
    print(f"Found {len(training_image_paths)} training images in {IMAGE_DIR}")
    
    success_count = 0
    for i, image_path in enumerate(training_image_paths):
        image_filename = image_path.name
        ground_truth_label = label_mapping.get(image_filename, "unknown")
        
        try:
            result = process_image(model, gradcam, image_path, ground_truth_label, device)
            if result:
                success_count += 1
                output_path, pred_label, confidence = result
                status = "✓" if ground_truth_label == pred_label else "✗"
                print(f"[{status} {success_count:2d}] {image_path.name}: GT={ground_truth_label:8s} Pred={pred_label:9s} Conf={confidence:.4f}")
        except Exception as e:
            print(f"[ERROR] {image_path.name}: {str(e)[:60]}")
    
    print(f"\n✓ Generated {success_count} Grad-CAM visualizations")
    print(f"Saved to: {OUTPUT_DIR}/")


if __name__ == "__main__":
    main()
