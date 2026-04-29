from pathlib import Path
import matplotlib.pyplot as plt
from PIL import Image
import re

image_only_dir = Path("image_only_baseline/gradcam_image_only")
multimodal_dir = Path("multimodal_pipeline/results/xai_cases_newtitle/")
output_dir = Path("comparing_gradcams/")

output_dir.mkdir(parents=True, exist_ok=True)

PREFIX_PATTERN = re.compile(r"^(FP|FN|TP|TN)[_\-\s]*", re.IGNORECASE)


def normalize_name(path, remove_gradcam):
    name = path.stem

    # Remove _gradcam suffix
    if remove_gradcam:
        name = re.sub(r"[_\-\s]*gradcam$", "", name, flags=re.IGNORECASE)

    # Remove probability suffix, e.g. _p0.445
    name = re.sub(r"[_\-\s]*p\d+(?:\.\d+)?$", "", name, flags=re.IGNORECASE)

    # Remove FP / FN / TP / TN prefix
    name = PREFIX_PATTERN.sub("", name)

    # Normalize separators
    name = name.strip()
    name = name.replace(" ", "")
    name = name.replace("_", "")
    name = name.replace("-", "")

    return name.lower()


image_only_files = {}
for path in image_only_dir.glob("*.png"):
    key = normalize_name(path, remove_gradcam=True)
    image_only_files[key] = path

multimodal_files = {}
for path in multimodal_dir.glob("*.png"):
    key = normalize_name(path, remove_gradcam=False)
    multimodal_files[key] = path


print("\n--- Example Image-only Keys ---")
for k, v in list(image_only_files.items())[:10]:
    print(k, " <-- ", v.name)

print("\n--- Example Multimodal Keys ---")
for k, v in list(multimodal_files.items())[:10]:
    print(k, " <-- ", v.name)


common_keys = sorted(set(image_only_files) & set(multimodal_files))

print("\nFound pairs:", len(common_keys))

if len(common_keys) == 0:
    print("\nNo pairs found.")
    print("First image-only file:", next(iter(image_only_files.values()), None))
    print("First multimodal file:", next(iter(multimodal_files.values()), None))
    raise SystemExit


def get_label(path):
    match = re.match(r"^(FP|FN|TP|TN)", path.stem, re.IGNORECASE)
    return match.group(1).upper() if match else "UNK"


for key in common_keys:
    img_only_path = image_only_files[key]
    img_multi_path = multimodal_files[key]

    img_only = Image.open(img_only_path).convert("RGB")
    img_multi = Image.open(img_multi_path).convert("RGB")

    label_only = get_label(img_only_path)
    label_multi = get_label(img_multi_path)

    fig, axes = plt.subplots(2, 1, figsize=(14, 12))

    axes[0].imshow(img_only)
    axes[0].set_title(f"Image only ({label_only})")
    axes[0].axis("off")

    axes[1].imshow(img_multi)
    axes[1].set_title(f"Multimodal ({label_multi})")
    axes[1].axis("off")

    fig.suptitle(f"ID: {key}")
    plt.tight_layout()

    save_path = output_dir / f"{label_only}_to_{label_multi}_{key}_comparison.png"
    plt.savefig(save_path, dpi=200, bbox_inches="tight")
    plt.close(fig)

print("Saved to:", output_dir.resolve())