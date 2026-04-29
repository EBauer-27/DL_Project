from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image
import re

image_only_dir = Path("image_only_baseline/gradcam_image_only")
multimodal_dir = Path("multimodal_pipeline/results/xai_cases_without_original/")
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


def get_label(path):
    match = re.match(r"^(FP|FN|TP|TN)", path.stem, re.IGNORECASE)
    return match.group(1).upper() if match else "UNK"


def is_correct(label):
    return label in ["TP", "TN"]


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


transition_counts = Counter()
change_counts = Counter()

image_only_correct_count = 0
multimodal_correct_count = 0

improved_examples = []
worsened_examples = []
changed_examples = []


for key in common_keys:
    img_only_path = image_only_files[key]
    img_multi_path = multimodal_files[key]

    label_only = get_label(img_only_path)
    label_multi = get_label(img_multi_path)

    transition = f"{label_only}_to_{label_multi}"
    transition_counts[transition] += 1

    only_correct = is_correct(label_only)
    multi_correct = is_correct(label_multi)

    image_only_correct_count += int(only_correct)
    multimodal_correct_count += int(multi_correct)

    if not only_correct and multi_correct:
        change_type = "Improved"
        improved_examples.append((key, transition))
    elif only_correct and not multi_correct:
        change_type = "Worsened"
        worsened_examples.append((key, transition))
    else:
        change_type = "Unchanged"

    change_counts[change_type] += 1

    if label_only != label_multi:
        changed_examples.append((key, transition))

    # ------------------------------------------------------------
    # Plot 
    # ------------------------------------------------------------
    img_only = Image.open(img_only_path).convert("RGB")
    img_multi = Image.open(img_multi_path).convert("RGB")
    
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


total = len(common_keys)

print("\n--- Transition Counts ---")
for transition, count in sorted(transition_counts.items()):
    percentage = count / total * 100
    print(f"{transition}: {count} ({percentage:.2f}%)")

print("\n--- Change Summary ---")
for change_type in ["Improved", "Worsened", "Unchanged"]:
    count = change_counts[change_type]
    percentage = count / total * 100
    print(f"{change_type}: {count} ({percentage:.2f}%)")

print("\n--- Accuracy on Matched Pairs ---")
image_only_acc = image_only_correct_count / total * 100
multimodal_acc = multimodal_correct_count / total * 100

print(f"Image-only correct: {image_only_correct_count}/{total} ({image_only_acc:.2f}%)")
print(f"Multimodal correct: {multimodal_correct_count}/{total} ({multimodal_acc:.2f}%)")
print(f"Difference: {multimodal_acc - image_only_acc:+.2f} percentage points")

print("\n--- Changed Predictions ---")
print(f"Changed labels: {len(changed_examples)}/{total} ({len(changed_examples) / total * 100:.2f}%)")

print("\n--- Improved Examples ---")
for key, transition in improved_examples[:20]:
    print(f"{key}: {transition}")

print("\n--- Worsened Examples ---")
for key, transition in worsened_examples[:20]:
    print(f"{key}: {transition}")

print("\nAnalysis complete.")