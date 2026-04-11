import argparse
from pathlib import Path
from PIL import Image, ImageFile
from tqdm import tqdm

"""
Build Image Cache with images croped to 224x224 to fasten the MIDAS Dataloader.
Run: python data_processing/image_cache.py --img_root <root to img> --cache_root <root to cache>

"""

ImageFile.LOAD_TRUNCATED_IMAGES = True

VALID_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def main(args):
    img_root = Path(args.img_root)
    cache_root = Path(args.cache_root)
    cache_root.mkdir(parents=True, exist_ok=True)

    files = [
        p for p in img_root.iterdir()
        if p.is_file() and p.suffix.lower() in VALID_EXTS
    ]

    print(f"Found {len(files)} image files.")

    ok, skipped, failed = 0, 0, 0

    for src_path in tqdm(files, desc="Caching images"):
        dst_path = cache_root / src_path.name

        if dst_path.exists():
            skipped += 1
            continue

        try:
            with Image.open(src_path) as img:
                img = img.convert("RGB")
                img = img.resize((args.size, args.size), Image.BILINEAR)
                img.save(dst_path)
            ok += 1
        except Exception as e:
            failed += 1
            print(f"Failed: {src_path.name} | {e}")

    print("\nDone:")
    print(f"Saved:   {ok}")
    print(f"Skipped: {skipped}")
    print(f"Failed:  {failed}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Cache all images from a folder")
    parser.add_argument("--img_root", type=str, required=True, help="Path to original image folder")
    parser.add_argument("--cache_root", type=str, required=True, help="Path to cache folder")
    parser.add_argument("--size", type=int, default=224, help="Target image size")

    args = parser.parse_args()
    main(args)