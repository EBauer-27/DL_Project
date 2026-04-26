import os
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedShuffleSplit


def create_manifests(
    master_path: str,
    image_root: str,
    out_dir: str = "manifests_record_split",
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    check_images: bool = False,
):
    """
    Create record-wise stratified train/val/test manifests from a master metadata file.

    Important:
    - Splitting is done at group level.
    - Preferred grouping order:
        1. patient_id
        2. patient
        3. midas_record_id
        4. midas_file_name fallback

    This prevents different images from the same MIDAS record from leaking across
    train/val/test.
    """

    os.makedirs(out_dir, exist_ok=True)

    # ------------------------------------------------------------
    # Load master table
    # ------------------------------------------------------------
    if master_path.endswith(".xlsx"):
        df = pd.read_excel(master_path)
    else:
        df = pd.read_csv(master_path)

    df = df.copy()

    # ------------------------------------------------------------
    # Create binary label from midas_path
    # ------------------------------------------------------------
    danger_words = [
        "melanoma",
        "bcc",
        "scc",
        "carcinoma",
        "malignant",
        "mct",
        "ak",
    ]

    if "midas_path" in df.columns:
        midas = df["midas_path"].fillna("").astype(str).str.lower()
        df["label"] = midas.apply(
            lambda x: 1 if any(w in x for w in danger_words) else 0
        )
    else:
        df["label"] = 0

    # ------------------------------------------------------------
    # Split labeled vs held-out
    # ------------------------------------------------------------
    labeled_mask = (
        df["midas_path"].notna()
        & (df["midas_path"].astype(str).str.strip() != "")
    )

    labeled = df[labeled_mask].copy()
    heldout = df[~labeled_mask].copy()

    # ------------------------------------------------------------
    # Normalize image filenames
    # ------------------------------------------------------------
    if "midas_file_name" in labeled.columns:
        labeled["midas_file_name"] = labeled["midas_file_name"].astype(str).apply(
            lambda x: os.path.basename(x) if x and x != "nan" else ""
        )

    # ------------------------------------------------------------
    # Optional image existence check
    # ------------------------------------------------------------
    if check_images and "midas_file_name" in labeled.columns:
        missing = []

        for i, fname in enumerate(labeled["midas_file_name"].astype(str)):
            path = os.path.join(image_root, os.path.basename(fname))

            if not os.path.exists(path):
                missing.append((i, fname))

        if missing:
            print(
                f"Warning: {len(missing)} labeled rows point to missing image files "
                f"(first 10): {missing[:10]}"
            )

    # ------------------------------------------------------------
    # Held-out clinician set
    # ------------------------------------------------------------
    if "clinical_impression_1" in heldout.columns:
        clin_col = heldout["clinical_impression_1"]
    elif "clinician_impression_1" in heldout.columns:
        clin_col = heldout["clinician_impression_1"]
    else:
        clin_col = pd.Series([""] * len(heldout), index=heldout.index)

    clin_mask = clin_col.notna() & (clin_col.astype(str).str.strip() != "")
    heldout_clinician = heldout[clin_mask].copy()

    # ------------------------------------------------------------
    # Determine grouping column
    # ------------------------------------------------------------
    if "patient_id" in labeled.columns:
        orig_group_col = "patient_id"
    elif "patient" in labeled.columns:
        orig_group_col = "patient"
    elif "midas_record_id" in labeled.columns:
        orig_group_col = "midas_record_id"
    elif "midas_file_name" in labeled.columns:
        orig_group_col = "midas_file_name"
    else:
        orig_group_col = None

    print(f"Using grouping column: {orig_group_col}")

    # ------------------------------------------------------------
    # Create stable group_id
    # ------------------------------------------------------------
    if orig_group_col is not None:
        labeled["group_id"] = labeled[orig_group_col].fillna("")
    else:
        labeled["group_id"] = ""

    def fallback_gid(row):
        gid = str(row["group_id"]).strip()

        if gid and gid.lower() != "nan":
            return gid

        if "midas_file_name" in row and str(row["midas_file_name"]).strip():
            return str(row["midas_file_name"]).strip()

        return f"row_{row.name}"

    labeled["group_id"] = labeled.apply(fallback_gid, axis=1)

    group_col = "group_id"

    # ------------------------------------------------------------
    # Build group-level table
    # ------------------------------------------------------------
    grp = (
        labeled.groupby(group_col)
        .agg(
            label_majority=(
                "label",
                lambda x: int(x.mode().iloc[0])
                if not x.mode().empty
                else int(x.iloc[0]),
            ),
            n_images=("label", "size"),
        )
        .reset_index()
    )

    groups = grp[group_col].values
    group_labels = grp["label_majority"].values

    # ------------------------------------------------------------
    # First split: train vs temporary val/test
    # ------------------------------------------------------------
    if not np.isclose(train_frac + val_frac + test_frac, 1.0):
        raise ValueError("train_frac + val_frac + test_frac must equal 1.0")

    sss = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - train_frac,
        random_state=seed,
    )

    train_idx, temp_idx = next(sss.split(groups, group_labels))

    train_groups = set(groups[train_idx])
    temp_groups = groups[temp_idx]
    temp_labels = group_labels[temp_idx]

    # ------------------------------------------------------------
    # Second split: validation vs test
    # ------------------------------------------------------------
    rel_val_frac = val_frac / (val_frac + test_frac)

    sss2 = StratifiedShuffleSplit(
        n_splits=1,
        test_size=1 - rel_val_frac,
        random_state=seed + 1,
    )

    val_idx_rel, test_idx_rel = next(sss2.split(temp_groups, temp_labels))

    val_groups = set(temp_groups[val_idx_rel])
    test_groups = set(temp_groups[test_idx_rel])

    # ------------------------------------------------------------
    # Map groups back to rows
    # ------------------------------------------------------------
    train_df = labeled[labeled[group_col].isin(train_groups)].copy()
    val_df = labeled[labeled[group_col].isin(val_groups)].copy()
    test_df = labeled[labeled[group_col].isin(test_groups)].copy()

    # ------------------------------------------------------------
    # Save manifests
    # ------------------------------------------------------------
    train_path = os.path.join(out_dir, "train.csv")
    val_path = os.path.join(out_dir, "val.csv")
    test_path = os.path.join(out_dir, "test.csv")
    heldout_clinician_path = os.path.join(out_dir, "heldout_clinician.csv")
    heldout_unlabeled_path = os.path.join(out_dir, "heldout_unlabeled.csv")

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    heldout_clinician.to_csv(heldout_clinician_path, index=False)
    heldout.to_csv(heldout_unlabeled_path, index=False)

    # ------------------------------------------------------------
    # Diagnostics
    # ------------------------------------------------------------
    assigned = pd.concat([train_df, val_df, test_df])

    print("\n================ Manifest diagnostics ================")
    print(f"Total rows in master: {len(df)}")
    print(f"Labeled rows: {len(labeled)}")
    print(f"Held-out rows: {len(heldout)}")
    print(f"Assigned labeled rows: {len(assigned)}")
    print(f"Unique groups: {labeled[group_col].nunique()}")

    print("\nTrain/Val/Test sizes:")
    print(f"Train images: {len(train_df)} | groups: {train_df[group_col].nunique()}")
    print(f"Val images:   {len(val_df)} | groups: {val_df[group_col].nunique()}")
    print(f"Test images:  {len(test_df)} | groups: {test_df[group_col].nunique()}")

    print("\nLabel balance:")
    print("Train:", train_df["label"].value_counts().to_dict())
    print("Val:  ", val_df["label"].value_counts().to_dict())
    print("Test: ", test_df["label"].value_counts().to_dict())

    # Check group leakage
    train_group_set = set(train_df[group_col].astype(str))
    val_group_set = set(val_df[group_col].astype(str))
    test_group_set = set(test_df[group_col].astype(str))

    print("\nGroup overlap checks:")
    print("Train-Val group overlap:", len(train_group_set & val_group_set))
    print("Train-Test group overlap:", len(train_group_set & test_group_set))
    print("Val-Test group overlap:", len(val_group_set & test_group_set))

    # Check midas_record_id leakage specifically
    if "midas_record_id" in labeled.columns:
        train_record_set = set(train_df["midas_record_id"].dropna().astype(str))
        val_record_set = set(val_df["midas_record_id"].dropna().astype(str))
        test_record_set = set(test_df["midas_record_id"].dropna().astype(str))

        print("\nMIDAS record overlap checks:")
        print("Train-Val midas_record_id overlap:", len(train_record_set & val_record_set))
        print("Train-Test midas_record_id overlap:", len(train_record_set & test_record_set))
        print("Val-Test midas_record_id overlap:", len(val_record_set & test_record_set))

    # Check mixed-label groups
    mixed = labeled.groupby(group_col)["label"].nunique()
    mixed = mixed[mixed > 1]

    if len(mixed):
        print(f"\nWarning: {len(mixed)} groups contain mixed labels.")
        print("First mixed groups:")
        print(mixed.head())
    else:
        print("\nNo mixed-label groups found.")

    print("\nManifests created in:", out_dir)
    print("======================================================\n")

    return {
        "train": {
            "images": len(train_df),
            "groups": train_df[group_col].nunique(),
            "label_balance": train_df["label"].value_counts().to_dict(),
        },
        "val": {
            "images": len(val_df),
            "groups": val_df[group_col].nunique(),
            "label_balance": val_df["label"].value_counts().to_dict(),
        },
        "test": {
            "images": len(test_df),
            "groups": test_df[group_col].nunique(),
            "label_balance": test_df["label"].value_counts().to_dict(),
        },
        "heldout_clinician": {
            "images": len(heldout_clinician),
        },
        "heldout_unlabeled": {
            "images": len(heldout),
        },
        "paths": {
            "train": train_path,
            "val": val_path,
            "test": test_path,
            "heldout_clinician": heldout_clinician_path,
            "heldout_unlabeled": heldout_unlabeled_path,
        },
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Create record-wise stratified train/val/test manifests."
    )

    parser.add_argument("master_path")
    parser.add_argument("image_root")
    parser.add_argument("--out_dir", default="manifests_record_split")
    parser.add_argument("--check_images", action="store_true")
    parser.add_argument("--seed", type=int, default=42)

    args = parser.parse_args()

    create_manifests(
        master_path=args.master_path,
        image_root=args.image_root,
        out_dir=args.out_dir,
        seed=args.seed,
        check_images=args.check_images,
    )