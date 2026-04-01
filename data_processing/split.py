import os
from typing import Optional

import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit


def create_manifests(
    master_path: str,
    image_root: str,
    out_dir: str = "manifests",
    seed: int = 42,
    train_frac: float = 0.8,
    val_frac: float = 0.1,
    test_frac: float = 0.1,
    check_images: bool = False,
):
    """Create patient-wise stratified train/val/test manifests from a master metadata file.

    Behavior:
    - Rows with a non-empty `midas_path` are considered labeled and used for train/val/test splits.
    - Rows with empty/missing `midas_path` are held out; a subset with `clinician_impression_1` present
      is saved as `heldout_clinician.csv` for later evaluation.
    - Splitting is done at the group level (prefer `patient_id` if present, otherwise `midas_file_name`).
    - Stratification is performed using the group's majority label.

    Outputs CSVs into `out_dir`: train.csv, val.csv, test.csv, heldout_clinician.csv, heldout_unlabeled.csv

    Example usage: python data_processing/split.py \
    data/MRA-MIDAS/midasmultimodalimagedatasetforaibasedskincancer/release_midas.xlsx \
    data/MRA-MIDAS/midasmultimodalimagedatasetforaibasedskincancer/ \
    --out_dir manifests
    """
    os.makedirs(out_dir, exist_ok=True)

    # Load master table
    if master_path.endswith('.xlsx'):
        df = pd.read_excel(master_path)
    else:
        df = pd.read_csv(master_path)

    # Normalize column names
    df = df.copy()

    # Create binary label from midas_path (same rules as MIDASDataset)
    danger_words = ['melanoma', 'bcc', 'scc', 'carcinoma', 'malignant', 'mct', 'ak']
    if 'midas_path' in df.columns:
        midas = df['midas_path'].fillna('').astype(str).str.lower()
        df['label'] = midas.apply(lambda x: 1 if any(w in x for w in danger_words) else 0)
    else:
        df['label'] = 0

    # Filter labeled vs held-out
    labeled_mask = df['midas_path'].notna() & (df['midas_path'].astype(str).str.strip() != '')
    labeled = df[labeled_mask].copy()
    heldout = df[~labeled_mask].copy()

    # Optionally check that image files exist and warn (doesn't remove by default)
    if check_images and 'midas_file_name' in labeled.columns:
        missing = []
        for i, fname in enumerate(labeled['midas_file_name'].astype(str)):
            path = os.path.join(image_root, os.path.basename(fname))
            if not os.path.exists(path):
                missing.append((i, fname))
        if missing:
            print(f"Warning: {len(missing)} labeled rows point to missing image files (first 5): {missing[:5]}")

    # Held-out clinician set
    # Build a clinician impression mask aligned with heldout.index to avoid reindexing errors
    if 'clinician_impression_1' in heldout.columns:
        clin_col = heldout['clinician_impression_1']
    else:
        clin_col = pd.Series([''] * len(heldout), index=heldout.index)

    clin_mask = clin_col.notna() & (clin_col.astype(str).str.strip() != '')
    heldout_clinician = heldout[clin_mask].copy()

    # Determine group column and create a stable group id.
    # If patient id exists, use it; otherwise fall back to filename. Fill missing group ids with filename
    if 'patient_id' in labeled.columns:
        orig_group_col = 'patient_id'
    elif 'patient' in labeled.columns:
        orig_group_col = 'patient'
    else:
        orig_group_col = 'midas_file_name'

    labeled = labeled.copy()
    # If midas_file_name exists use its basename when filling
    if 'midas_file_name' in labeled.columns:
        labeled['midas_file_name'] = labeled['midas_file_name'].astype(str).apply(lambda x: os.path.basename(x) if x and x != 'nan' else '')

    labeled['group_id'] = labeled[orig_group_col].fillna('')
    # For any empty group_id (NaN or empty) fall back to the filename or the row index if filename missing
    def fallback_gid(row):
        if row['group_id'] is None or str(row['group_id']).strip() == '':
            if 'midas_file_name' in row and str(row['midas_file_name']).strip() != '':
                return str(row['midas_file_name'])
            return f"row_{row.name}"
        return str(row['group_id'])

    labeled['group_id'] = labeled.apply(fallback_gid, axis=1)

    # use the stable group_id for splitting and stats
    group_col = 'group_id'

    # Build group-level dataframe with majority label per group
    grp = labeled.groupby(group_col).agg(label_majority=('label', lambda x: int(x.mode().iloc[0]) if not x.mode().empty else int(x.iloc[0])),
                                         indices=('label', lambda x: list(x.index)))
    grp = grp.reset_index()

    groups = grp[group_col].values
    group_labels = grp['label_majority'].values

    # First split: train vs temp (train_frac)
    sss = StratifiedShuffleSplit(n_splits=1, test_size=1 - train_frac, random_state=seed)
    train_idx, temp_idx = next(sss.split(groups, group_labels))
    train_groups = set(groups[train_idx])

    temp_groups = groups[temp_idx]
    temp_labels = group_labels[temp_idx]

    # Second split: val vs test (split temp equally according to proportions)
    if val_frac + test_frac <= 0:
        raise ValueError('val_frac + test_frac must be > 0')
    rel_val_frac = val_frac / (val_frac + test_frac)
    sss2 = StratifiedShuffleSplit(n_splits=1, test_size=1 - rel_val_frac, random_state=seed + 1)
    val_idx_rel, test_idx_rel = next(sss2.split(temp_groups, temp_labels))
    val_groups = set(temp_groups[val_idx_rel])
    test_groups = set(temp_groups[test_idx_rel])

    # Map back to rows
    train_df = labeled[labeled[group_col].isin(train_groups)].copy()
    val_df = labeled[labeled[group_col].isin(val_groups)].copy()
    test_df = labeled[labeled[group_col].isin(test_groups)].copy()

    # Diagnostics: report counts and check for any labeled rows that were not assigned
    total_rows = len(df)
    labeled_rows = len(labeled)
    heldout_rows = len(heldout)
    print(f"Total rows in master: {total_rows}")
    print(f"Labeled rows (non-empty midas_path): {labeled_rows}")
    print(f"Held-out rows (no midas_path): {heldout_rows}")

    # Map back to rows (we'll create these first, then check)
    train_path = os.path.join(out_dir, 'train.csv')
    val_path = os.path.join(out_dir, 'val.csv')
    test_path = os.path.join(out_dir, 'test.csv')
    heldout_clinician_path = os.path.join(out_dir, 'heldout_clinician.csv')
    heldout_unlabeled_path = os.path.join(out_dir, 'heldout_unlabeled.csv')

    train_df.to_csv(train_path, index=False)
    val_df.to_csv(val_path, index=False)
    test_df.to_csv(test_path, index=False)
    heldout_clinician.to_csv(heldout_clinician_path, index=False)
    heldout.to_csv(heldout_unlabeled_path, index=False)

    # Post-save diagnostics
    assigned = pd.concat([train_df, val_df, test_df])
    print(f"Assigned labeled rows (train+val+test): {len(assigned)}")
    if len(assigned) != labeled_rows:
        missing = labeled[~labeled.index.isin(assigned.index)]
        print(f"Warning: {len(missing)} labeled rows were not assigned to any split. Example rows:")
        print(missing.head()[['midas_file_name','midas_path','label']])

    # Group diagnostics
    print(f"Unique groups (group_id): {labeled['group_id'].nunique()}")
    # detect any groups with mixed labels
    mixed = labeled.groupby('group_id')['label'].nunique()
    mixed = mixed[mixed > 1]
    if len(mixed):
        print(f"Warning: {len(mixed)} groups contain mixed labels. Example groups:")
        print(mixed.head())

    # Quick stats
    def stats(name, df_):
        return {
            'images': len(df_),
            'patients': df_.get(group_col, pd.Series()).nunique() if group_col in df_.columns else None,
            'label_balance': df_['label'].value_counts().to_dict()
        }

    results = {
        'train': stats('train', train_df),
        'val': stats('val', val_df),
        'test': stats('test', test_df),
        'heldout_clinician': stats('heldout_clinician', heldout_clinician),
        'heldout_unlabeled': stats('heldout_unlabeled', heldout),
        'paths': {
            'train': train_path,
            'val': val_path,
            'test': test_path,
            'heldout_clinician': heldout_clinician_path,
            'heldout_unlabeled': heldout_unlabeled_path,
        }
    }

    print('Manifests created in', out_dir)
    print('Train/Val/Test sizes (images):', len(train_df), len(val_df), len(test_df))

    return results


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Create train/val/test manifests (patient-wise stratified)')
    parser.add_argument('master_path')
    parser.add_argument('image_root')
    parser.add_argument('--out_dir', default='manifests')
    parser.add_argument('--check_images', action='store_true')
    args = parser.parse_args()

    create_manifests(args.master_path, args.image_root, out_dir=args.out_dir, check_images=args.check_images)
