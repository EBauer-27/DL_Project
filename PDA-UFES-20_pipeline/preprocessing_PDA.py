import os
from typing import Dict, List, Optional, Sequence

import numpy as np
import pandas as pd
from PIL import Image

import zipfile
import shutil
from pathlib import Path

import torch
from torch.utils.data import Dataset

try:
    from torchvision import transforms
except Exception:
    transforms = None

from sklearn.preprocessing import StandardScaler


class _SimpleToTensor:
    def __call__(self, image):
        arr = np.asarray(image, dtype=np.float32) / 255.0
        if arr.ndim == 2:
            arr = arr[:, :, None]
        arr = np.transpose(arr, (2, 0, 1))
        return torch.from_numpy(arr)


def extract_all_image_zips(image_dir: str):
    image_dir = Path(image_dir)
    image_dir.mkdir(parents=True, exist_ok=True)

    zip_files = sorted(image_dir.rglob("*.zip"))

    if not zip_files:
        print(f"No ZIP files found in {image_dir.resolve()}")
        return

    image_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}

    print(f"Found {len(zip_files)} ZIP files in {image_dir.resolve()}")

    for zip_path in zip_files:
        print(f"Extracting {zip_path.name}...")

        with zipfile.ZipFile(zip_path, "r") as zip_ref:
            for member in zip_ref.namelist():
                if member.endswith("/"):
                    continue

                filename = Path(member).name

                if not filename:
                    continue

                if Path(filename).suffix.lower() not in image_extensions:
                    continue

                target_path = image_dir / filename

                # Nicht überschreiben, falls gleiche Dateinamen vorkommen
                if target_path.exists():
                    stem = target_path.stem
                    suffix = target_path.suffix
                    target_path = image_dir / f"{zip_path.stem}_{stem}{suffix}"

                with zip_ref.open(member) as source, open(target_path, "wb") as target:
                    shutil.copyfileobj(source, target)

        print(f"Done: {zip_path.name}")

    print(f"All images extracted flat into: {image_dir.resolve()}")


class PADDataset(Dataset):
    PAD_KEEP_COLS = [
        "patient_id",
        "lesion_id",
        "img_id",
        "age",
        "gender",
        "fitspatrick",
        "region",
        "diameter_1",
        "diameter_2",
        "diagnostic",
    ]

    LABEL_MAP = {
        "ACK": 1,
        "BCC": 1,
        "SCC": 1,
        "MEL": 1,
        "NEV": 0,
        "SEK": 0,
    }

    FITZPATRICK_TO_MIDAS = {
        "1": "i pale white skin, blue/green eyes, blond/red hair",
        "1.0": "i pale white skin, blue/green eyes, blond/red hair",
        "2": "ii fair skin, blue eyes",
        "2.0": "ii fair skin, blue eyes",
        "3": "iii darker white skin",
        "3.0": "iii darker white skin",
        "4": "iv light brown skin",
        "4.0": "iv light brown skin",
        "5": "v brown skin",
        "5.0": "v brown skin",
        "6": "vi dark brown or black skin",
        "6.0": "vi dark brown or black skin",
    }

    REGION_TO_MIDAS = {
        "ABDOMEN": "l abdomen",
        "ARM": "l arm",
        "BACK": "l back",
        "CHEST": "l chest",
        "EAR": "left ear",
        "FACE": "l cheek",
        "FOOT": "l foot",
        "FOREARM": "l forearm",
        "HAND": "l hand",
        "LIP": "l lower lip",
        "NECK": "l neck",
        "NOSE": "l nose",
        "SCALP": "l frontal scalp",
        "THIGH": "l thigh",
    }

    DEFAULT_CONTINUOUS_COLS = [
        "midas_age",
        "length_(mm)",
        "width_(mm)",
        "midas_iscontrol",
    ]

    def __init__(
        self,
        file_path: str,
        image_root: Optional[str] = "PDA-UFES-20/images",
        transform=None,
        use_images: bool = True,
        scaler: Optional[StandardScaler] = None,
        cat_maps: Optional[Dict[str, List[str]]] = None,
        drop_incomplete: bool = True,
        continuous_cols: Optional[Sequence[str]] = None,
        strict_unknown_labels: bool = True,
        verbose: bool = True,
        extract_zips: bool = True,
    ):
        self.image_root = image_root

        if self.image_root is not None and extract_zips:
            extract_all_image_zips(self.image_root)

        self.transform = transform
        self.to_tensor = transforms.ToTensor() if transforms is not None else _SimpleToTensor()
        self.use_images = use_images
        self.verbose = verbose

        if scaler is None:
            raise ValueError("PAD is test-only. Pass the scaler fitted on the MIDAS training set.")

        if cat_maps is None:
            raise ValueError("PAD is test-only. Pass cat_maps fitted on the MIDAS training set.")

        self.scaler = scaler
        self.cat_maps = cat_maps

        self.categorical_cols: List[str] = []
        self.continuous_cols: List[str] = []
        self.categories = ()
        self.num_continuous = 0

        if file_path.endswith(".xlsx"):
            raw_df = pd.read_excel(file_path)
        else:
            raw_df = pd.read_csv(file_path)

        raw_df = raw_df[[c for c in self.PAD_KEEP_COLS if c in raw_df.columns]].copy()
        raw_df = self._create_label_column(raw_df, strict=strict_unknown_labels)

        if drop_incomplete:
            raw_df = self._drop_incomplete_pad_rows(raw_df)

        self.df = self._convert_pad_to_midas_schema(raw_df)

        self.categorical_cols = list(self.cat_maps.keys())
        self.continuous_cols = self._resolve_continuous_cols(continuous_cols)

        self.df = self._preprocess_tabular(self.df)

    def _create_label_column(self, df: pd.DataFrame, strict: bool = True) -> pd.DataFrame:
        df = df.copy()

        if "diagnostic" not in df.columns:
            raise ValueError("Expected column 'diagnostic' in PAD dataset.")

        diagnosis = df["diagnostic"].fillna("").astype(str).str.upper().str.strip()
        unknown_labels = sorted(set(diagnosis.unique()) - set(self.LABEL_MAP.keys()))

        if unknown_labels and strict:
            raise ValueError(f"Unknown diagnostic labels found: {unknown_labels}")

        df["label"] = diagnosis.map(self.LABEL_MAP).astype("float32")
        return df

    def _drop_incomplete_pad_rows(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df = df.replace(r"^\s*$", np.nan, regex=True)
        df = df.replace(
            ["unknown", "UNKNOWN", "UNK", "nan", "NaN", "NAN", "None", "NONE"],
            np.nan,
        )

        required_cols = [
            "age",
            "gender",
            "fitspatrick",
            "region",
            "diameter_1",
            "diameter_2",
            "diagnostic",
            "label",
        ]
        required_cols = [c for c in required_cols if c in df.columns]

        before = len(df)
        df = df.dropna(subset=required_cols).copy().reset_index(drop=True)
        after = len(df)

        if self.verbose:
            print(f"Dropped incomplete PAD rows: {before - after}")
            print(f"Remaining PAD rows: {after}")

        return df

    @staticmethod
    def _to_numeric(series: pd.Series) -> pd.Series:
        return pd.to_numeric(series, errors="coerce")

    def _map_gender(self, value) -> str:
        if pd.isna(value):
            return ""
        value = str(value).strip().lower()
        if value in {"female", "f"}:
            return "female"
        if value in {"male", "m"}:
            return "male"
        return ""

    def _map_fitzpatrick(self, value) -> str:
        if pd.isna(value):
            return ""
        key = str(value).strip()
        return self.FITZPATRICK_TO_MIDAS.get(key, "")

    def _map_region(self, value) -> str:
        if pd.isna(value):
            return ""

        key = str(value).strip().upper()
        mapped = self.REGION_TO_MIDAS.get(key, "")

        valid_locations = set(self.cat_maps.get("midas_location", []))

        if mapped in valid_locations:
            return mapped

        return ""

    def _convert_pad_to_midas_schema(self, df: pd.DataFrame) -> pd.DataFrame:
        pad = df.copy()
        midas = pd.DataFrame(index=pad.index)

        midas["patient_id"] = pad.get("patient_id", pd.Series(index=pad.index, dtype=object))
        midas["lesion_id"] = pad.get("lesion_id", pd.Series(index=pad.index, dtype=object))
        midas["img_id"] = pad.get("img_id", pd.Series(index=pad.index, dtype=object))
        midas["diagnostic"] = pad["diagnostic"]
        midas["label"] = pad["label"].astype("float32")

        midas["midas_gender"] = pad["gender"].apply(self._map_gender)
        midas["midas_fitzpatrick"] = pad["fitspatrick"].apply(self._map_fitzpatrick)
        midas["midas_location"] = pad["region"].apply(self._map_region)

        midas["midas_ethnicity"] = "unknown"
        midas["midas_race"] = "unknown"

        midas["midas_age"] = self._to_numeric(pad["age"])
        midas["length_(mm)"] = self._to_numeric(pad["diameter_1"])
        midas["width_(mm)"] = self._to_numeric(pad["diameter_2"])

        midas["midas_iscontrol"] = "no"
        midas["midas_distance"] = "dscope"

        return midas

    def _resolve_continuous_cols(self, continuous_cols: Optional[Sequence[str]]) -> List[str]:
        if continuous_cols is not None:
            return list(continuous_cols)

        if hasattr(self.scaler, "feature_names_in_"):
            return list(self.scaler.feature_names_in_)

        if hasattr(self.scaler, "n_features_in_"):
            n_features = int(self.scaler.n_features_in_)
            if n_features <= len(self.DEFAULT_CONTINUOUS_COLS):
                return self.DEFAULT_CONTINUOUS_COLS[:n_features]

        return self.DEFAULT_CONTINUOUS_COLS.copy()

    def _preprocess_tabular(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        for col in self.categorical_cols:
            if col not in df.columns:
                df[col] = ""

        for col in self.categorical_cols:
            ser = df[col].fillna("").astype(str).str.strip()
            cats = self.cat_maps[col]
            cat = pd.Categorical(ser, categories=cats)

            unseen_mask = cat.codes == -1
            if self.verbose and unseen_mask.any():
                values = sorted(ser[unseen_mask].unique().tolist())[:10]
                print(
                    f"Warning: {int(unseen_mask.sum())} unseen/unknown values in "
                    f"categorical column '{col}' encoded as 0. Examples: {values}"
                )

            df[col] = (cat.codes + 1).astype(np.int64)

        for col in self.continuous_cols:
            if col not in df.columns:
                df[col] = 0.0
            df[col] = pd.to_numeric(df[col], errors="coerce").fillna(0.0).astype("float32")

        if self.continuous_cols:
            try:
                df[self.continuous_cols] = self.scaler.transform(df[self.continuous_cols])
            except Exception as e:
                raise ValueError(
                    "Could not apply MIDAS scaler to PAD continuous columns. "
                    f"Using columns: {self.continuous_cols}. "
                    "Pass continuous_cols in the exact MIDAS training order if your scaler "
                    "was saved without feature_names_in_."
                ) from e

        self.categories = tuple(int(df[col].max()) + 1 for col in self.categorical_cols)
        self.num_continuous = len(self.continuous_cols)

        if self.verbose:
            print(f"PAD categorical columns: {self.categorical_cols}")
            print(f"PAD continuous columns: {self.continuous_cols}")
            print(f"PAD label balance: {df['label'].value_counts().to_dict()}")

        return df.reset_index(drop=True)

    def _create_local_path(self, row: pd.Series) -> Optional[str]:
        if self.image_root is None:
            return None

        fname = row.get("img_id", "")
        if not fname or pd.isna(fname):
            return None

        return os.path.join(self.image_root, os.path.basename(str(fname)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        output = {}

        if self.use_images:
            img_path = self._create_local_path(row)

            if img_path and os.path.exists(img_path):
                with Image.open(img_path) as img:
                    image = img.convert("RGB")
            else:
                raise FileNotFoundError(f"Image not found: {img_path}")

            if self.transform is not None:
                image = self.transform(image)
            else:
                image = self.to_tensor(image)

            output["image"] = image

        if self.categorical_cols:
            x_categ = torch.tensor(
                row[self.categorical_cols].values.astype(np.int64),
                dtype=torch.long,
            )
        else:
            x_categ = torch.empty(0, dtype=torch.long)

        if self.continuous_cols:
            x_cont = torch.tensor(
                row[self.continuous_cols].values.astype(np.float32),
                dtype=torch.float32,
            )
        else:
            x_cont = torch.empty(0, dtype=torch.float32)

        output["x_categ"] = x_categ
        output["x_cont"] = x_cont
        output["label"] = torch.tensor(row["label"], dtype=torch.float32)

        return output


