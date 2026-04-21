import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler
from typing import Optional, Dict, List
from torchvision import transforms


class MIDASDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        image_root: str,
        transform=None,
        is_training: bool = True,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        cat_maps: Optional[Dict[str, List[str]]] = None,
    ):
        """Dataset for images + tabular metadata.

        Args:
            file_path: path to CSV/XLSX manifest.
            image_root: directory that contains image files.
            transform: torchvision transforms applied to images.
            is_training: whether this is a training dataset (informational).
            scaler: optional pre-fit StandardScaler for numeric columns.
            fit_scaler: if True, fit a new scaler on this dataset (useful for train only).
            cat_maps: optional mapping of categorical columns to category lists to ensure consistent codes.
        """
        self.image_root = image_root
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.is_training = is_training

        # Preprocessing artifacts
        self.scaler = scaler
        self.cat_maps = cat_maps or {}
        self.feature_columns: Optional[List[str]] = None

        # 1. Load data
        if file_path.endswith('.xlsx'):
            self.df = pd.read_excel(file_path)
        else:
            self.df = pd.read_csv(file_path)

        # 2. Create label column first
        self.df = self._create_label_column(self.df)

        # 3. Preprocess tabular (optionally fit scaler)
        self.df = self._preprocess_tabular(self.df, fit_scaler=fit_scaler)

    def _create_label_column(self, df: pd.DataFrame) -> pd.DataFrame:
        """Create binary label from `midas_path` using keyword matching."""
        danger_words = ['melanoma', 'bcc', 'scc', 'carcinoma', 'malignant', 'mct', 'ak']
        midas = df.get('midas_path')
        if midas is None:
            df['label'] = 0
            return df
        label_df = midas.fillna('').astype(str).str.lower()
        df['label'] = label_df.apply(lambda x: 1 if any(w in x for w in danger_words) else 0)
        return df

    def _preprocess_tabular(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        """Handle missing values, encode categoricals, and scale numericals.

        Returns the processed DataFrame and updates self.scaler, self.cat_maps, and self.feature_columns.
        """
        df = df.copy()

        # Identify columns
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude identifiers and label from numeric scaling
        exclude_cols = {'midas_path', 'midas_file_name', 'label'}
        numerical_cols = [c for c in numerical_cols if c not in exclude_cols]

        # Impute
        if object_cols:
            df[object_cols] = df[object_cols].fillna('')
        if numerical_cols:
            df[numerical_cols] = df[numerical_cols].fillna(0)

        # Encode object/categorical columns
        for col in object_cols:
            if col in exclude_cols:
                continue
            ser = df[col].astype(str).str.strip()
            lower = ser.str.lower()
            # boolean-like mapping
            if lower.isin(['yes', 'no', 'y', 'n', 'true', 'false', '0', '1']).all():
                df[col] = lower.replace({'yes': 1, 'y': 1, 'true': 1, '1': 1, 'no': 0, 'n': 0, 'false': 0, '0': 0}).astype(np.float32)
            else:
                # use provided mapping if available
                if col in self.cat_maps and isinstance(self.cat_maps[col], (list, tuple)):
                    cats = list(self.cat_maps[col])
                    cat = pd.Categorical(ser, categories=cats)
                    df[col] = cat.codes.astype(np.int32)
                else:
                    cat = pd.Categorical(ser)
                    df[col] = cat.codes.astype(np.int32)
                    # save mapping for future datasets
                    self.cat_maps[col] = list(cat.categories)

        # Scale numerical columns
        if numerical_cols:
            if fit_scaler:
                scaler = StandardScaler()
                df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
                self.scaler = scaler
            elif self.scaler is not None:
                df[numerical_cols] = self.scaler.transform(df[numerical_cols])
            else:
                # fallback: fit local scaler (not recommended for validation/test)
                scaler = StandardScaler()
                df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
                self.scaler = scaler

        # Set feature columns deterministically (original column order minus excluded ones)
        self.feature_columns = [c for c in df.columns if c not in ['midas_path', 'midas_file_name', 'label']]

        return df

    def _create_local_path(self, row: pd.Series) -> Optional[str]:
        fname = row.get('midas_file_name', '')
        if not fname or pd.isna(fname):
            return None
        return os.path.join(self.image_root, os.path.basename(str(fname)))

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        # Load image
        row = self.df.iloc[idx]
        img_path = self._create_local_path(row)

        if img_path and os.path.exists(img_path):
            with Image.open(img_path) as img:
                image = img.convert('RGB')
        else:
            image = Image.new('RGB', (224, 224), color='black')

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)

        # Tabular data
        if self.feature_columns is None:
            cols = [c for c in ['midas_path', 'midas_file_name', 'label'] if c in self.df.columns]
            tab_vals = self.df.drop(columns=cols).iloc[idx].values
        else:
            tab_vals = self.df[self.feature_columns].iloc[idx].values
        tabular_data = torch.tensor(tab_vals.astype(np.float32))

        # Label
        label = torch.tensor(self.df['label'].iloc[idx]).float()

        return {'image': image, 'tabular': tabular_data, 'label': label}

class MIDASTabularDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        is_training: bool = True,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        cat_maps: Optional[Dict[str, List[str]]] = None,
    ):
        self.is_training = is_training
        self.scaler = scaler
        self.cat_maps = cat_maps or {}

        self.categorical_cols = []
        self.continuous_cols = []
        self.categories = ()
        self.num_continuous = 0

        if file_path.endswith(".xlsx"):
            self.df = pd.read_excel(file_path)
        else:
            self.df = pd.read_csv(file_path)

        self.df = self._create_label_column(self.df)
        self.df = self._preprocess_tabular(self.df, fit_scaler=fit_scaler)

    def _create_label_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        danger_words = ['melanoma', 'bcc', 'scc', 'carcinoma', 'malignant', 'mct', 'ak']
        midas = df.get('midas_path')

        if midas is None:
            df["label"] = 0
            return df

        label_df = midas.fillna("").astype(str).str.lower()
        df["label"] = label_df.apply(
            lambda x: 1 if any(word in x for word in danger_words) else 0
        )
        return df

    def _preprocess_tabular(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        df = df.copy()

        exclude_cols = {
                "midas_path",
                "midas_file_name",
                "label",
                "Unnamed: 0",
                "midas_record_id",
                "group_id",
                "midas_pathreport",
                "clinical_impression_1",
                "clinical_impression_2",
                "clinical_impression_3",
                "midas_melanoma",
            }

        object_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in exclude_cols]
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

        if object_cols:
            df[object_cols] = df[object_cols].fillna("")
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].fillna(0)

        categorical_cols = []
        continuous_cols = numeric_cols.copy()

        for col in object_cols:
            ser = df[col].astype(str).str.strip()
            lower = ser.str.lower()

            is_binary_like = lower.isin(
                ["yes", "no", "y", "n", "true", "false", "0", "1"]
            ).all()

            if is_binary_like:
                df[col] = lower.replace({
                    "yes": 1, "y": 1, "true": 1, "1": 1,
                    "no": 0, "n": 0, "false": 0, "0": 0
                }).astype(np.float32)
                continuous_cols.append(col)
            else:
                if col in self.cat_maps:
                    cats = self.cat_maps[col]
                    cat = pd.Categorical(ser, categories=cats)
                else:
                    cat = pd.Categorical(ser)
                    self.cat_maps[col] = list(cat.categories)

                # shift by +1 so 0 is reserved for unknown/unseen
                df[col] = (cat.codes + 1).astype(np.int64)
                categorical_cols.append(col)

        if continuous_cols:
            if fit_scaler:
                self.scaler = StandardScaler()
                df[continuous_cols] = self.scaler.fit_transform(df[continuous_cols])
            elif self.scaler is not None:
                df[continuous_cols] = self.scaler.transform(df[continuous_cols])
            else:
                self.scaler = StandardScaler()
                df[continuous_cols] = self.scaler.fit_transform(df[continuous_cols])

        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.categories = tuple(int(df[col].max()) + 1 for col in self.categorical_cols)
        self.num_continuous = len(self.continuous_cols)

        return df

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        if self.categorical_cols:
            x_categ = torch.tensor(
                row[self.categorical_cols].values.astype(np.int64),
                dtype=torch.long
            )
        else:
            x_categ = torch.empty(0, dtype=torch.long)

        if self.continuous_cols:
            x_cont = torch.tensor(
                row[self.continuous_cols].values.astype(np.float32),
                dtype=torch.float32
            )
        else:
            x_cont = torch.empty(0, dtype=torch.float32)

        label = torch.tensor(row["label"], dtype=torch.float32)

        return x_categ, x_cont, label
    
class MIDASMultimodalDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        image_root: str,
        transform=None,
        is_training: bool = True,
        scaler: Optional[StandardScaler] = None,
        fit_scaler: bool = False,
        cat_maps: Optional[Dict[str, List[str]]] = None,
    ):
        self.image_root = image_root
        self.transform = transform
        self.to_tensor = transforms.ToTensor()
        self.is_training = is_training
        self.scaler = scaler
        self.cat_maps = cat_maps or {}

        self.categorical_cols = []
        self.continuous_cols = []
        self.categories = ()
        self.num_continuous = 0

        if file_path.endswith(".xlsx"):
            self.df = pd.read_excel(file_path)
        else:
            self.df = pd.read_csv(file_path)

        self.df = self._create_label_column(self.df)
        self.df = self._preprocess_tabular(self.df, fit_scaler=fit_scaler)

    def _create_label_column(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        danger_words = ["melanoma", "bcc", "scc", "carcinoma", "malignant", "mct", "ak"]
        midas = df.get("midas_path")

        if midas is None:
            df["label"] = 0
            return df

        label_df = midas.fillna("").astype(str).str.lower()
        df["label"] = label_df.apply(
            lambda x: 1 if any(word in x for word in danger_words) else 0
        )
        return df

    def _preprocess_tabular(self, df: pd.DataFrame, fit_scaler: bool = False) -> pd.DataFrame:
        df = df.copy()

        exclude_cols = {
            "midas_path",
            "midas_file_name",
            "label",
            "Unnamed: 0",
            "midas_record_id",
            "group_id",
            "midas_pathreport",
            "clinical_impression_1",
            "clinical_impression_2",
            "clinical_impression_3",
            "midas_melanoma",
        }

        object_cols = [c for c in df.select_dtypes(include=["object"]).columns if c not in exclude_cols]
        numeric_cols = [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude_cols]

        if object_cols:
            df[object_cols] = df[object_cols].fillna("")
        if numeric_cols:
            df[numeric_cols] = df[numeric_cols].fillna(0)

        categorical_cols = []
        continuous_cols = numeric_cols.copy()

        for col in object_cols:
            ser = df[col].astype(str).str.strip()
            lower = ser.str.lower()

            is_binary_like = lower.isin(
                ["yes", "no", "y", "n", "true", "false", "0", "1"]
            ).all()

            if is_binary_like:
                mapped = lower.replace({
                    "yes": 1, "y": 1, "true": 1, "1": 1,
                    "no": 0, "n": 0, "false": 0, "0": 0
                })
                df[col] = mapped.infer_objects(copy=False).astype(np.float32)
                continuous_cols.append(col)
            else:
                if col in self.cat_maps:
                    cats = self.cat_maps[col]
                    cat = pd.Categorical(ser, categories=cats)
                else:
                    cat = pd.Categorical(ser)
                    self.cat_maps[col] = list(cat.categories)

                # reserve 0 for unknown / unseen categories
                df[col] = (cat.codes + 1).astype(np.int64)
                categorical_cols.append(col)

        if continuous_cols:
            if fit_scaler:
                self.scaler = StandardScaler()
                df[continuous_cols] = self.scaler.fit_transform(df[continuous_cols])
            elif self.scaler is not None:
                df[continuous_cols] = self.scaler.transform(df[continuous_cols])
            else:
                self.scaler = StandardScaler()
                df[continuous_cols] = self.scaler.fit_transform(df[continuous_cols])

        self.categorical_cols = categorical_cols
        self.continuous_cols = continuous_cols
        self.categories = tuple(int(df[col].max()) + 1 for col in self.categorical_cols)
        self.num_continuous = len(self.continuous_cols)

        return df

    def _create_local_path(self, row: pd.Series) -> Optional[str]:
        fname = row.get("midas_file_name", "")
        if not fname or pd.isna(fname):
            return None
        return os.path.join(self.image_root, os.path.basename(str(fname)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]

        # image
        img_path = self._create_local_path(row)
        if img_path and os.path.exists(img_path):
            with Image.open(img_path) as img:
                image = img.convert("RGB")
        else:
            image = Image.new("RGB", (224, 224), color="black")

        if self.transform is not None:
            image = self.transform(image)
        else:
            image = self.to_tensor(image)

        # categorical metadata
        if self.categorical_cols:
            x_categ = torch.tensor(
                row[self.categorical_cols].values.astype(np.int64),
                dtype=torch.long
            )
        else:
            x_categ = torch.empty(0, dtype=torch.long)

        # continuous metadata
        if self.continuous_cols:
            x_cont = torch.tensor(
                row[self.continuous_cols].values.astype(np.float32),
                dtype=torch.float32
            )
        else:
            x_cont = torch.empty(0, dtype=torch.float32)

        label = torch.tensor(row["label"], dtype=torch.float32)

        return {
            "image": image,
            "x_categ": x_categ,
            "x_cont": x_cont,
            "label": label,
        }