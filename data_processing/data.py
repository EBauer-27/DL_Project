import os
import numpy as np
import pandas as pd
from PIL import Image
import torch
from torch.utils.data import Dataset
from sklearn.preprocessing import StandardScaler

class MIDASDataset(Dataset):
    def __init__(
        self,
        file_path: str,
        image_root: str,
        transform=None,
        is_training: bool = True,
    ):
        """
        Unified Dataset for Skin Lesion Classification & Metadata Investigation.
        """
        self.image_root = image_root
        self.transform = transform
        self.is_training = is_training

        # stateful preprocessing artifacts
        self.scaler = None
        self.cat_maps = {}
        self.feature_columns = None

        # 1. Load Data
        if file_path.endswith('.xlsx'):
            self.df = pd.read_excel(file_path)
        else:
            self.df = pd.read_csv(file_path)

        # 2. Create label column first (avoid overwriting text columns when imputing)
        self.df = self._create_label_column(self.df)

        # 3. Preprocess tabular (impute/scale/encode). Scaler and encoders are fit on whatever rows are passed here.
        self.df = self._preprocess_tabular(self.df)


    def _create_label_column(self, df):
        """Creates a label column from column midas_path and based on key words make it binary classification. All missing values are kept to be used as test set later"""
        danger_words = ['melanoma', 'bcc', 'scc', 'carcinoma', 'malignant', 'mct', 'ak']
        # make a safe copy and ensure we treat non-string as missing
        midas = df.get('midas_path')
        if midas is None:
            # if column doesn't exist create empty labels
            df['label'] = 0
            return df
        # replace NaN with empty string, coerce non-strings to string and lowercase
        label_df = midas.fillna('').astype(str).str.lower()
        # create new column 'label' with 1 if any of the danger words are in the midas_path, else 0
        df['label'] = label_df.apply(lambda x: 1 if any(w in x for w in danger_words) else 0)
        return df
    
    def _preprocess_tabular(self, df):
        """Preprocesses the tabular data: handle missing values, encode categoricals, and scale numericals."""
        df = df.copy()
        # don't impute object/text columns with 0 (which would break .str operations)
        object_cols = df.select_dtypes(include=['object']).columns.tolist()
        numerical_cols = df.select_dtypes(include=[np.number]).columns.tolist()

        # Exclude identifier/path/label columns from scaling if they appear as numeric
        exclude_cols = {'midas_path', 'midas_file_name', 'label'}
        numerical_cols = [c for c in numerical_cols if c not in exclude_cols]

        # Impute
        df[object_cols] = df[object_cols].fillna('')
        df[numerical_cols] = df[numerical_cols].fillna(0)

        # Handle object/categorical columns: try boolean-like mapping, else categorical codes
        for col in list(object_cols):
            if col in exclude_cols:
                continue
            ser = df[col].astype(str).str.strip()
            lower = ser.str.lower()
            # boolean-like
            if lower.isin(['yes', 'no', 'y', 'n', 'true', 'false', '0', '1']).all():
                df[col] = lower.replace({'yes': 1, 'y': 1, 'true': 1, '1': 1, 'no': 0, 'n': 0, 'false': 0, '0': 0}).astype(np.float32)
            else:
                # convert to categorical codes and save mapping
                cat = pd.Categorical(ser)
                codes = cat.codes.astype(np.int32)
                df[col] = codes
                # store categories so other datasets can align (if desired)
                self.cat_maps[col] = list(cat.categories)

        # Now scale numerical columns and store scaler for reuse
        if numerical_cols:
            scaler = StandardScaler()
            df[numerical_cols] = scaler.fit_transform(df[numerical_cols])
            self.scaler = scaler

        # Determine feature columns (exclude paths and label)
        self.feature_columns = [c for c in df.columns if c not in ['midas_path', 'midas_file_name', 'label']]

        return df
    
    def _create_local_path(self, row):
        """Creates a local path for the image based on the midas_file_name column."""
        fname = row.get('midas_file_name', '')
        if not fname or pd.isna(fname):
            return None
        # if the metadata contains full paths, prefer basename join
        return os.path.join(self.image_root, os.path.basename(str(fname)))

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        # Load Image
        img_path = self._create_local_path(self.df.iloc[idx])
        if img_path and os.path.exists(img_path):
            image = Image.open(img_path).convert('RGB')
        else:
            # fallback blank image (could alternatively raise)
            image = Image.new('RGB', (224, 224), color='black')
        if self.transform:
            image = self.transform(image)
        # Load Tabular using deterministic feature column order
        if self.feature_columns is None:
            # fallback: drop the usual columns
            cols = [c for c in ['midas_path', 'midas_file_name', 'label'] if c in self.df.columns]
            tab_vals = self.df.drop(columns=cols).iloc[idx].values
        else:
            tab_vals = self.df[self.feature_columns].iloc[idx].values
        tabular_data = tab_vals.astype(np.float32)
        tabular_data = torch.tensor(tabular_data)
        # Load Label
        label = torch.tensor(self.df['label'].iloc[idx]).float()
        return {
            'image': image,
            'tabular': tabular_data,
            'label': label
        }   