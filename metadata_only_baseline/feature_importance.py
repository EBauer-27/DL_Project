import pandas as pd
import numpy as np
import torch
from torch.utils.data import DataLoader
from tab_transformer_pytorch import TabTransformer
from metadata_only_baseline.utils import permutation_feature_importance_tabtransformer, plot_feature_importance
from data_processing.data import MIDASTabularDataset
import torch.nn as nn


"Perform different visualizations for evaluation of the tabtransform, feature importance, feature weights matrix, etc."
TEST_PATH = "manifests/test.csv"
MODEL_PATH = "metadata_only_baseline/best_tabtransformer.pth"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

categories = tuple(checkpoint["categories"])
num_continuous = int(checkpoint["num_continuous"])
scaler = checkpoint["scaler"]
cat_maps = checkpoint["cat_maps"]

# Recreate model with same training config
model = TabTransformer(
    categories=categories,
    num_continuous=num_continuous,
    dim=32,
    dim_out=1,
    depth=6,
    heads=8,
    attn_dropout=0.1,
    ff_dropout=0.1,
    mlp_hidden_mults=(4, 2),
    mlp_act=nn.ReLU(),
).to(device)

model.load_state_dict(checkpoint["model_state_dict"])
model.eval()

# Prepare test dataset with training preprocessing artifacts
test_ds = MIDASTabularDataset(
    file_path=TEST_PATH,
    is_training=False,
    scaler=scaler,
    cat_maps=cat_maps,
)
test_loader = DataLoader(test_ds, batch_size=BATCH_SIZE, shuffle=False)


importance_df, baseline_auc = permutation_feature_importance_tabtransformer(
    model=model,
    dataset=test_ds,   
    device=device,
    batch_size=32,
    metric="auc",
    n_repeats=5,
)

print("Baseline AUC:", baseline_auc)
print(importance_df)

plot_feature_importance(
    importance_df,
    top_k=None,
    title="TabTransformer Feature Importance on Test Set"
)