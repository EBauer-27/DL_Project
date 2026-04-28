import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tab_transformer_pytorch import TabTransformer

from data_processing.data import MIDASTabularDataset
from metadata_only_baseline.utils import (
    collect_layerwise_attention_matrices,
    plot_layerwise_attention_heatmaps,
    plot_layerwise_attention_deviation,
)

TEST_PATH = "manifests_record_split/test.csv"
MODEL_PATH = "metadata_only_baseline/model/best_tabtransformer.pth"
BATCH_SIZE = 32
OUTPUT_DIR = "metadata_only_baseline/results"

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)

    categories = tuple(checkpoint["categories"])
    num_continuous = int(checkpoint["num_continuous"])
    scaler = checkpoint["scaler"]
    cat_maps = checkpoint["cat_maps"]

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

    test_ds = MIDASTabularDataset(
        file_path=TEST_PATH,
        is_training=False,
        scaler=scaler,
        cat_maps=cat_maps,
    )

    test_loader = DataLoader(
        test_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    feature_names = test_ds.categorical_cols

    attn_layers = collect_layerwise_attention_matrices(
        model=model,
        loader=test_loader,
        device=device,
        max_batches=None,
    )

    print("Categorical features:", feature_names)
    print("Attention layers shape:", attn_layers.shape)

    fig1 = plot_layerwise_attention_heatmaps(
        attn_layers,
        feature_names=feature_names,
        title_prefix="Layer",
    )
    fig1.savefig(
        os.path.join(OUTPUT_DIR, "layerwise_attention_raw.png"),
        dpi=300,
        bbox_inches="tight",
    )

    fig2 = plot_layerwise_attention_deviation(
        attn_layers,
        feature_names=feature_names,
        title_prefix="Layer",
    )
    fig2.savefig(
        os.path.join(OUTPUT_DIR, "layerwise_attention_delta.png"),
        dpi=300,
        bbox_inches="tight",
    )

    print("Saved raw attention heatmaps to:", os.path.join(OUTPUT_DIR, "layerwise_attention_raw.png"))
    print("Saved deviation heatmaps to:", os.path.join(OUTPUT_DIR, "layerwise_attention_delta.png"))


if __name__ == "__main__":
    main()