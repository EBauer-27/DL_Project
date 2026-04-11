import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tab_transformer_pytorch import TabTransformer

from .utils import compute_metrics
from data_processing.data import MIDASTabularDataset

TEST_PATH = "manifests/test.csv"
MODEL_PATH = "metadata_only_baseline/best_tabtransformer.pth"
BATCH_SIZE = 32

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    # Load checkpoint dict
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

    # Evaluate
    all_labels = []
    all_probs = []

    with torch.no_grad():
        for x_categ, x_cont, labels in test_loader:
            x_categ = x_categ.to(device)
            x_cont = x_cont.to(device)
            labels = labels.to(device)

            logits = model(x_categ, x_cont)
            probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
            labels = labels.cpu().numpy()

            all_labels.extend(labels.tolist())
            all_probs.extend(probs.tolist())

    metrics = compute_metrics(all_labels, all_probs)

    print(f"Test Accuracy:  {metrics['accuracy']:.4f}")
    print(f"Test Precision: {metrics['precision']:.4f}")
    print(f"Test Recall:    {metrics['recall']:.4f}")
    print(f"Test AUC:       {metrics['auc']:.4f}")


if __name__ == "__main__":
    main()