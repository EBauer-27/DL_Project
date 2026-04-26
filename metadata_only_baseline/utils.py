import random
import numpy as np
import torch
from sklearn.metrics import accuracy_score, precision_score, recall_score, roc_auc_score, f1_score
from sklearn.metrics import roc_auc_score
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns



# ============================================================
# Reproducibility
# ============================================================
def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


# ============================================================
# Metrics
# ============================================================
def compute_metrics(y_true, y_probs, threshold=0.5):
    y_true = np.array(y_true)
    y_probs = np.array(y_probs)
    y_pred = (y_probs >= threshold).astype(int)

    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec = recall_score(y_true, y_pred, zero_division=0)
    f1 = f1_score(y_true, y_pred, zero_division=0)

    if len(np.unique(y_true)) > 1:
        auc = roc_auc_score(y_true, y_probs)
    else:
        auc = float("nan")

    return {
        "accuracy": acc,
        "precision": prec,
        "recall": rec,
        "f1": f1,
        "auc": auc,
    }




# ============================================================
# Inference
# ============================================================
@torch.no_grad()
def get_predictions(model, loader, device):
    model.eval()
    all_probs = []
    all_labels = []

    for x_categ, x_cont, y in loader:
        x_categ = x_categ.to(device)
        x_cont = x_cont.to(device)

        logits = model(x_categ, x_cont)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()

        all_probs.extend(probs.tolist())
        all_labels.extend(y.numpy().tolist())

    return np.array(all_labels), np.array(all_probs)


@torch.no_grad()
def permutation_feature_importance_tabtransformer(
    model,
    dataset,
    device,
    batch_size=32,
    metric="auc",
    n_repeats=5,
    random_state=42,
):
    """
    Compute permutation feature importance for a trained TabTransformer.

    Returns:
        importance_df: pandas DataFrame with feature names and importance scores
    """
    rng = np.random.default_rng(random_state)

    # original feature names
    cat_cols = list(dataset.categorical_cols)
    cont_cols = list(dataset.continuous_cols)
    feature_names = cat_cols + cont_cols

    # collect full dataset tensors once
    x_categ_all = []
    x_cont_all = []
    y_all = []

    for i in range(len(dataset)):
        x_categ, x_cont, y = dataset[i]
        x_categ_all.append(x_categ.numpy())
        x_cont_all.append(x_cont.numpy())
        y_all.append(float(y))

    x_categ_all = np.stack(x_categ_all) if len(cat_cols) > 0 else np.empty((len(dataset), 0), dtype=np.int64)
    x_cont_all = np.stack(x_cont_all) if len(cont_cols) > 0 else np.empty((len(dataset), 0), dtype=np.float32)
    y_all = np.array(y_all)

    # baseline performance
    baseline_probs = []

    model.eval()
    n_samples = len(dataset)
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)

        x_cat_batch = torch.tensor(x_categ_all[start:end], dtype=torch.long, device=device)
        x_cont_batch = torch.tensor(x_cont_all[start:end], dtype=torch.float32, device=device)

        logits = model(x_cat_batch, x_cont_batch)
        probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
        baseline_probs.extend(probs.tolist())

    baseline_probs = np.array(baseline_probs)

    if metric == "auc":
        baseline_score = roc_auc_score(y_all, baseline_probs)
    else:
        raise ValueError("Currently only metric='auc' is supported.")

    importances = []

    # categorical features
    for j, col in enumerate(cat_cols):
        drops = []

        for _ in range(n_repeats):
            x_categ_perm = x_categ_all.copy()
            perm_idx = rng.permutation(n_samples)
            x_categ_perm[:, j] = x_categ_perm[perm_idx, j]

            perm_probs = []
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)

                x_cat_batch = torch.tensor(x_categ_perm[start:end], dtype=torch.long, device=device)
                x_cont_batch = torch.tensor(x_cont_all[start:end], dtype=torch.float32, device=device)

                logits = model(x_cat_batch, x_cont_batch)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                perm_probs.extend(probs.tolist())

            perm_score = roc_auc_score(y_all, np.array(perm_probs))
            drops.append(baseline_score - perm_score)

        importances.append((col, float(np.mean(drops)), float(np.std(drops))))

    # continuous features
    for j, col in enumerate(cont_cols):
        drops = []

        for _ in range(n_repeats):
            x_cont_perm = x_cont_all.copy()
            perm_idx = rng.permutation(n_samples)
            x_cont_perm[:, j] = x_cont_perm[perm_idx, j]

            perm_probs = []
            for start in range(0, n_samples, batch_size):
                end = min(start + batch_size, n_samples)

                x_cat_batch = torch.tensor(x_categ_all[start:end], dtype=torch.long, device=device)
                x_cont_batch = torch.tensor(x_cont_perm[start:end], dtype=torch.float32, device=device)

                logits = model(x_cat_batch, x_cont_batch)
                probs = torch.sigmoid(logits).squeeze(1).cpu().numpy()
                perm_probs.extend(probs.tolist())

            perm_score = roc_auc_score(y_all, np.array(perm_probs))
            drops.append(baseline_score - perm_score)

        importances.append((col, float(np.mean(drops)), float(np.std(drops))))

    importance_df = pd.DataFrame(importances, columns=["feature", "importance_mean", "importance_std"])
    importance_df = importance_df.sort_values("importance_mean", ascending=False).reset_index(drop=True)

    return importance_df, baseline_score

# ============================================================
# Plots
# ============================================================

def plot_feature_importance(importance_df, top_k=None, title="Permutation Feature Importance"):
    df = importance_df.copy()

    if top_k is not None:
        df = df.head(top_k)

    plt.figure(figsize=(10, max(4, 0.5 * len(df))))
    plt.barh(df["feature"][::-1], df["importance_mean"][::-1], xerr=df["importance_std"][::-1])
    plt.xlabel("Mean drop in AUC after permutation")
    plt.ylabel("Feature")
    plt.title(title)
    plt.tight_layout()
    plt.show()


# ============================================================
# Attention Recording
# ============================================================


def find_attention_modules(model):
    attention_modules = []
    for name, module in model.named_modules():
        cls_name = module.__class__.__name__.lower()
        if "attention" in cls_name or cls_name == "attn":
            attention_modules.append((name, module))
    return attention_modules


class AttentionRecorder:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.records = []

    def _hook(self, name):
        def fn(module, inputs, output):
            rec = {"name": name, "output_type": type(output).__name__}

            if isinstance(output, tuple):
                rec["tuple_len"] = len(output)
                rec["output"] = output
            else:
                rec["output"] = output

            self.records.append(rec)
        return fn

    def attach(self):
        for name, module in find_attention_modules(self.model):
            h = module.register_forward_hook(self._hook(name))
            self.handles.append(h)

    def clear(self):
        self.records = []

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []



class TabTransformerAttentionRecorder:
    def __init__(self, model):
        self.model = model
        self.handles = []
        self.attn_outputs = []

    def _hook(self, module, inputs, output):
        # expected output: (x, attn)
        if isinstance(output, tuple) and len(output) == 2:
            _, attn = output
            if torch.is_tensor(attn) and attn.ndim == 4:
                self.attn_outputs.append(attn.detach().cpu())

    def attach(self):
        for name, module in self.model.named_modules():
            if "transformer.layers" in name and name.endswith("0.branch.fn"):
                handle = module.register_forward_hook(self._hook)
                self.handles.append(handle)

    def clear(self):
        self.attn_outputs = []

    def remove(self):
        for h in self.handles:
            h.remove()
        self.handles = []

@torch.no_grad()
def collect_mean_attention_matrix(model, loader, device, max_batches=None):
    recorder = TabTransformerAttentionRecorder(model)
    recorder.attach()

    total_attn = None
    n_batches = 0

    model.eval()

    for batch_idx, (x_categ, x_cont, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        recorder.clear()

        x_categ = x_categ.to(device)
        x_cont = x_cont.to(device)

        _ = model(x_categ, x_cont)

        if not recorder.attn_outputs:
            continue

        # recorder.attn_outputs is a list of layer attentions
        # each tensor has shape [B, H, T, T]
        layer_stack = torch.stack(recorder.attn_outputs, dim=0)   # [L, B, H, T, T]

        # average over layers, heads, and batch
        mean_attn = layer_stack.mean(dim=0).mean(dim=1).mean(dim=0)   # [T, T]

        if total_attn is None:
            total_attn = mean_attn
        else:
            total_attn += mean_attn

        n_batches += 1

    recorder.remove()

    if total_attn is None or n_batches == 0:
        raise RuntimeError("No attention matrices were collected.")

    return (total_attn / n_batches).numpy()


def plot_attention_heatmap(attn_matrix, feature_names, title="Mean Attention Between Categorical Features"):
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        attn_matrix,
        xticklabels=feature_names,
        yticklabels=feature_names,
        annot=True,
        fmt=".2f",
        cmap="viridis"
    )
    plt.xlabel("Key / attended feature")
    plt.ylabel("Query feature")
    plt.title(title)
    plt.tight_layout()
    plt.show()



@torch.no_grad()
def collect_layerwise_attention_matrices(model, loader, device, max_batches=None):
    """
    Returns:
        attn_layers: numpy array of shape [L, T, T]
            mean attention per layer, averaged over samples and heads
    """
    recorder = TabTransformerAttentionRecorder(model)
    recorder.attach()

    total_attn = None
    n_batches = 0

    model.eval()

    for batch_idx, (x_categ, x_cont, _) in enumerate(loader):
        if max_batches is not None and batch_idx >= max_batches:
            break

        recorder.clear()

        x_categ = x_categ.to(device)
        x_cont = x_cont.to(device)

        _ = model(x_categ, x_cont)

        if not recorder.attn_outputs:
            continue

        # list of [B, H, T, T], one per layer
        layer_stack = torch.stack(recorder.attn_outputs, dim=0)   # [L, B, H, T, T]

        # average over batch and heads, keep layers
        mean_per_layer = layer_stack.mean(dim=1).mean(dim=1)      # [L, T, T]

        if total_attn is None:
            total_attn = mean_per_layer
        else:
            total_attn += mean_per_layer

        n_batches += 1

    recorder.remove()

    if total_attn is None or n_batches == 0:
        raise RuntimeError("No attention matrices were collected.")

    return (total_attn / n_batches).numpy()


def plot_layerwise_attention_heatmaps(attn_layers, feature_names, title_prefix="Layer"):
    """
    attn_layers: [L, T, T]
    """
    n_layers = attn_layers.shape[0]
    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))

    if n_layers == 1:
        axes = [axes]

    vmin = attn_layers.min()
    vmax = attn_layers.max()

    for i, ax in enumerate(axes):
        sns.heatmap(
            attn_layers[i],
            xticklabels=feature_names,
            yticklabels=feature_names,
            annot=True,
            fmt=".2f",
            cmap="viridis",
            vmin=vmin,
            vmax=vmax,
            ax=ax,
            cbar=(i == n_layers - 1),
        )
        ax.set_title(f"{title_prefix} {i}")
        ax.set_xlabel("Key / attended feature")
        if i == 0:
            ax.set_ylabel("Query feature")
        else:
            ax.set_ylabel("")

    plt.tight_layout()
    return fig


def plot_layerwise_attention_deviation(attn_layers, feature_names, title_prefix="Layer"):
    """
    Plot attention minus uniform attention.
    """
    n_layers, n_tokens, _ = attn_layers.shape
    uniform = 1.0 / n_tokens
    delta = attn_layers - uniform

    fig, axes = plt.subplots(1, n_layers, figsize=(4 * n_layers, 4))

    if n_layers == 1:
        axes = [axes]

    abs_max = np.abs(delta).max()

    for i, ax in enumerate(axes):
        sns.heatmap(
            delta[i],
            xticklabels=feature_names,
            yticklabels=feature_names,
            annot=True,
            fmt=".3f",
            cmap="coolwarm",
            center=0.0,
            vmin=-abs_max,
            vmax=abs_max,
            ax=ax,
            cbar=(i == n_layers - 1),
        )
        ax.set_title(f"{title_prefix} {i}\nΔ from uniform")
        ax.set_xlabel("Key / attended feature")
        if i == 0:
            ax.set_ylabel("Query feature")
        else:
            ax.set_ylabel("")

    plt.tight_layout()
    return fig