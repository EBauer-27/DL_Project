import os
import argparse
from copy import deepcopy

import optuna
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms

from data_processing.data import MIDASDataset
from image_only_baseline.models import create_model
from image_only_baseline.utils import (
    set_seed,
    ensure_dir,
    save_json,
    compute_pos_weight_from_dataset,
    train_one_epoch,
    evaluate,
    save_checkpoint,
)


# ------------------------------------------------------------
# Device helper
# ------------------------------------------------------------
def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")

    if torch.backends.mps.is_available():
        return torch.device("mps")

    return torch.device("cpu")


# ------------------------------------------------------------
# Transforms
# ------------------------------------------------------------
def get_transforms(augment: bool = True):
    imagenet_mean = [0.485, 0.456, 0.406]
    imagenet_std = [0.229, 0.224, 0.225]

    if augment:
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(degrees=20),
            transforms.ColorJitter(
                brightness=0.15,
                contrast=0.15,
                saturation=0.10,
                hue=0.03,
            ),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])
    else:
        train_tf = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
        ])

    val_tf = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=imagenet_mean, std=imagenet_std),
    ])

    return train_tf, val_tf


# ------------------------------------------------------------
# Optimizer
# ------------------------------------------------------------
def build_optimizer(model, lr, weight_decay):
    trainable_params = [p for p in model.parameters() if p.requires_grad]

    return torch.optim.AdamW(
        trainable_params,
        lr=lr,
        weight_decay=weight_decay,
    )


# ------------------------------------------------------------
# Optuna objective
# ------------------------------------------------------------
def objective(trial, args, device):
    set_seed(args.seed + trial.number)

    # --------------------------------------------------------
    # Hyperparameters
    # --------------------------------------------------------
    model_name = trial.suggest_categorical(
        "model_name",
        [
            "resnet18",
            "resnet50",
            "vgg16",
            "googlenet",
        ],
    )

    batch_size = trial.suggest_categorical(
        "batch_size",
        [
            16,
            32,
        ],
    )

    # VGG16 is large and prone to overfitting on a small dataset,
    # so we force feature extraction for VGG16.
    if model_name == "vgg16":
        freeze_backbone = True
    else:
        freeze_backbone = trial.suggest_categorical(
            "freeze_backbone",
            [True, False],
        )

    # Conditional LR:
    # frozen backbone -> only head trains -> higher LR is okay
    # unfrozen backbone -> full pretrained CNN trains -> smaller LR is safer
    if freeze_backbone:
        lr = trial.suggest_float(
            "lr",
            1e-4,
            1e-3,
            log=True,
        )
    else:
        lr = trial.suggest_float(
            "lr",
            1e-5,
            3e-4,
            log=True,
        )

    weight_decay = trial.suggest_float(
        "weight_decay",
        1e-6,
        1e-3,
        log=True,
    )

    dropout = trial.suggest_categorical(
        "dropout",
        [
            0.2,
            0.3,
            0.5,
        ],
    )

    augment = True

    config = {
        "model_name": model_name,
        "batch_size": batch_size,
        "lr": lr,
        "weight_decay": weight_decay,
        "dropout": dropout,
        "freeze_backbone": freeze_backbone,
        "augment": augment,
    }

    print("\n" + "=" * 80)
    print(f"Trial {trial.number}")
    print(config)
    print("=" * 80)

    # --------------------------------------------------------
    # Data
    # --------------------------------------------------------
    train_tf, val_tf = get_transforms(augment=augment)

    train_dataset = MIDASDataset(
        file_path=args.train_csv,
        image_root=args.image_root,
        transform=train_tf,
        is_training=True,
        fit_scaler=True,
    )

    val_dataset = MIDASDataset(
        file_path=args.val_csv,
        image_root=args.image_root,
        transform=val_tf,
        is_training=False,
        scaler=train_dataset.scaler,
        cat_maps=train_dataset.cat_maps,
        fit_scaler=False,
    )

    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples:   {len(val_dataset)}")

    pin_memory = device.type == "cuda"

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=pin_memory,
    )

    # --------------------------------------------------------
    # Model
    # --------------------------------------------------------
    model = create_model(
        model_name=model_name,
        pretrained=True,
        dropout=dropout,
        freeze_backbone=freeze_backbone,
    ).to(device)

    # --------------------------------------------------------
    # Loss
    # --------------------------------------------------------
    pos_weight = compute_pos_weight_from_dataset(train_dataset)

    if args.use_pos_weight and pos_weight is not None:
        pos_weight = pos_weight.to(device)
        criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
        print(f"Using pos_weight: {pos_weight.item():.4f}")
    else:
        criterion = nn.BCEWithLogitsLoss()
        print("Using BCEWithLogitsLoss without pos_weight.")

    # --------------------------------------------------------
    # Optimizer
    # --------------------------------------------------------
    optimizer = build_optimizer(
        model=model,
        lr=lr,
        weight_decay=weight_decay,
    )

    # --------------------------------------------------------
    # Training loop
    # --------------------------------------------------------
    best_val_auc = -1.0
    best_val_metrics = None
    best_state_dict = None
    best_epoch = -1

    patience_counter = 0

    for epoch in range(args.epochs):
        train_loss = train_one_epoch(
            model=model,
            dataloader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device,
        )

        val_metrics = evaluate(
            model=model,
            dataloader=val_loader,
            criterion=criterion,
            device=device,
            threshold=args.threshold,
        )

        val_auc = val_metrics["auc"]

        print(
            f"Trial {trial.number:03d} | "
            f"Epoch {epoch + 1:03d}/{args.epochs} | "
            f"Train loss: {train_loss:.4f} | "
            f"Val loss: {val_metrics['loss']:.4f} | "
            f"Val acc: {val_metrics['accuracy']:.4f} | "
            f"Val precision: {val_metrics['precision']:.4f} | "
            f"Val recall: {val_metrics['recall']:.4f} | "
            f"Val F1: {val_metrics['f1']:.4f} | "
            f"Val AUC: {val_auc:.4f}"
        )

        trial.report(val_auc, step=epoch)

        if trial.should_prune():
            print(f"Trial {trial.number} pruned at epoch {epoch + 1}")
            raise optuna.TrialPruned()

        if not pd.isna(val_auc) and val_auc > best_val_auc:
            best_val_auc = val_auc
            best_val_metrics = deepcopy(val_metrics)
            best_state_dict = deepcopy(model.state_dict())
            best_epoch = epoch + 1
            patience_counter = 0
        else:
            patience_counter += 1

        if patience_counter >= args.patience:
            print(f"Early stopping trial {trial.number} at epoch {epoch + 1}")
            break

    if best_val_metrics is None:
        best_val_metrics = val_metrics
        best_state_dict = deepcopy(model.state_dict())
        best_epoch = epoch + 1

    model.load_state_dict(best_state_dict)

    # --------------------------------------------------------
    # Save trial checkpoint
    # --------------------------------------------------------
    trial_dir = os.path.join(args.output_dir, f"trial_{trial.number:03d}")
    ensure_dir(trial_dir)

    checkpoint_path = os.path.join(trial_dir, "model.pth")

    save_checkpoint(
        model=model,
        config=config,
        metrics=best_val_metrics,
        path=checkpoint_path,
    )

    save_json(
        {
            "trial_number": trial.number,
            "best_epoch": best_epoch,
            "config": config,
            "metrics": best_val_metrics,
            "checkpoint_path": checkpoint_path,
        },
        os.path.join(trial_dir, "config_and_metrics.json"),
    )

    trial.set_user_attr("best_epoch", best_epoch)
    trial.set_user_attr("checkpoint_path", checkpoint_path)
    trial.set_user_attr("val_loss", best_val_metrics["loss"])
    trial.set_user_attr("val_accuracy", best_val_metrics["accuracy"])
    trial.set_user_attr("val_precision", best_val_metrics["precision"])
    trial.set_user_attr("val_recall", best_val_metrics["recall"])
    trial.set_user_attr("val_f1", best_val_metrics["f1"])
    trial.set_user_attr("val_auc", best_val_metrics["auc"])

    return best_val_auc


# ------------------------------------------------------------
# Main
# ------------------------------------------------------------
def main(args):
    ensure_dir(args.output_dir)
    ensure_dir(args.optuna_db_dir)

    set_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    sampler = optuna.samplers.TPESampler(
        seed=args.seed,
        multivariate=True,
    )

    pruner = optuna.pruners.SuccessiveHalvingPruner(
        min_resource=3,
        reduction_factor=3,
        min_early_stopping_rate=0,
    )

    study = optuna.create_study(
        study_name=args.study_name,
        storage=f"sqlite:///{args.optuna_db_dir}/{args.study_name}.db",
        load_if_exists=True,
        direction="maximize",
        sampler=sampler,
        pruner=pruner,
    )

    study.optimize(
        lambda trial: objective(trial, args, device),
        n_trials=args.n_trials,
        show_progress_bar=True,
    )

    print("\n" + "=" * 80)
    print("Optuna optimization finished.")
    print("=" * 80)

    print("Best trial:")
    print(f"Trial number: {study.best_trial.number}")
    print(f"Best validation AUC: {study.best_trial.value:.4f}")

    print("\nBest params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")

    print("\nBest user attrs:")
    for k, v in study.best_trial.user_attrs.items():
        print(f"  {k}: {v}")

    trials_df = study.trials_dataframe()
    trials_csv_path = os.path.join(args.output_dir, "optuna_trials.csv")
    trials_df.to_csv(trials_csv_path, index=False)

    best_checkpoint_path = study.best_trial.user_attrs.get("checkpoint_path")

    if best_checkpoint_path is not None:
        best_checkpoint = torch.load(
            best_checkpoint_path,
            map_location="cpu",
        )

        best_model_path = os.path.join(args.output_dir, "best_model.pth")
        torch.save(best_checkpoint, best_model_path)

        save_json(
            {
                "best_trial_number": study.best_trial.number,
                "best_value_val_auc": study.best_trial.value,
                "best_params": study.best_trial.params,
                "best_user_attrs": study.best_trial.user_attrs,
                "best_model_path": best_model_path,
                "search_strategy": "Optuna TPE search with SuccessiveHalvingPruner",
            },
            os.path.join(args.output_dir, "best_config.json"),
        )

        print(f"\nBest model saved to: {best_model_path}")

    print(f"All trials saved to: {trials_csv_path}")


# ------------------------------------------------------------
# CLI
# ------------------------------------------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--train_csv",
        type=str,
        default="manifests/train.csv",
    )

    parser.add_argument(
        "--val_csv",
        type=str,
        default="manifests/val.csv",
    )

    parser.add_argument(
        "--image_root",
        type=str,
        default="data/MRA-MIDAS/midasmultimodalimagedatasetforaibasedskincancer/",
        help="Root folder containing the real image files.",
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="image_only_baseline/results/optuna_hpo",
    )

    parser.add_argument(
        "--optuna_db_dir",
        type=str,
        default="image_only_baseline/results/optuna_db",
    )

    parser.add_argument(
        "--study_name",
        type=str,
        default="image_model_hpo",
    )

    parser.add_argument(
        "--n_trials",
        type=int,
        default=12,
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=30,
    )

    parser.add_argument(
        "--patience",
        type=int,
        default=7,
    )

    parser.add_argument(
        "--threshold",
        type=float,
        default=0.5,
    )

    parser.add_argument(
        "--num_workers",
        type=int,
        default=0,
    )

    parser.add_argument(
        "--seed",
        type=int,
        default=42,
    )

    parser.add_argument(
        "--use_pos_weight",
        action="store_true",
    )

    args = parser.parse_args()
    main(args)