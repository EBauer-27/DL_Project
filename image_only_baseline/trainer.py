import os
from datetime import datetime
from typing import Dict, Optional

import numpy as np
import optuna
import torch
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, roc_auc_score
from tqdm import tqdm


class Trainer:
    def __init__(
        self,
        model: torch.nn.Module,
        train_loader,
        valid_loader,
        criterion,
        optimizer,
        end_epoch: int,
        checkpoint_dir: str = "Image_Models/checkpoints_opt",
        save: bool = False,
        threshold: float = 0.5,
    ):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.criterion = criterion
        self.optimizer = optimizer

        self.end_epoch = end_epoch
        self.save = save
        self.threshold = threshold

        self.checkpoint_dir = checkpoint_dir
        os.makedirs(self.checkpoint_dir, exist_ok=True)

        self.best_auc = -float("inf")
        self.best_model_path = None

    def _forward_logits(self, images: torch.Tensor) -> torch.Tensor:
        logits = self.model(images)

        if isinstance(logits, tuple):
            logits = logits[0]

        return logits.view(-1)

    def train_step(self) -> float:
        self.model.train()
        losses = []

        for batch in self.train_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).float().view(-1)

            logits = self._forward_logits(images)
            loss = self.criterion(logits, labels)

            self.optimizer.zero_grad(set_to_none=True)
            loss.backward()
            self.optimizer.step()

            losses.append(loss.item())

        return float(np.mean(losses))

    @torch.no_grad()
    def valid_step(self) -> Dict[str, float]:
        self.model.eval()

        losses = []
        all_labels = []
        all_probs = []

        for batch in self.valid_loader:
            images = batch["image"].to(self.device)
            labels = batch["label"].to(self.device).float().view(-1)

            logits = self._forward_logits(images)
            loss = self.criterion(logits, labels)

            probs = torch.sigmoid(logits)

            losses.append(loss.item())
            all_labels.extend(labels.cpu().numpy().tolist())
            all_probs.extend(probs.cpu().numpy().tolist())

        all_labels = np.asarray(all_labels).astype(int)
        all_probs = np.asarray(all_probs)
        all_preds = (all_probs >= self.threshold).astype(int)

        try:
            auc = float(roc_auc_score(all_labels, all_probs))
        except ValueError:
            auc = 0.5

        return {
            "val_loss": float(np.mean(losses)),
            "val_auc": auc,
            "val_f1": float(f1_score(all_labels, all_preds, zero_division=0)),
            "val_acc": float(accuracy_score(all_labels, all_preds)),
            "val_precision": float(precision_score(all_labels, all_preds, zero_division=0)),
            "val_recall": float(recall_score(all_labels, all_preds, zero_division=0)),
        }

    def _save_checkpoint(self, epoch: int, metric: float, trial_number: Optional[int] = None) -> str:
        trial_part = f"trial_{trial_number}" if trial_number is not None else datetime.now().strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.checkpoint_dir, f"{trial_part}_best.pt")

        torch.save(
            {
                "epoch": epoch,
                "model_state_dict": self.model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "best_val_auc": metric,
            },
            path,
        )

        self.best_model_path = path
        return path

    def fit(self, trial: Optional[optuna.Trial] = None) -> Dict[str, list]:
        history = {
            "train_loss": [],
            "val_loss": [],
            "val_auc": [],
            "val_f1": [],
            "val_acc": [],
            "val_precision": [],
            "val_recall": [],
        }

        progress = tqdm(range(1, self.end_epoch + 1), desc="Training")

        for epoch in progress:
            train_loss = self.train_step()
            history["train_loss"].append(train_loss)

            should_validate = (epoch % 10 == 0) or (epoch == self.end_epoch)

            if should_validate:
                val_metrics = self.valid_step()

                for key, value in val_metrics.items():
                    history[key].append(value)

                val_auc = val_metrics["val_auc"]
                progress.set_postfix(
                    train_loss=f"{train_loss:.4f}",
                    val_auc=f"{val_auc:.4f}",
                )

                if val_auc > self.best_auc:
                    self.best_auc = val_auc
                    if self.save:
                        self._save_checkpoint(
                            epoch=epoch,
                            metric=val_auc,
                            trial_number=trial.number if trial is not None else None,
                        )

                if trial is not None:
                    trial.report(val_auc, step=epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()

            else:
                progress.set_postfix(train_loss=f"{train_loss:.4f}")

        return history


