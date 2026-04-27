import os
from itertools import product

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from torchvision import transforms
import optuna

from data_processing.data import MIDASDataset
from image_only_baseline.models import create_model
from image_only_baseline.trainer import Trainer



def data_loaders(train_path, val_path, img_root, train_tf, val_tf, batch_size):
    train_ds = MIDASDataset(file_path=train_path, 
                            image_root=img_root, 
                            transform=train_tf, 
                            is_training=True)
    
    val_ds   = MIDASDataset(file_path=val_path, 
                            image_root=img_root, 
                            transform=val_tf, 
                            is_training=False)

    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False)

    return train_loader, val_loader 


IMG_ROOT = 'data/MRA-MIDAS/midas_224_cache/'
TRAIN_PATH = 'manifests_record_split/train.csv'
VAL_PATH   = 'manifests_record_split/val.csv'

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])



val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]), 
])

def objective(trial):
    model_name = trial.suggest_categorical('model_name', ['resnet18', 'resnet50', 'googlenet', 'vgg16'])
    lr = trial.suggest_float('lr', 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical('batch_size', [32, 64])
    weight_decay = trial.suggest_float('weight_decay', 1e-6, 1e-3, log=True)

    epochs = 50

    train_loader, val_loader = data_loaders(TRAIN_PATH, VAL_PATH, IMG_ROOT, train_transforms, val_transforms, batch_size=batch_size)
    model = create_model(model_name)

    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.BCEWithLogitsLoss()

    trainer = trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            end_epoch=epochs,
            save=True
        )
    
    history = trainer.fit(trial=trial)

    best_val_auc = max(history['val_auc'])
    trial.set_user_attr("best_val_loss", min(history["val_loss"]))
    trial.set_user_attr("final_val_auc", history["val_auc"][-1])
    trial.set_user_attr("final_val_f1", history["val_f1"][-1])
    trial.set_user_attr("final_val_acc", history["val_acc"][-1])
    trial.set_user_attr("final_val_pres", history["val_pres"][-1])

    os.makedirs('Image_Models/checkpoints_opt', exist_ok=True)
    best_model_path = f"Image_Models/checkpoints_opt/trial_{trial.number}.pt"
    torch.save(model.state_dict(), best_model_path)
    trial.set_user_attr("model_path", best_model_path)

    return best_val_auc

if __name__ == '__main__':
    os.makedirs('Image_Models/optuna_db', exist_ok=True)

    study = optuna.create_study(study_name='img_models',
                                storage='sqlite:///Image_Models/optuna_db/image_model_hyperopt_final.db',
                                load_if_exists=True,
                                direction='maximize',
                                pruner=optuna.pruners.MedianPruner(n_startup_trials=5,
                                                                   n_warmup_steps=5),
                                )
    
    #study.optimize(objective, n_trials=20, show_progress_bar=False)

    try:
        study.optimize(objective, n_trials=20)
    except Exception as e:
        print("TOP LEVEL:", repr(e))
        print("CAUSE:", repr(getattr(e, "__cause__", None)))
        raise

    print("\nBest trial:")
    print("Value (best val_auc):", study.best_trial.value)
    print("Params:")
    for k, v in study.best_trial.params.items():
        print(f"  {k}: {v}")





