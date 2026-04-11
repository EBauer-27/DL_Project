import os
from itertools import product

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torch.utils.tensorboard.summary import hparams
from torchvision import transforms

from data_processing.data import MIDASDataset
from Image_Models.Models import ResNet18, VggNet, GoogLeNet
from Image_Models.train import Trainer

## CHECK OPTUNA !!!!!

def build_model(model_name):
    if model_name == 'resnet':
        return ResNet18()
    elif model_name == 'googlenet':
        return GoogLeNet()
    elif model_name == 'vggnet':
        return VggNet()
    else:
        raise ValueError(f'Unknown model: {model_name}')


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
TRAIN_PATH = 'manifests/train.csv'
VAL_PATH   = 'manifests/val.csv'
EPOCHS = 100

search_space = {'model': ['resnet', 'googlenet', 'vggnet'],
                'lr': [1e-2, 1e-3, 1e-4],
                'batch_size': [32, 64],
                'weight_decay': [1e-4, 1e-5],
                }

train_transforms = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

val_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]), # adjust mean/std if needed (currently set to [0.5] for grayscale images; change to ImageNet stats if using pretrained models on RGB)
])


keys = list(search_space.keys())
values = list(search_space.values())

base_logdir = "runs/hparam_search"
os.makedirs(base_logdir, exist_ok=True)

best_result = None
best_score = -1

for run_id, combo in enumerate(product(*values)):
    config = dict(zip(keys, combo))
    print(f'Run {run_id}: {config}')

    train_loader, val_loader = data_loaders(TRAIN_PATH, VAL_PATH, IMG_ROOT, train_transforms, val_transforms, batch_size=config['batch_size'])
    model = build_model(config['model'])

    optimizer = torch.optim.AdamW(model.parameters(), lr=config['lr'], weight_decay=config['weight_decay'])
    criterion = nn.BCEWithLogitsLoss()

    trainer = Trainer(
            model=model,
            train_loader=train_loader,
            valid_loader=val_loader,
            criterion=criterion,
            optimizer=optimizer,
            end_epoch=EPOCHS
        )
    
    train_losses, val_losses = trainer.fit()

    final_train_loss = train_losses[-1]
    final_val_loss = val_losses[-1] if len(val_losses) > 0 else None

    run_name = (
        f"run_{run_id}_"
        f"{config['model']}_"
        f"lr{config['lr']}_"
        f"bs{config['batch_size']}_"
        f"wd{config['weight_decay']}"
    )

    writer = SummaryWriter(log_dir=os.path.join(base_logdir, run_name))

    for k, v in config.items():
        if isinstance(v, str):
            writer.add_text(f"hparams/{k}", str(v))
        else:
            writer.add_scalar(f"hparams/{k}", v, 0)

    for epoch, loss in enumerate(train_losses):
        writer.add_scalar("loss/train", loss, epoch)

    for epoch, loss in enumerate(val_losses):
        writer.add_scalar("loss/val", loss, epoch)

    metric_dict = {
        "hparam/final_train_loss": final_train_loss,
        "hparam/final_val_loss": final_val_loss if final_val_loss is not None else float("nan"),
    }

    exp, ssi, sei = hparams(config, metric_dict)
    writer.file_writer.add_summary(exp)
    writer.file_writer.add_summary(ssi)
    writer.file_writer.add_summary(sei)

    writer.close()

    if final_val_loss is not None:
            score = -final_val_loss
            if score > best_score:
                best_score = score
                best_result = {
                    "config": config,
                    "final_train_loss": final_train_loss,
                    "final_val_loss": final_val_loss,
                }

    print("\nBest result:")
    print(best_result)





