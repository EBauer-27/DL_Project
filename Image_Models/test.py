import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from data_processing.data import MIDASDataset

from Image_Models.Models import ResNet18, GoogLeNet, VggNet
from Image_Models.train import Trainer

import matplotlib.pyplot as plt
import numpy as np


IMG_ROOT = 'data/MRA-MIDAS/midas_224_cache/'
TRAIN_PATH = 'manifests/train.csv'
VAL_PATH   = 'manifests/val.csv'

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
train_ds = MIDASDataset(file_path=TRAIN_PATH, image_root=IMG_ROOT, transform=train_transforms, is_training=True)
val_ds   = MIDASDataset(file_path=VAL_PATH,   image_root=IMG_ROOT, transform=val_transforms, is_training=False) 

train_loader = DataLoader(train_ds, batch_size=32, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=32, shuffle=False)

model = VggNet()

optimizer = torch.optim.AdamW(model.parameters(), lr=0.001)
criterion = nn.BCEWithLogitsLoss()

train = Trainer(model=model,
                train_loader=train_loader,
                valid_loader=val_loader,
                criterion=criterion,
                optimizer=optimizer,
                end_epoch=20)

train_loss, val_loss = train.fit()


fig, ax = plt.subplots(1, 2)
ax[0].plot(train_loss)
ax[1].plot(val_loss)
plt.show()