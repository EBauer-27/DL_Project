import pandas as pd 
import numpy as np 
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision
from data_processing.data import MIDASDataset
import matplotlib.pyplot as plt

"""Example usage: python -m training_scripts.tets"""

IMG_ROOT = 'data/MRA-MIDAS/midasmultimodalimagedatasetforaibasedskincancer/'
TRAIN_PATH = 'manifests/train.csv'
VAL_PATH   = 'manifests/val.csv'



train_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]),
])

val_transforms = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5], std=[0.5]), # adjust mean/std if needed (currently set to [0.5] for grayscale images; change to ImageNet stats if using pretrained models on RGB)
])
train_ds = MIDASDataset(file_path=TRAIN_PATH, image_root=IMG_ROOT, transform=train_transforms, is_training=True)
val_ds   = MIDASDataset(file_path=VAL_PATH,   image_root=IMG_ROOT, transform=val_transforms, is_training=False) 

train_loader = DataLoader(train_ds, batch_size=16, shuffle=True)
val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False)

#CHECK IF EVERYTHING WORKED CORRECTLY 
#paste an image after transformations 
def imshow(img):
    img = img / 2 + 0.5     # unnormalize
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.show()

# Get a batch of training data
dataiter = iter(train_loader)
batch = next(dataiter)
images = batch['image']
labels = batch['label']
imshow(torchvision.utils.make_grid(images))
print('batch tabular shape:', batch['tabular'].shape)
print('batch labels shape:', labels.shape)