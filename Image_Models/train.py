from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from tqdm import tqdm

import shutil
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Trainer:
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, end_epoch, save=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(self.device)
        self.model = model.to(self.device)

        self.optimizer = optimizer
        self.criterion = criterion
        
        self.train_loader = train_loader
        self.valid_loader = valid_loader

        self.end_epoch = end_epoch

        self.save = save

        if save:
            os.mkdir('models_saved', exist_ok=True)
            name = datetime.now().strftime("%Y_%m_%d_%H_%M_%S_")
            self.save_path = os.path.join('models_saved', name)


    def train_step(self):
        loss_list = []
        self.model.train()
        for batch in self.train_loader:
            img = batch['image']
            img = img.to(self.device)

            pred = self.model(img).squeeze(1)
            
            label = batch['label']
            label = label.to(self.device).float()
            loss = self.criterion(pred, label)
            loss_list.append(loss.item())

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

        return round(np.mean(loss_list), 5)

    @torch.no_grad()
    def valid_step(self, epoch):
        self.model.eval()

        loss_list = []
        for batch in self.valid_loader:
            img = batch['image']
            img = img.to(self.device)

            pred = self.model(img).squeeze(1)

            label = batch['label']
            label = label.to(self.device).float()
            loss = self.criterion(pred, label)
            loss_list.append(loss.item())

        return round(np.mean(loss_list), 5)
    
    def fit(self):
        progress = tqdm(range(self.end_epoch))

        train_loss = []
        val_loss = []

        for epoch in progress:
            loss = self.train_step()
            train_loss.append(loss)

            progress.set_description(f'Epoch: {epoch}/{self.end_epoch}, Train Loss:{loss}')
        
            if (epoch+1)%10 == 0:
                loss = self.valid_step(epoch)
                val_loss.append(loss)

        if self.save:
            torch.save({'epoch': epoch,
                       'model_state_dict': self.model.state_dict(),
                       'optimizer_state_dict': self.optimizer.state_dict()}, self.save_path)

            print('Model saved!')
            
        return train_loss, val_loss

        