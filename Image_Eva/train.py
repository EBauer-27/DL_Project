from PIL import Image
import matplotlib.pyplot as plt
import os
import torch
import numpy as np
from tqdm import tqdm
from sklearn.metrics import roc_auc_score, f1_score, accuracy_score, precision_score
import optuna

import shutil
from torch.utils.tensorboard import SummaryWriter
from datetime import datetime

class Trainer:
    def __init__(self, model, train_loader, valid_loader, criterion, optimizer, end_epoch, trail='test', save=False):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
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

        TBOARD = f'Image_Models/Tensorboard/Hyperopt/trial_{trail}'
        os.makedirs(TBOARD)
        

        shutil.rmtree(TBOARD)
        self.writer = SummaryWriter(TBOARD)


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
        all_labels = []
        all_probs = []

        for batch in self.valid_loader:
            img = batch['image']
            img = img.to(self.device)

            pred = self.model(img).squeeze(1)

            label = batch['label']
            label = label.to(self.device).float()
            loss = self.criterion(pred, label)
            loss_list.append(loss.item())

            probs = torch.sigmoid(pred)
            all_probs.extend(probs.cpu().numpy().tolist())
            all_labels.extend(label.cpu().numpy().tolist())

        val_loss = float(np.mean(loss_list))
        
        all_labels = np.array(all_labels).astype(int)
        all_probs = np.array(all_probs)

        all_preds = (all_probs >= 0.5).astype(int)
        
        val_auc = float(roc_auc_score(all_labels, all_probs))
        val_f1 = float(f1_score(all_labels, all_preds))
        val_acc = float(accuracy_score(all_labels, all_preds))
        val_pres = float(precision_score(all_labels, all_preds))

        return val_loss, val_auc, val_f1, val_acc, val_pres
    
    def fit(self, trial):
        progress = tqdm(range(self.end_epoch))

        history = {
            'train_loss': [],
            'val_loss': [],
            'val_auc': [],
            'val_f1': [],
            'val_acc': [],
            'val_pres': []
        }

        for epoch in progress:
            train_loss = self.train_step()
            self.writer.add_scalar('Train Loss', train_loss, global_step=epoch)
            
            history['train_loss'].append(train_loss)

            progress.set_description(f'Epoch: {epoch}/{self.end_epoch}, Train Loss:{train_loss}')
        
            if (epoch)%10 == 0:
                val_loss, val_auc, val_f1, val_acc, val_pres = self.valid_step(epoch)
                self.writer.add_scalar('Val Loss', val_loss, global_step=epoch)
                self.writer.add_scalar('Val F1', val_f1, global_step=epoch)
                self.writer.add_scalar('Val ACC', val_acc, global_step=epoch)
                self.writer.add_scalar('Val pres', val_pres, global_step=epoch)

                history['val_loss'].append(val_loss)
                history['val_auc'].append(val_auc)
                history['val_f1'].append(val_f1)
                history['val_acc'].append(val_acc)
                history['val_pres'].append(val_pres)

        val_loss, val_auc, val_f1, val_acc, val_pres = self.valid_step(epoch)
        history['val_loss'].append(val_loss)
        history['val_auc'].append(val_auc)
        history['val_f1'].append(val_f1)
        history['val_acc'].append(val_acc)
        history['val_pres'].append(val_pres)

        
        if trial is not None:
            trial.report(val_auc, step=epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if self.save:
            torch.save({'epoch': epoch,
                       'model_state_dict': self.model.state_dict(),
                       'optimizer_state_dict': self.optimizer.state_dict()}, self.save_path)

            print('Model saved!')
            
        self.writer.close()

        return history

        