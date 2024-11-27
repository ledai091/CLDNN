import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from typing import List, Tuple
import os
from datetime import timedelta
from copy import deepcopy
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
import time
import seaborn as sns
import json
import gc

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)
    
def custom_collate_fn(batch):
    batch.sort(key=lambda x: x[0].shape[0], reverse=True)
    sequences, labels = zip(*batch)
    max_len = max([s.shape[0] for s in sequences])
    padded_seqs = []
    for seq in sequences:
        seq_len = seq.shape[0]
        if seq_len < max_len:
            last_frame = seq[-1].unsqueeze(0)
            num_repeat = max_len - seq_len
            padding = last_frame.repeat(num_repeat, 1, 1, 1)
            padded = torch.cat([seq, padding], dim=0)
        else:
            padded = seq
        padded_seqs.append(padded)
    try:
        padded_seqs = torch.stack(padded_seqs, dim=0)
    except:
        print('error')
    labels = torch.stack(labels)
    return padded_seqs, labels

class YMufTTrainer:
    def __init__(self, model, train_dataset, val_dataset, test_dataset, batch_size, num_classes, device, lr, folder_name):
        self.model = model
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
        self.test_dataset = test_dataset
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.device = device
        self.lr = lr
        self.folder_name = folder_name
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=1e-4)
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        self.loss_fn = nn.CrossEntropyLoss()
        os.makedirs(f'results/{self.folder_name}', exist_ok=True)
        os.makedirs(f'weight/{self.folder_name}', exist_ok=True)
    def stat_species(self, dataset):
        tke = [[] for _ in range(self.num_classes)]
        for idx, (_, label) in enumerate(dataset):
            tke[label].append(idx)
        return tke
    
    def arrange_data(self, list_IDs):
        return np.array([len(class_data) for class_data in list_IDs])

    
    def YMufT(self):
        A = deepcopy(self.list_IDs)
        B = deepcopy(self.lst_ratio)
        
        if not np.any(self.B_temp):
            print('End of data, resetting...')
            self.A_temp = deepcopy(self.list_IDs)
            self.B_temp = deepcopy(self.lst_ratio)
            gc.collect()
        
        MC = np.where(self.B_temp > 0, self.B_temp, np.inf).argmin()
        eps = 0.5 * self.B_temp[MC]
        bou1 = np.where(self.B_temp <= self.B_temp[MC] + eps)[0]
        bou2 = self.B_temp[bou1]
        MB = self.B_temp[bou1[np.argmax(bou2)]]
        
        F = []
        for i in range(self.num_classes):
            if self.B_temp[i] > 0:
                nt = int(min(self.B_temp[i], MB))
                np.random.shuffle(self.A_temp[i])
                F.extend(self.A_temp[i][:nt])
                del self.A_temp[i][:nt]
                self.B_temp[i] -= nt
            else:
                np.random.shuffle(A[i])
                nt = int(min(B[i], MB))
                F.extend(A[i][:nt])
        
        return F
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0.0
        for X, y in train_loader:
            X, y = X.to(self.device), y.to(self.device)
            self.optimizer.zero_grad()
            outputs = self.model(X)
            loss = self.loss_fn(outputs, y)
            loss.backward()
            self.optimizer.step()
            total_loss += loss.item()
        return total_loss / len(train_loader)
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0.0
        all_preds = []
        all_targets = []
        with torch.no_grad():
            for X, y in val_loader:
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                loss = self.loss_fn(outputs, y)
                total_loss += loss.item()
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
        
        val_loss = total_loss / len(val_loader)
        val_acc = accuracy_score(all_targets, all_preds)
        return val_loss, val_acc
    
    def train(self, num_loop_eps, total_fold, epochs):
            self.list_IDs = self.stat_species(self.train_dataset)
            self.lst_ratio = self.arrange_data(self.list_IDs)
            self.A_temp = deepcopy(self.list_IDs)
            self.B_temp = deepcopy(self.lst_ratio)
            
            val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)
            
            best_val_acc = 0.0
            best_val_loss = float('inf')
            
            start_time = time.time()
            train_losses = []
            val_losses = []
            
            for training_period in range(num_loop_eps, 0, -1):
                for fold in range(total_fold):
                    print(f'Training period: {num_loop_eps - training_period + 1}, fold: {fold + 1}')
                    
                    F = self.YMufT()
                    fold_dataset = torch.utils.data.Subset(self.train_dataset, F)
                    fold_loader = DataLoader(fold_dataset, batch_size=self.batch_size, shuffle=True, collate_fn=custom_collate_fn)
                    
                    for epoch in range(epochs):
                        epoch_start_time = time.time()
                        train_loss = self.train_epoch(fold_loader)
                        val_loss, val_acc = self.validate(val_loader)
                        
                        train_losses.append(train_loss)
                        val_losses.append(val_loss)
                        
                        self.scheduler.step(val_loss)
                        
                        epoch_end_time = time.time()
                        epoch_duration = epoch_end_time - epoch_start_time
                        epochs_left = epochs * total_fold * num_loop_eps - (epoch + 1 + epochs * (fold + total_fold * (num_loop_eps - training_period)))
                        eta_seconds = epochs_left * epoch_duration
                        eta = str(timedelta(seconds=int(eta_seconds)))
                        print(f'Epoch: {epoch + 1:3d}/{epochs:<3d} | '
                            f'Train Loss: {train_loss:.5f} | '
                            f'Val Loss: {val_loss:.5f} | '
                            f'Val Accuracy: {val_acc:.5f} | '
                            f'LR: {self.optimizer.param_groups[0]["lr"]:.2e} | '
                            f'Epoch Time: {epoch_duration:<7.2f}s | '
                            f'ETA: {eta:<8}')
                        
                        if val_acc > best_val_acc or (val_acc == best_val_acc and val_loss < best_val_loss):
                            best_val_acc = val_acc
                            best_val_loss = val_loss
                            torch.save(self.model.state_dict(), f'weight/{self.folder_name}/best.pt')
                            print(f'Model improved: Val Acc: {best_val_acc:.5f}, Val Loss: {best_val_loss:.5f}')
                    
                    del fold_dataset, fold_loader
                    gc.collect()
            
            total_time = time.time() - start_time
            print(f'Total training time: {str(timedelta(seconds=int(total_time)))}')
            
            # Save the loss chart
            plt.figure(figsize=(10, 6))
            plt.plot(range(1, len(train_losses)+1), train_losses, 'b-', label='Training Loss')
            plt.plot(range(1, len(val_losses)+1), val_losses, 'r-', label='Validation Loss')
            plt.xlabel('Steps')
            plt.ylabel('Loss')
            plt.title('Training and Validation Loss')
            plt.legend()
            plt.tight_layout()
            
            plt.savefig(f'results/{self.folder_name}/loss_chart.jpg')
            print("Loss chart saved")
            print("Training completed")

    
    def test(self):
        self.model.load_state_dict(torch.load(f'weight/{self.folder_name}/best.pt'))
        test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        
        self.model.eval()
        all_preds = []
        all_targets = []
        misclassified_indices = []
        
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(test_loader):
                X, y = X.to(self.device), y.to(self.device)
                outputs = self.model(X)
                _, predicted = torch.max(outputs, 1)
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                start_idx = batch_idx * test_loader.batch_size
                batch_misclassified = (predicted != y).nonzero(as_tuple=True)[0]
                misclassified_indices.extend(start_idx + batch_misclassified.cpu().numpy())
        accuracy = accuracy_score(all_targets, all_preds)
        cm = confusion_matrix(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, average='weighted')
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=[0,1], yticklabels=[0,1])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'results/{self.folder_name}/cfm.jpg')
        plt.close()
        
        results = {
            'accuracy': accuracy,
            'confusion_matrix': cm.tolist(), 
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'misclassified_indices': misclassified_indices
        }
        
        with open(f'results/{self.folder_name}/evaluation_metrics.json', 'w') as json_file:
            json.dump(results, json_file, indent=4, cls=NumpyEncoder)
        return results