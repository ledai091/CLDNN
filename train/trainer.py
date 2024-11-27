import time
import json
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import timedelta
from typing import List, Tuple
from model import cldnn
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from utils import dataset, FocalLoss

class NumpyEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, np.integer):
            return int(obj)
        elif isinstance(obj, np.floating):
            return float(obj)
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        return super(NumpyEncoder, self).default(obj)

class Trainer:
    def __init__(
        self,
        args
    ):
        self.device = args.device
        
        data = dataset.main(args)
        train_loader, val_loader, test_loader = data[:3]
        class_weight = data[3] if len(data)==4 else None
        # self.loss_fn = nn.CrossEntropyLoss(weight=class_weight.to(self.device) if class_weight is not None else None)
        self.loss_fn = FocalLoss()
        
        self.model = cldnn.main(args)

        self.train_loader = train_loader
        self.val_loader = val_loader
        self.test_loader = test_loader
        
        self.optimizer = AdamW(self.model.parameters(), lr=1e-3)
        self.scheduler = ReduceLROnPlateau(self.optimizer, mode='min', factor=0.1, patience=10)
        
        
        
        
        self.folder_name = args.folder_name
        
        self.train_losses: List[float] = []
        self.val_losses: List[float] = []
        
    def train(self, epochs: int) -> Tuple[List[float], List[float]]:
        self.model.to(self.device)
        start_time = time.time()
        
        for epoch in range(epochs):
            epoch_start_time = time.time()
            
            train_loss = self._train_epoch()
            self.train_losses.append(train_loss)
            
            val_loss = self._validate_epoch()
            self.val_losses.append(val_loss)
            
            self.scheduler.step(val_loss)
            
            epoch_duration = time.time() - epoch_start_time
            eta = self._calculate_eta(epoch, epochs, epoch_duration)
            
            self._print_progress(epoch + 1, epochs, train_loss, val_loss, 
                               self.optimizer.param_groups[0]["lr"], 
                               epoch_duration, eta)
        
        total_time = time.time() - start_time
        print(f'Total training time: {str(timedelta(seconds=int(total_time)))}')
    
        self._save_loss_chart(epochs)
        self._save_model()
        
        return self.train_losses, self.val_losses
    
    def _train_epoch(self) -> float:
        self.model.train()
        total_loss = 0.0
        
        for X, y in self.train_loader:
            X = X.to(self.device)
            y = y.to(self.device)
            
            y_hat = self.model(X)
            loss = self.loss_fn(y_hat, y)
            total_loss += loss.item()
            
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()
            
        return total_loss / len(self.train_loader)
    
    def _validate_epoch(self) -> float:
        self.model.eval()
        total_loss = 0.0
        
        with torch.no_grad():
            for X, y in self.val_loader:
                X = X.to(self.device)
                y = y.to(self.device)
                
                y_hat = self.model(X)
                loss = self.loss_fn(y_hat, y)
                total_loss += loss.item()
                
        return total_loss / len(self.val_loader)
    
    def evaluate(self) -> None:
        self.model.eval()
        all_preds = []
        all_targets = []
        misclassified_indices = []
        
        with torch.no_grad():
            for batch_idx, (X, y) in enumerate(self.test_loader):
                X = X.to(self.device)
                y = y.to(self.device)
                
                outputs = self.model(X)
                _, predicted = torch.max(outputs, 1)
                
                all_preds.extend(predicted.cpu().numpy())
                all_targets.extend(y.cpu().numpy())
                
                start_idx = batch_idx * self.test_loader.batch_size
                batch_misclassified = (predicted != y).nonzero(as_tuple=True)[0]
                misclassified_indices.extend(start_idx + batch_misclassified.cpu().numpy())
        
        self._save_evaluation_metrics(all_preds, all_targets, misclassified_indices)
    
    def _save_evaluation_metrics(self, all_preds: np.ndarray, all_targets: np.ndarray, 
                               misclassified_indices: List[int]) -> None:
        all_preds = np.array(all_preds)
        all_targets = np.array(all_targets)
    
        acc = accuracy_score(all_targets, all_preds)
        cm = confusion_matrix(all_targets, all_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(all_targets, all_preds, 
                                                                 average='weighted')
    
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=[0, 1], yticklabels=[0, 1])
        plt.title('Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        plt.savefig(f'results/{self.folder_name}/cfm.jpg')
        plt.close()
    
        results = {
            'accuracy': acc,
            'confusion_matrix': cm.tolist(),
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'misclassified_indices': misclassified_indices
        }
        
        with open(f'results/{self.folder_name}/evaluation_metrics.json', 'w') as json_file:
            json.dump(results, json_file, indent=4, cls=NumpyEncoder)
    
    def _calculate_eta(self, current_epoch: int, total_epochs: int, 
                      epoch_duration: float) -> str:
        epochs_left = total_epochs - (current_epoch + 1)
        eta_seconds = epochs_left * epoch_duration
        return str(timedelta(seconds=int(eta_seconds)))
    
    def _print_progress(self, epoch: int, total_epochs: int, train_loss: float, 
                       val_loss: float, lr: float, epoch_duration: float, eta: str) -> None:
        print(f'Epoch: {epoch:3d}/{total_epochs:<3d} | '
              f'Train Loss: {train_loss:.20f} | '
              f'Val Loss: {val_loss:.20f} | '
              f'LR: {lr:.2e} | '
              f'Epoch Time: {epoch_duration:<7.2f}s | '
              f'ETA: {eta:<8}')
    
    def _save_loss_chart(self, epochs: int) -> None:
        plt.figure(figsize=(10, 6))
        plt.plot(range(1, epochs+1), self.train_losses, 'b-', label='Training Loss')
        plt.plot(range(1, epochs+1), self.val_losses, 'r-', label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f'results/{self.folder_name}/loss_chart.jpg')
        plt.close()
        print("Loss chart saved")
    
    def _save_model(self) -> None:
        torch.save(self.model.state_dict(), f'weight/{self.folder_name}/last.pt')
        print("Model saved")