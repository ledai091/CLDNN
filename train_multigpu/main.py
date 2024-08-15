import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from typing import List, Tuple, Optional
from torch import Tensor
import os
import cv2
import matplotlib.pyplot as plt
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision
from glob import glob
from torchvision.io import read_image
from torchvision.io import ImageReadMode
import re
from torch.optim import AdamW
from torch.nn import CrossEntropyLoss
from sklearn.model_selection import train_test_split
from torchvision.transforms import v2
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision.transforms import CenterCrop
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import seaborn as sns

from CLDNN import CLDNN
from dataset import SeqImageDataset, custom_collate_fn, get_stratified_test_set, BalancedBatchSampler

torch.backends.cudnn.benchmark = True

def ddp_setup(rank, world_size):
    print(f"Setting up DDP for rank {rank} with world size {world_size}")
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12355"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def plot_and_save_confusion_matrix(cm, classes, file_name='confusion_matrix.png'):
    plt.figure(figsize=(10, 7))
    sns.heatmap(cm, annot=True, fmt="d", cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.title('Confusion Matrix')
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(file_name)
    plt.close()

def evaluate(model, data_loader, device, classes):
    model.eval()
    all_preds = []
    all_targets = []
    misclassified_indices = []
    with torch.no_grad():
        for batch_idx, (X, y) in enumerate(data_loader):
            X = X.to(device)
            y = y.to(device)
            outputs = model(X)
            _, predicted = torch.max(outputs, 1)
            all_preds.extend(predicted.cpu().numpy())
            all_targets.extend(y.cpu().numpy())
            start_idx = batch_idx * data_loader.batch_size
            batch_misclassified = (predicted != y).nonzero(as_tuple=True)[0]
            misclassified_indices.extend(start_idx + batch_misclassified.cpu().numpy())
    acc = accuracy_score(all_targets, all_preds)
    cm = confusion_matrix(all_targets, all_preds)
    
    plot_and_save_confusion_matrix(cm, classes, file_name="confusion_matrix.png")
    
    return acc, cm, misclassified_indices

class Trainer:
    def __init__(self, model, train_loader, val_loader, optimizer, gpu_id, criterion):
        if torch.cuda.is_available():
            model = model.to(gpu_id)
            self.model = DDP(model, device_ids=[gpu_id], find_unused_parameters=True)
        else:
            self.model = DDP(model, find_unused_parameters=True)  
        
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.optimizer = optimizer
        self.gpu_id = gpu_id
        self.criterion = criterion
        
    def _run_batch(self, source, targets):
        self.optimizer.zero_grad()
        output = self.model(source)
        loss = self.criterion(output, targets)
        loss.backward()
        self.optimizer.step()
        return loss.item()
    
    def _run_epoch(self, epoch):
        print(f'[GPU:{self.gpu_id} Starting epoch {epoch+1}]')
        batch_size = len(next(iter(self.train_loader))[0])
        # print(f'[GPU:{self.gpu_id}] Epoch {epoch} | Batchsize: {batch_size} | Steps: {len(self.train_loader)}')
        train_losses = 0.0
        self.model.train()
        for source, targets in self.train_loader:
            source = source.to(self.gpu_id)
            targets = targets.to(self.gpu_id)
            loss = self._run_batch(source, targets)
            train_losses += loss
        
        self.model.eval()
        val_losses = 0.0
        with torch.no_grad():
            for source, targets in self.val_loader:
                source = source.to(self.gpu_id)
                targets = targets.to(self.gpu_id)
                outputs = self.model(source)
                loss = self.criterion(outputs, targets)
                val_losses += loss.item()
                
        return train_losses / len(self.train_loader), val_losses / len(self.val_loader)

    def train(self, max_epochs):
        train_losses = []
        val_losses = []
        for epoch in range(max_epochs):
            train_loss, val_loss = self._run_epoch(epoch)
            train_losses.append(train_loss)
            val_losses.append(val_loss)
            
            print(f'[GPU: {self.gpu_id}] Epoch: {epoch+1}: Training Loss: {train_loss} | Validation Loss: {val_loss}')
        
        plt.figure(figsize=(10, 5))
        plt.plot(range(1, max_epochs + 1), train_losses, label='Training Loss')
        plt.plot(range(1,  max_epochs + 1), val_losses, label='Validation Loss')
        plt.xlabel('Epochs')
        plt.ylabel('Loss')
        plt.title('Training and Validation Loss Over Epochs')
        plt.legend()
        plt.savefig('loss_chart.png')
        return train_losses, val_losses
    def save_model(self, file_name):
    # Save the model's state_dict only from the main GPU
        if self.gpu_id == 0:
            torch.save(self.model.module.state_dict(), file_name)
            print(f'Model saved to {file_name}')
            
def main(rank: int, world_size: int, total_epochs: int, batch_size: int):
    ddp_setup(rank, world_size)
    
    df = pd.read_csv('data.csv')
    X, y = df['img_path'], df['label']
    X_remainder, X_test, y_remainder, y_test = get_stratified_test_set(X, y)
    X_train, X_val, y_train, y_val = train_test_split(X_remainder, y_remainder, test_size=0.15, stratify=y_remainder)

    transform_pipeline = v2.Compose([
        v2.Resize((224, 224)),                 
        v2.RandomHorizontalFlip(p=0.5),        
        v2.ToDtype(torch.float32, scale=True),
        v2.Normalize(mean=[0.485], std=[0.229]) 
    ])

    train_dataset = SeqImageDataset(X_train, y_train, transforms=transform_pipeline)
    val_dataset = SeqImageDataset(X_val, y_val, transforms=transform_pipeline)
    test_dataset = SeqImageDataset(X_test, y_test, transforms=transform_pipeline)

    train_sampler = BalancedBatchSampler(y_train, batch_size, num_replicas=world_size, rank=rank)

    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=custom_collate_fn)

    val_loader = DataLoader(val_dataset, batch_size=batch_size, sampler=DistributedSampler(val_dataset), collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, sampler=DistributedSampler(test_dataset), collate_fn=custom_collate_fn)
    
    in_channels = 1
    cnn_output_size = 200
    lstm_input_size = cnn_output_size
    lstm_hidden_size = 100
    lstm_num_blocks = 10
    lstm_num_cells_per_block = 10
    dnn_output_size = 2
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = CLDNN(
        in_channels=in_channels,
        cnn_out_channels=cnn_output_size,
        lstm_input_size=lstm_input_size, 
        lstm_hidden_size=lstm_hidden_size, 
        lstm_num_blocks=lstm_num_blocks,
        lstm_num_cells_per_block=lstm_num_cells_per_block,
        dnn_output_size=dnn_output_size,
        device=device).to(device)
    model.load_state_dict(torch.load('weight/model.pt'))
    optimizer = AdamW(model.parameters(), lr = 0.001)
    criterion = CrossEntropyLoss()
    
    trainer = Trainer(model, train_loader, val_loader, optimizer, rank, criterion)
    trainer.train(total_epochs)
    if rank == 0:
        trainer.save_model('weight/model_parallel.pt')
    classes = ['Class 1', 'Class 2']  # Update with your class names
    accuracy, _, _ = evaluate(model, test_loader, device, classes)
    print('Accuracy: ', accuracy)
    destroy_process_group()

if __name__ == "__main__":
    world_size = 2
    total_epochs = 100
    batch_size = 8
    
    mp.spawn(main, args=(world_size, total_epochs, batch_size), nprocs=world_size, join=True)