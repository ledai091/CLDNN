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
from  sklearn.model_selection import train_test_split
from torchvision.transforms import v2
import pandas as pd
from sklearn.metrics import confusion_matrix, accuracy_score
from torchvision.transforms import CenterCrop
from sklearn.utils.class_weight import compute_class_weight
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_fscore_support, roc_auc_score
import seaborn as sns
import time
from datetime import timedelta


from model import CLDNN
from dataset import *
from train import train
from evaluate import evaluate

df = pd.read_csv('data.csv')
X, y = df['img_path'], df['label']
X_remainder, X_test, y_remainder, y_test = get_stratified_test_set(X, y)
X_train, X_val, y_train, y_val = train_test_split(X_remainder, y_remainder, test_size=0.15, stratify=y_remainder)
class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
class_weights = torch.tensor(class_weights, dtype=torch.float)

transform_pipeline = v2.Compose([
    v2.Resize((300, 300)),                 
    v2.RandomHorizontalFlip(p=0.5),        
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485], std=[0.229]) 
])

train_dataset = SeqImageDataset(X_train, y_train, transforms=transform_pipeline)
val_dataset = SeqImageDataset(X_val, y_val, transforms=transform_pipeline)
test_dataset = SeqImageDataset(X_test, y_test, transforms=transform_pipeline)
batch_size = 8
train_sampler = BalancedBatchSampler(y_train, batch_size)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=custom_collate_fn)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=True, collate_fn=custom_collate_fn)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, collate_fn=custom_collate_fn)

in_channels = 1
cnn_output_size = 300
lstm_input_size = cnn_output_size
lstm_hidden_size = 200
lstm_num_blocks = 10
lstm_num_cells_per_block = 10
dnn_output_size = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = CLDNN(
    in_channels=in_channels,
    cnn_out_channels=cnn_output_size,
    lstm_input_size=lstm_input_size, 
    lstm_hidden_size=lstm_hidden_size, 
    lstm_num_blocks=lstm_num_blocks,
    lstm_num_cells_per_block=lstm_num_cells_per_block,
    dnn_output_size=dnn_output_size,
    device=device)
optimizer = AdamW(model.parameters(), lr = 0.001)
loss_fn = CrossEntropyLoss(weight=class_weights)
if loss_fn.weight is not None:
    loss_fn.weight = loss_fn.weight.to(device)
epochs = 100

if __name__ == "__main__":
    train_losses, val_losses = train(model, train_loader, val_loader, epochs, optimizer, loss_fn, device)
    class_names = ['0','1']
    results = evaluate(model, data_loader, device, class_names)
    print(f"Accuracy: {results['accuracy']:.4f}")
    print(f"Precision: {results['precision']:.4f}")
    print(f"Recall: {results['recall']:.4f}")
    print(f"F1-score: {results['f1_score']:.4f}")
    print(f"Number of misclassified samples: {len(results['misclassified_indices'])}")