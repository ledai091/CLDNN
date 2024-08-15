import os
import re
from glob import glob
import torch
from torch.utils.data import Dataset, Sampler
from torchvision.io import read_image, ImageReadMode
import numpy as np

class SeqImageDataset(Dataset):
    def __init__(self, X, y, transforms=None):
        self.X = X
        self.y = y
        self.transforms = transforms
        self.image_paths = [
            sorted(glob(os.path.join(dir_path, '*.jpg')), key=self.custom_sort_key)
            for dir_path in self.X
        ]
        print(f'Loaded {len(self.image_paths)} sequences')
    
    @staticmethod
    def custom_sort_key(filename):
        match = re.search(r'frame_(\d+)', filename)
        if match:
            return int(match.group(1))
        return filename  

    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        image_sequence = [
            read_image(img_path, mode=ImageReadMode.GRAY)
            for img_path in self.image_paths[idx]
        ]
        images = torch.stack(image_sequence, dim=0)
        label = torch.tensor(self.y.iloc[idx])
        if self.transforms:
            images = self.transforms(images)
        return images, label
    
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

def get_stratified_test_set(X, y, n_samples_per_class=10):
    indices_class_0 = np.where(y == 0)[0]
    indices_class_1 = np.where(y == 1)[0]

    test_indices_class_0 = np.random.choice(indices_class_0, n_samples_per_class, replace=False)
    test_indices_class_1 = np.random.choice(indices_class_1, n_samples_per_class, replace=False)

    test_indices = np.concatenate([test_indices_class_0, test_indices_class_1])

    mask = np.zeros(len(y), dtype=bool)
    mask[test_indices] = True

    X_test, X_remainder = X[mask], X[~mask]
    y_test, y_remainder = y[mask], y[~mask]

    return X_remainder, X_test, y_remainder, y_test


class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size):
        self.labels = labels
        self.batch_size = batch_size
        self.idx_0 = np.where(self.labels == 0)[0]
        self.idx_1 = np.where(self.labels == 1)[0]
        self.num_0 = len(self.idx_0)
        self.num_1 = len(self.idx_1)
        self.start_0 = 0
        self.start_1 = 0
        
    def __len__(self):
        return (self.num_1 + self.batch_size//2 - 1) // (self.batch_size // 2)
    
    def __iter__(self):
        
        np.random.shuffle(self.idx_0)
        np.random.shuffle(self.idx_1)
        
        max_batches = len(self)
    
        for i in range(max_batches):
            batch = []
            
            start_0 = self.start_0
            end_0 = start_0 + self.batch_size // 2
            if end_0 > self.num_0:
                batch.extend(self.idx_0[start_0: self.num_0])
                lack_0 = end_0 - self.num_0
                batch.extend(self.idx_0[:lack_0])
                self.start_0 = lack_0
            else:
                batch.extend(self.idx_0[start_0:end_0])
                self.start_0 = end_0 % self.num_0
            
            start_1 = self.start_1
            end_1 = start_1 + self.batch_size // 2
            if end_1 > self.num_1:
                batch.extend(self.idx_1[start_1: self.num_1])
                lack_1 = end_1 - self.num_1
                batch.extend(self.idx_1[:lack_1])
                self.start_1 = lack_1
            else:
                batch.extend(self.idx_1[start_1:end_1])
                self.start_1 = end_1 % self.num_1
                
            np.random.shuffle(batch)
            yield batch