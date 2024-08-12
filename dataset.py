import torch
import torch.nn as nn
import numpy as np
from torch import Tensor
import os
import cv2
from torch.utils.data import Dataset, DataLoader, Sampler
import torchvision
from glob import glob
from torchvision.io import read_image
from torchvision.io import ImageReadMode
import re
from torchvision.transforms import v2

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
        # Trích xuất số frame từ tên file
        match = re.search(r'frame_(\d+)', filename)
        if match:
            return int(match.group(1))
        return filename  # Trả về nguyên tên file nếu không tìm thấy số frame

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
    padded_seqs = torch.stack(padded_seqs, dim=0)
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
    def __init__(self, labels, batch_size, num_replicas=None, rank=None):
        self.labels = labels
        self.batch_size = batch_size
        self.indices_0 = np.where(labels == 0)[0]
        self.indices_1 = np.where(labels == 1)[0]
        
        if num_replicas is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = dist.get_world_size()
        if rank is None:
            if not dist.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = dist.get_rank()

        self.num_replicas = num_replicas
        self.rank = rank
        self.num_samples_per_replica = len(self.indices_1) // (self.batch_size // 2) // num_replicas
        self.total_samples = self.num_samples_per_replica * num_replicas
        self.used_1 = 0

    def __iter__(self):
        np.random.shuffle(self.indices_0)
        np.random.shuffle(self.indices_1)
        
        # Determine the starting index for this replica
        start = self.rank * self.num_samples_per_replica * (self.batch_size // 2)
        end = start + self.num_samples_per_replica * (self.batch_size // 2)
        
        for i in range(self.num_samples_per_replica):
            batch = []
            batch.extend(self.indices_0[start + i * (self.batch_size // 2):start + (i + 1) * (self.batch_size // 2)])
            batch.extend(self.indices_1[self.used_1:self.used_1 + (self.batch_size // 2)])
            self.used_1 = (self.used_1 + (self.batch_size // 2)) % len(self.indices_1)
            np.random.shuffle(batch)
            yield batch

    def __len__(self):
        return self.num_samples_per_replica
