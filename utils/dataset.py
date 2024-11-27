import torch
from torch.utils.data import Dataset, DataLoader
from glob import glob
import re
from torchvision.io import read_image, ImageReadMode
import os
from typing import List
from torchvision.transforms import v2
import utils.load_data as load_data
from .batch_sampler import BalancedBatchSampler
from .custom_collate_fn import custom_collate_fn

class SeqImageDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y
        self.transforms = v2.Compose([
        v2.ToDtype(torch.float32, scale=True),
        ])
        
        self.image_paths = self._load_image_paths()
        print(f'Loaded {len(self.image_paths)} sequences')
    
    def _load_image_paths(self) -> List[List[str]]:
        image_paths = []
        for dir_path in self.X:
            
            if not os.path.exists(dir_path):
                raise FileNotFoundError(f"Directory not found: {dir_path}")
            
            sequence_paths = sorted(
                glob(os.path.join(dir_path, '*.jpg')), 
                key=self.custom_sort_key  
            )
            
            if not sequence_paths:
                raise ValueError(f"No images found in directory: {dir_path}")
                
            image_paths.append(sequence_paths)
        
        return image_paths
    
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
            read_image(img_path, mode=ImageReadMode.RGB)
            for img_path in self.image_paths[idx]
        ]
        images = torch.stack(image_sequence, dim=0)
        label = torch.tensor(self.y.iloc[idx])
        if self.transforms:
            images = self.transforms(images)
        return images, label
    
def main(args):
    X_train, X_val, X_test, y_train, y_val, y_test = load_data.main(args)
    train_dataset = SeqImageDataset(X_train, y_train)
    val_dataset = SeqImageDataset(X_val, y_val)
    test_dataset = SeqImageDataset(X_test, y_test)
    
    train_sampler = BalancedBatchSampler(y_train, batch_size=args.batch_size, class_0_weight=2)
    train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=custom_collate_fn)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    test_loader = DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=custom_collate_fn)
    
    if args.class_weight:
        class_weight=load_data.LoadData().is_class_weight(y_train)
        return train_loader, val_loader, test_loader, class_weight
    
    return train_loader, val_loader, test_loader