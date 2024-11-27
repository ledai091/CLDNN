from .augmentation import ImageAugmenter
from .split_data import get_stratified_test_set
import pandas as pd
from sklearn.utils.class_weight import compute_class_weight
import torch
import numpy as np

class LoadData:
    def __init__(self) -> None:
        super(LoadData, self).__init__()
        
    def read_data(self, path):
        df = pd.read_csv(path)
        X, y = df['img_path'], df['label']
        X_remainder, X_test, y_remainder, y_test = get_stratified_test_set(X, y, n_samples_per_class=10)
        X_train, X_val, y_train, y_val = get_stratified_test_set(X_remainder, y_remainder, n_samples_per_class=10)
        return X_train, X_val, X_test, y_train, y_val, y_test
    
    def is_augmentation(self, X_train, y_train):
        IA = ImageAugmenter()
        X_train, y_train = IA.augment_dataset(X_train, y_train)
        return X_train, y_train
    
    def is_del_augmentation(self):
        IA = ImageAugmenter()
        IA.delete_augmentations()
    
    def is_class_weight(self, y_train):
        class_weights = compute_class_weight(class_weight='balanced', classes=np.unique(y_train), y=y_train)
        class_weights = torch.tensor(class_weights, dtype=torch.float)
        return class_weights
    

def main(args):
    load_data = LoadData()
    X_train, X_val, X_test, y_train, y_val, y_test = load_data.read_data('img_resize/img_resize.csv')
    load_data.is_del_augmentation()
    if args.augmentation:
        X_train, y_train = load_data.is_augmentation(X_train, y_train)
    return X_train, X_val, X_test, y_train, y_val, y_test