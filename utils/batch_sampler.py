from torch.utils.data import Sampler
import numpy as np

class BalancedBatchSampler(Sampler):
    def __init__(self, labels, batch_size, class_0_weight=3):
        self.labels = labels
        self.batch_size = batch_size
        self.class_0_weight = class_0_weight
        self.idx_0 = np.where(self.labels == 0)[0]
        self.idx_1 = np.where(self.labels == 1)[0]
        self.num_0 = len(self.idx_0)
        self.num_1 = len(self.idx_1)
        self.start_0 = 0
        self.start_1 = 0
        
    def __len__(self):
        return (self.num_1 + self.batch_size//(self.class_0_weight + 1) - 1) // (self.batch_size // (self.class_0_weight + 1))
    
    def __iter__(self):
        np.random.shuffle(self.idx_0)
        np.random.shuffle(self.idx_1)
        
        max_batches = len(self)
    
        for _ in range(max_batches):
            batch = []
            
            start_0 = self.start_0
            end_0 = start_0 + (self.batch_size * self.class_0_weight) // (self.class_0_weight + 1)
            if end_0 > self.num_0:
                batch.extend(self.idx_0[start_0: self.num_0])
                lack_0 = end_0 - self.num_0
                batch.extend(self.idx_0[:lack_0])
                self.start_0 = lack_0
            else:
                batch.extend(self.idx_0[start_0:end_0])
                self.start_0 = end_0 % self.num_0
            
            start_1 = self.start_1
            end_1 = start_1 + self.batch_size // (self.class_0_weight + 1)
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