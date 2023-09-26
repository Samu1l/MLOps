from torch.utils.data import Dataset

import numpy as np


class WineDataset(Dataset):
    def __init__(self, X, y):
        self.X = X
        self.y = y

    def __len__(self):
        return len(self.y)
    
    def __getitem__(self, idx) -> (np.ndarray, int):
        feature = self.X[idx]
        target = self.y[idx]
        return feature, target
