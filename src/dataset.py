from typing import Optional

import numpy as np
from numpy.typing import ArrayLike
from sklearn.preprocessing import MinMaxScaler
from torch.utils.data import Dataset


class WineDataset(Dataset):
    def __init__(self, X: ArrayLike, y: Optional[ArrayLike] = None):
        self.X = MinMaxScaler(feature_range=(0, 1)).fit_transform(X)
        self.y = y

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx) -> (np.ndarray, Optional[int]):
        feature = self.X[idx]
        if self.y is not None:
            target = self.y[idx]
        else:
            target = None
        return feature, target
