import torch
import random
import numpy as np

from sklearn.datasets import load_wine
from sklearn.model_selection import train_test_split
from sklearn.utils.class_weight import compute_class_weight


def seed_everything(seed=42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available(): 
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed) 
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

def prepare_data(train_ratio: float = 0.7):
    data = load_wine()
    X_train, X_test, y_train, y_test = train_test_split(data.data, data.target, train_size=train_ratio, stratify=data.target, shuffle=True)
    class_weigths = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
    return X_train, X_test, y_train, y_test, class_weigths
