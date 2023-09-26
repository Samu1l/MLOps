import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Dropout1d(p=0.3),
            nn.Linear(num_features, num_classes),
            nn.LeakyReLU()
        )
    
    def forward(self, features: torch.Tensor) -> torch.Tensor:
        preds = self.model(features)
        return preds