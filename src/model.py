import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, 7),
            nn.ReLU(),
            nn.Linear(7, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        preds = self.model(features)
        return preds
