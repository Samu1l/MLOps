import torch
import torch.nn as nn


class Classifier(nn.Module):
    def __init__(self, num_features: int, num_classes: int, num_inter_features: int = 7):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(num_features, num_inter_features),
            nn.ReLU(),
            nn.Linear(num_inter_features, num_classes),
        )

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        preds = self.model(features)
        return preds
