import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from tqdm import tqdm

from typing import Callable
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report



def train(
          epoch: int,
          model: nn.Module, 
          train_dataloder: DataLoader, 
          test_dataloder: DataLoader,
          loss_func: Callable,
          optimizer: torch.optim,
          device: str
):
    train_losses = []
    test_losses = []
    for epoch_idx in range(epoch):
        print("Epoch:", epoch_idx)
        model.train()
        for features, targets in tqdm(train_dataloder, desc="train"):
            optimizer.zero_grad()
            targets = torch.tensor(targets).long().to(device)
            preds = model(features.to(torch.float32).to(device))
            loss = loss_func(preds, targets)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        test_preds = []
        test_targets = []
        model.eval()
        with torch.no_grad():
            for features, targets in tqdm(test_dataloder, desc="test"):
                targets = torch.tensor(targets).long().to(device)
                preds = model(features.to(torch.float32).to(device))
                test_preds.extend(preds.argmax(-1))
                test_targets.extend(targets)
                loss = loss_func(preds, targets)
                test_losses.append(loss.item())

        print(classification_report(test_targets, test_preds))

    plt.plot(train_losses)
    plt.title("Train losses")
    plt.savefig(f"/Users/s.burovin/Documents/HW/11sem/MLOps/Train losses.png")
    plt.clf()

    plt.plot(test_losses)
    plt.title("Test losses")
    plt.savefig(f"/Users/s.burovin/Documents/HW/11sem/MLOps/Test losses.png")
    return model
