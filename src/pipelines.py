from pathlib import Path
from typing import Callable, List

import torch
import torch.nn as nn
from matplotlib import pyplot as plt
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader
from tqdm import tqdm


def inference(model: nn.Module, dataloder: DataLoader, device) -> List[int]:
    preds_all = []
    model.eval()
    with torch.no_grad():
        for features, _ in dataloder:
            preds = model(features.to(torch.float32).to(device))
            preds_all.extend(preds.argmax(-1).tolist())
    return preds_all


def train(
    epoch: int,
    model: nn.Module,
    train_dataloder: DataLoader,
    test_dataloder: DataLoader,
    loss_func: Callable,
    optimizer: torch.optim,
    device: str,
    path_save_best_model: Path,
) -> nn.Module:
    best_accuracy = 0
    train_losses = []
    test_losses = []
    for _ in tqdm(range(epoch), desc="Epoch"):
        model.train()
        for features, targets in train_dataloder:
            optimizer.zero_grad()
            targets = targets.long().to(device)
            preds = model(features.to(torch.float32).to(device))
            loss = loss_func(preds, targets)
            train_losses.append(loss.item())
            loss.backward()
            optimizer.step()

        test_preds = []
        test_targets = []
        model.eval()
        with torch.no_grad():
            for features, targets in test_dataloder:
                targets = targets.long().to(device)
                preds = model(features.to(torch.float32).to(device))
                test_preds.extend(preds.argmax(-1).tolist())
                test_targets.extend(targets.tolist())
                loss = loss_func(preds, targets)
                test_losses.append(loss.item())

        tmp_metrics = classification_report(test_targets, test_preds, zero_division=0, output_dict=True)
        if tmp_metrics["accuracy"] > best_accuracy:
            best_accuracy = tmp_metrics["accuracy"]
            torch.save(model, path_save_best_model)

    plt.plot(train_losses)
    plt.title("Train losses")
    plt.savefig(path_save_best_model.parent / "Train losses.png")
    plt.clf()

    plt.plot(test_losses)
    plt.title("Test losses")
    plt.savefig(path_save_best_model.parent / "Test losses.png")
    return model
