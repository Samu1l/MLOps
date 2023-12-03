from pathlib import Path
from typing import Callable, List

import mlflow
import onnx
import torch
import torch.nn as nn
from mlflow.models import infer_signature
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
    model_artifact_path: str,
    registered_model_name: str,
):
    best_accuracy = 0
    train_losses = []
    test_losses = []
    # with mlflow.start_run():
    for _ in tqdm(range(epoch), desc="Epoch"):
        model.train()
        for features, targets in train_dataloder:
            optimizer.zero_grad()
            targets = targets.long().to(device)
            preds = model(features.to(torch.float32).to(device))
            loss = loss_func(preds, targets)
            train_losses.append(loss.item())
            mlflow.log_metric("train_loss", loss.item())
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
                mlflow.log_metric("test_loss", loss.item())
                test_losses.append(loss.item())

        tmp_metrics = classification_report(test_targets, test_preds, zero_division=0, output_dict=True)
        mlflow.log_metric("test_accuracy", tmp_metrics["accuracy"])
        mlflow.log_metric("test_macro_precision", tmp_metrics["macro avg"]["precision"])
        mlflow.log_metric("test_macro_recall", tmp_metrics["macro avg"]["recall"])
        if tmp_metrics["accuracy"] > best_accuracy:
            best_accuracy = tmp_metrics["accuracy"]
            torch.save(model, str(path_save_best_model).rsplit(".", 1)[0] + ".pt")
            dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
            torch.onnx.export(
                model,
                features.to(torch.float32).to(device),
                path_save_best_model,
                input_names=["input"],
                output_names=["output"],
                dynamic_axes=dynamic_axes,
            )
    onnx_model = onnx.load_model(path_save_best_model)
    signature = infer_signature(features.to(torch.float32).numpy(), preds.detach().numpy())
    mlflow.onnx.log_model(
        onnx_model, model_artifact_path, registered_model_name=registered_model_name, signature=signature
    )
