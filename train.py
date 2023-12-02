from pathlib import Path

import hydra
import numpy as np
import pandas as pd
import torch
from dvc.api import DVCFileSystem
from hydra.utils import instantiate
from omegaconf import DictConfig
from sklearn.utils.class_weight import compute_class_weight
from torch.utils.data import DataLoader

from src.dataset import WineDataset
from src.pipelines import train
from src.utils import seed_everything


@hydra.main(config_path="conf", config_name="train", version_base="1.2")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    dvc_fs = DVCFileSystem(Path(__file__).parent)
    if not (Path(__file__).parent / cfg.train_data_path).exists():
        dvc_fs.get_file(cfg.train_data_path, cfg.train_data_path)
    df_train = pd.read_csv(cfg.train_data_path)
    X_train, y_train = (
        df_train[cfg.feature_name_fields].to_numpy(),
        df_train[cfg.target_name_field].to_numpy(),
    )

    if not (Path(__file__).parent / cfg.test_data_path).exists():
        dvc_fs.get_file(cfg.test_data_path, cfg.test_data_path)
    df_test = pd.read_csv(cfg.test_data_path)
    X_test, y_test = (
        df_test[cfg.feature_name_fields].to_numpy(),
        df_test[cfg.target_name_field].to_numpy(),
    )

    device = "cuda" if torch.cuda.is_available() else "cpu"

    if cfg.use_class_weigths:
        class_weigths = compute_class_weight(class_weight="balanced", classes=np.unique(y_train), y=y_train)
        class_weigths = torch.tensor(class_weigths, dtype=torch.float).to(device)
    else:
        class_weigths = None

    train_dataset = WineDataset(X_train, y_train)
    test_dataset = WineDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)

    model = instantiate(cfg.model)
    model.to(device)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    loss_func = instantiate(cfg.loss_func, weight=class_weigths)

    save_best_model_path = Path(__file__).parent / cfg.save_best_model_path
    save_best_model_path.parent.mkdir(parents=True, exist_ok=True)
    model = train(
        cfg.epoch,
        model,
        train_dataloader,
        test_dataloader,
        loss_func,
        optimizer,
        device,
        save_best_model_path,
    )


if __name__ == "__main__":
    main()
