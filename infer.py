from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader

from src.dataset import WineDataset
from src.pipelines import inference
from src.utils import prepare_data, seed_everything


@hydra.main(config_path=".", config_name="config", version_base="1.2")
def main(cfg: DictConfig):
    seed_everything(cfg.seed)

    device = "cuda" if torch.cuda.is_available() else "cpu"

    _, X_test, _, y_test, _, df_test = prepare_data(return_test_dataframe=True)

    test_dataset = WineDataset(X_test, y_test)
    test_dataloader = DataLoader(
        test_dataset, batch_size=cfg["batch_size"], shuffle=False
    )

    model = torch.load(Path(__file__).parent / cfg.path_save_best_model)
    model.to(device)

    preds_all, _ = inference(model, test_dataloader, device)
    df_test["prediction"] = preds_all

    df_test.to_csv(
        Path(__file__).parent / cfg.path_save_test_predictions_csv, index=False
    )


if __name__ == "__main__":
    main()
