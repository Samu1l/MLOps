from pathlib import Path

import hydra
import mlflow
import pandas as pd
import torch
from omegaconf import DictConfig
from sklearn.metrics import classification_report
from torch.utils.data import DataLoader

from src.dataset import WineDataset


@hydra.main(config_path="conf", config_name="infer", version_base="1.2")
def main(cfg: DictConfig):
    mlflow.set_tracking_uri(uri=cfg.mlflow_uri)

    df_infer = pd.read_csv(cfg.infer_data_path)
    X_infer = df_infer[cfg.feature_name_fields].to_numpy()
    if cfg.is_target_in_data:
        y_infer = df_infer[cfg.target_name_field].to_numpy()
    else:
        y_infer = None

    infer_dataset = WineDataset(X_infer, y_infer)
    infer_dataloader = DataLoader(infer_dataset, batch_size=cfg.batch_size, shuffle=False)

    onnx_model = mlflow.pyfunc.load_model(cfg.model_uri)

    preds_all = []
    for features, _ in infer_dataloader:
        preds = onnx_model.predict(features.to(torch.float32).numpy())["output"]
        preds_all.extend(preds.argmax(-1).tolist())

    df_infer["prediction"] = preds_all

    if cfg.is_target_in_data:
        print(
            classification_report(
                df_infer[cfg.target_name_field], df_infer["prediction"], zero_division=0, output_dict=False
            )
        )

    path_to_save_results = Path(__file__).parent / cfg.save_predictions_path
    path_to_save_results.parent.mkdir(parents=True, exist_ok=True)
    df_infer.to_csv(path_to_save_results, index=False)


if __name__ == "__main__":
    main()
