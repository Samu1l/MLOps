from pathlib import Path

import hydra
import torch
from dvc.api import DVCFileSystem
from omegaconf import DictConfig


@hydra.main(config_path="conf", config_name="export", version_base="1.2")
def main(cfg: DictConfig):
    dvc_fs = DVCFileSystem(Path(__file__).parent)
    if not (Path(__file__).parent / cfg.saved_best_model_path).exists():
        dvc_fs.get_file(cfg.saved_best_model_path, cfg.saved_best_model_path)

    if not (Path(__file__).parent / cfg.saved_scaler_path).exists():
        dvc_fs.get_file(cfg.saved_scaler_path, cfg.saved_scaler_path)

    with open(cfg.saved_best_model_path, "rb") as model_file:
        model = torch.load(model_file)

    example_input = torch.randn((3, 13), dtype=torch.float32)
    dynamic_axes = {"input": {0: "batch_size"}, "output": {0: "batch_size"}}
    torch.onnx.export(
        model,
        example_input,
        cfg.save_best_model_onnx_format,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes=dynamic_axes,
    )


if __name__ == "__main__":
    main()
