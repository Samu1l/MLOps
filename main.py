import torch
from torch.utils.data import DataLoader

from src.dataset import WineDataset
from src.utils import seed_everything, prepare_data
from src.pipelines import train

import hydra
from omegaconf import DictConfig
from hydra.utils import instantiate



@hydra.main(config_path=".", config_name="config")
def main(cfg: DictConfig):
    #seed_everything()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    X_train, X_test, y_train, y_test, class_weigths = prepare_data()

    if cfg.use_class_weigths:
        class_weigths = torch.tensor(class_weigths, dtype=torch.float).to(device)
    else:
        class_weigths = None

    train_dataset = WineDataset(X_train, y_train)
    test_dataset = WineDataset(X_test, y_test)
    train_dataloader = DataLoader(train_dataset, batch_size=cfg["batch_size"], shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg["batch_size"], shuffle=False)

    model = instantiate(cfg.model)
    print(model)
    model.to(device)

    optimizer = instantiate(cfg.optimizer, params=model.parameters())

    loss_func = instantiate(cfg.loss_func, weight=class_weigths)

    model = train(cfg.epoch, model, train_dataloader, test_dataloader, loss_func, optimizer, device)


if __name__ == "__main__":
    main()