import hydra
import lightning as L
import torch
from omegaconf import DictConfig, OmegaConf

# import os
# os.environ["HYDRA_FULL_ERROR"] = "1"

@hydra.main(config_path="configs", config_name="config", version_base="1.3")
def train(cfg: DictConfig) -> None:
    print(OmegaConf.to_yaml(cfg))

    if cfg.get("seed") is not None:
        L.seed_everything(cfg.seed, workers=True)
    
    # Handle PyTorch precision using config parameters
    if cfg.get("matmul_precision") is not None:
        torch.set_float32_matmul_precision(cfg.matmul_precision)

    print(f"Instantiating DataModule: <{cfg.data._target_}>")
    datamodule = hydra.utils.instantiate(cfg.data)

    print(f"Instantiating LightningModule: <{cfg.model._target_}>")
    model = hydra.utils.instantiate(cfg.model)

    print(f"Instantiating Trainer: <{cfg.trainer._target_}>")
    trainer = hydra.utils.instantiate(cfg.trainer)

    print("Starting training pipeline...")
    trainer.fit(model, datamodule=datamodule)


if __name__ == "__main__":
    train()