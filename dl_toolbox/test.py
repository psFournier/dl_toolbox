import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import torch

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')

@hydra.main(version_base="1.3", config_path="../configs", config_name="default_pred.yaml")
def main(cfg: DictConfig) -> None:
    
    pl.seed_everything(cfg.seed)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    module = hydra.utils.instantiate(
        cfg.module,
        num_classes=datamodule.num_classes,
        in_channels=datamodule.in_channels,
    )
    callbacks = {key: hydra.utils.instantiate(cb) for key, cb in cfg.callbacks.items()}
    
    trainer = hydra.utils.instantiate(cfg.trainer)(
        callbacks=list(callbacks.values())
    )
    trainer.test(module, datamodule=datamodule, ckpt_path=cfg.ckpt, verbose=True)

if __name__ == "__main__":
    main()
