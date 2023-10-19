import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import torch
#from torchsummary import summary

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')

@hydra.main(version_base="1.3", config_path="../configs", config_name="default.yaml")
def train(cfg: DictConfig) -> None:
    
    pl.seed_everything(cfg.seed)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    # Let hydra manage directory outputs
    tensorboard = pl.loggers.TensorBoardLogger(
        ".", "", "", default_hp_metric=False
    )
    csv = pl.loggers.csv_logs.CSVLogger(".", "", "")
    
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    module = hydra.utils.instantiate(
        cfg.module,
        num_classes=datamodule.num_classes,
        in_channels=datamodule.in_channels,
        # Don't instantiate optimizer submodules with hydra, let `configure_optimizers()` do it
        # _recursive_=False,
    )
    #summary(module.network, (3, 512, 512), depth=4)

    callbacks = {key: hydra.utils.instantiate(cb) for key, cb in cfg.callbacks.items()}
    trainer = hydra.utils.instantiate(cfg.trainer)(
        logger=[tensorboard,csv], #tensorboard should be first 
        callbacks=list(callbacks.values())
    )
    
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.ckpt)
    trainer.validate(module, datamodule=datamodule)
    trainer.test(module, datamodule=datamodule, verbose=True)

if __name__ == "__main__":
    train()
