import logging
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import torch

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')

@hydra.main(version_base="1.3", config_path="../configs", config_name="default_train.yaml")
def train(cfg: DictConfig) -> None:   
    pl.seed_everything(cfg.seed)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    module = hydra.utils.instantiate(
        cfg.module,
        class_list=datamodule.class_list
    )
    callbacks = {key: hydra.utils.instantiate(cb) for key, cb in cfg.callbacks.items()}
    #dsm = pl.callbacks.DeviceStatsMonitor()
    trainer = hydra.utils.instantiate(cfg.trainer)(
        logger=pl.loggers.TensorBoardLogger(
            ".", "", "", default_hp_metric=False
        ),
        callbacks=list(callbacks.values())#+[dsm]
    )
    #module = torch.compile(module)
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.ckpt)
    trainer.predict(module, dataloaders=[datamodule.val_dataloader()])
    
if __name__ == "__main__":
    train()