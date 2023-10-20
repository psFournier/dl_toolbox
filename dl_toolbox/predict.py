import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import torch
#from torchsummary import summary

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')

@hydra.main(version_base="1.3", config_path="../configs", config_name="default_pred.yaml")
def predict(cfg: DictConfig) -> None:
    
    pl.seed_everything(cfg.seed)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)

    datamodule = hydra.utils.instantiate(cfg.datamodule)
    module = hydra.utils.instantiate(
        cfg.module,
        num_classes=datamodule.num_classes,
        in_channels=datamodule.in_channels,
        # Don't instantiate optimizer submodules with hydra, let `configure_optimizers()` do it
        # _recursive_=False,
    )

    callbacks = {key: hydra.utils.instantiate(cb) for key, cb in cfg.callbacks.items()}
    callbacks['preds_writer'] = hydra.utils.instantiate(cfg.preds_writer)
    trainer = hydra.utils.instantiate(cfg.trainer)(
        callbacks=list(callbacks.values())
    )
    
    trainer.predict(module, datamodule=datamodule, ckpt_path=cfg.ckpt, return_predictions=False)

if __name__ == "__main__":
    predict()
