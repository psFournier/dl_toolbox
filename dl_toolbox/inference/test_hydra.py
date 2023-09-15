import logging
import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

def test(cfg: DictConfig, dataloaders) -> None:
    pl.seed_everything(cfg.seed)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    module = hydra.utils.instantiate(
        cfg.module,
        num_classes=datamodule.num_classes,
        in_channels=datamodule.in_channels,
        class_weights=datamodule.class_weights,
    )
    callbacks = {key: hydra.utils.instantiate(cb) for key, cb in cfg.callbacks.items()}
    trainer = hydra.utils.instantiate(cfg.trainer)(
        logger=False, 
        callbacks=list(callbacks.values())
    )
    trainer.test(module, dataloaders=dataloaders, verbose=True, ckpt_path=cfg.ckpt)