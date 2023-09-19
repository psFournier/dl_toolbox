import hydra
import logging
import torch
import pytorch_lightning as pl
from pathlib import Path
from dl_toolbox.datamodules import Resisc
import dl_toolbox.transforms as tf
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')

@hydra.main(version_base="1.3", config_path="../../../configs", config_name="default.yaml")
def main(cfg) -> None:
    dm = Resisc(
        data_path=Path('/data'),
        merge='all45',
        sup=10,
        unsup=0,
        dataset_tf=tf.StretchToMinmax([0]*3, [255.]*3),
        batch_size=16,
        num_workers=4,
        pin_memory=True,
        class_weights=None
    )
    pl.seed_everything(cfg.seed)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    module = hydra.utils.instantiate(
        cfg.module,
        num_classes=dm.num_classes,
        in_channels=dm.in_channels,
        class_weights=dm.class_weights,
    )
    callbacks = {key: hydra.utils.instantiate(cb) for key, cb in cfg.callbacks.items()}
    trainer = hydra.utils.instantiate(cfg.trainer)(
        logger=False, 
        callbacks=list(callbacks.values())
    )
    trainer.test(module, dm, verbose=True, ckpt_path=cfg.ckpt)

if __name__ == "__main__":
    main()