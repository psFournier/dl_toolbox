import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

logger = logging.getLogger(__name__)

@hydra.main(config_path="../configs", config_name="train.yaml")
def train(cfg: DictConfig) -> None:
    pl.seed_everything(1234)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    
    datamodule = hydra.utils.instantiate(
        cfg.datamodule
    )
    
    # Instantiate all modules specified in the configs
    module = hydra.utils.instantiate(
        cfg.module,  # Object to instantiate
        # Overwrite arguments at runtime that depends on other modules
        num_classes=datamodule.num_classes,
        input_dim=datamodule.input_dim,
        # Don't instantiate optimizer submodules with hydra, let `configure_optimizers()` do it
        #_recursive_=False,
    )

    # Let hydra manage direcotry outputs
    tensorboard = pl.loggers.TensorBoardLogger(".", "", "", log_graph=True, default_hp_metric=False)
    callbacks = [
        pl.callbacks.ModelCheckpoint(monitor='loss/val'),
        pl.callbacks.EarlyStopping(monitor='loss/val', patience=50),
    ]

    trainer = pl.Trainer(
        **OmegaConf.to_container(cfg.trainer),
        logger=tensorboard,
        callbacks=callbacks,
    )

    trainer.fit(model, datamodule=data_module)
    trainer.test(model, datamodule=data_module)  # Optional


if __name__ == '__main__':
    train()
