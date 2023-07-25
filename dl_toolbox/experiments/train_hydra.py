import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf


@hydra.main(version_base="1.3", config_path="../../configs", config_name="train.yaml")
def train(cfg: DictConfig) -> None:
    pl.seed_everything(1234)
    logger.info("\n" + OmegaConf.to_yaml(cfg))

    datamodule = hydra.utils.instantiate(cfg.datamodule)

    module = hydra.utils.instantiate(
        cfg.module,
        num_classes=datamodule.num_classes,
        in_channels=datamodule.in_channels,
        class_weights=datamodule.class_weights,
        # Don't instantiate optimizer submodules with hydra, let `configure_optimizers()` do it
        # _recursive_=False,
    )

    # Let hydra manage directory outputs
    tensorboard = pl.loggers.TensorBoardLogger(
        ".", "", "", log_graph=True, default_hp_metric=False
    )

    trainer = hydra.utils.instantiate(cfg.trainer)(logger=tensorboard)

    trainer.fit(module, datamodule=datamodule)

if __name__ == "__main__":
    train()
