import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf
import torch
#import wandb

logger = logging.getLogger(__name__)
torch.set_float32_matmul_precision('high')

@hydra.main(version_base="1.3", config_path="../configs", config_name="default_train.yaml")
def train(cfg: DictConfig) -> None:
    
    #wandb.config = omegaconf.OmegaConf.to_container(
    #    cfg, resolve=True, throw_on_missing=True
    #)
    #wandb.init(project="test_sweep")
    
    pl.seed_everything(cfg.seed)
    logger.info("\n" + OmegaConf.to_yaml(cfg))
    logging.getLogger("lightning.pytorch").setLevel(logging.ERROR)
    tensorboard = pl.loggers.TensorBoardLogger(
        ".", "", "", default_hp_metric=False
    )
    #wandb_logger = pl.loggers.WandbLogger(
    #    project=cfg.name,
    #    log_model=False,
    #    save_dir=".",
    #    name="",
    #    version=''
    #)
    datamodule = hydra.utils.instantiate(cfg.datamodule)
    module = hydra.utils.instantiate(
        cfg.module,
        num_classes=datamodule.num_classes,
        in_channels=datamodule.in_channels,
    )
    #wandb_logger.watch(module.network)
    callbacks = {key: hydra.utils.instantiate(cb) for key, cb in cfg.callbacks.items()}
    dsm = pl.callbacks.DeviceStatsMonitor()
    trainer = hydra.utils.instantiate(cfg.trainer)(
        logger=tensorboard,
        callbacks=list(callbacks.values())+[dsm]
    )
    
    trainer.fit(module, datamodule=datamodule, ckpt_path=cfg.ckpt)
    #metrics = trainer.validate(module, datamodule=datamodule)
    #return metrics[0][cfg.optimized_metric]

if __name__ == "__main__":
    train()
    #sweep_config = {
    #    'method': 'grid',
    #    'metric': {
    #        'goal': 'minimize',
    #        'name': 'binary cross entropy/val'
    #    },
    #    'parameters': {
    #        'seed': {'values': [2,3]},
    #    }
    #}
    #sweep_id=wandb.sweep(sweep_config, project="test_sweep")
    #wandb.agent(sweep_id=sweep_id, function=train, count=2)
