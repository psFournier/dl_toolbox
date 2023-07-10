import logging

import hydra
import pytorch_lightning as pl
from omegaconf import DictConfig, OmegaConf

from dl_toolbox.datasources import main_nomenclature
from dl_toolbox.utils import NomencToRgb

logger = logging.getLogger(__name__)

@hydra.main(version_base="1.3", config_path="../configs", config_name="train.yaml")
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
        in_channels=datamodule.input_dim,
        class_weights=[1]*datamodule.num_classes,
        # Don't instantiate optimizer submodules with hydra, let `configure_optimizers()` do it
        #_recursive_=False,
    )

    # Let hydra manage direcotry outputs
    tensorboard = pl.loggers.TensorBoardLogger(".", "", "", log_graph=True, default_hp_metric=False)
    
    #metrics_from_confmat = callbacks.MetricsFromConfmat(        
    #    num_classes=num_classes,
    #    class_names=[label.name for label in nomenclature_desc]
    #)
    
    callbacks = {key: hydra.utils.instantiate(cb) for key, cb in cfg.callbacks.items()}
    callbacks['confmat'] = callbacks['confmat'](
        num_classes=datamodule.num_classes,
        class_names=datamodule.class_names
    ) 
    callbacks['image_visu'] = callbacks['image_visu'](
        visu_fn=NomencToRgb(nomenc=main_nomenclature)
    )
    #for name, cb in cfg.callbacks:
    #    if name=='confmat':
    #        callbacks.append()
    #    else:
    #        callbacks.append(hydra.utils.instantiate(cb))
    #callbacks = [ for cb in cfg.callbacks.values()]
    #confmat_cb = hydra.utils.instantiate(
    
    trainer = hydra.utils.instantiate(cfg.trainer)(
        logger=tensorboard,
        callbacks=list(callbacks.values())
    )

    trainer.fit(module, datamodule=datamodule)

if __name__ == '__main__':
    train()
