from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import TensorBoardLogger
import dl_toolbox.modules as modules
import dl_toolbox.datamodules as datamodules
from pathlib import Path, PurePath
from datetime import datetime
import os
from random import shuffle
from functools import partial
import segmentation_models_pytorch as smp
import torch
import dl_toolbox.transforms as transforms
import dl_toolbox.callbacks as callbacks


def main():
    """
    TODO
    """
    #data_path = Path('/data/toy_dataset_flair-one')
    
    datamodule = datamodules.Flair(
        data_path=Path('/data/flair_merged'),
        bands=[1,2,3],
        batch_size=8,
        crop_size=256,
        train_tf=transforms.Compose([transforms.D4()]),
        val_tf=transforms.NoOp(),
        merge='main13',
        num_workers=0,
        pin_memory=False,
        class_weights=None
    )
    
    module = modules.Multilabel(
        network=partial(
            smp.Unet,
            encoder_name='efficientnet-b0'
        ),
        optimizer=partial(
            torch.optim.SGD,
            lr=0.04,
            momentum=0.9,
            weight_decay=0.0001
        ),
        scheduler=partial(
            torch.optim.lr_scheduler.StepLR,
            step_size=10,
            gamma=0.1
        ),
        class_weights=datamodule.class_weights,
        in_channels=datamodule.in_channels,
        num_classes=datamodule.num_classes,
    )

    trainer = Trainer(
        max_epochs=3,
        accelerator='gpu',
        devices=1,
        default_root_dir='/data/outputs/flair',
        limit_train_batches=30,
        limit_val_batches=30,
        logger=TensorBoardLogger(
            save_dir='/data/outputs/flair',
            version=f'{datetime.now():%d%b%y-%Hh%Mm%S}'
        ),
        callbacks=[
            ModelCheckpoint(
                monitor='loss/val',
                filename="epoch_{epoch:03d}"
            ),
            callbacks.MetricsFromConfmat(
                num_classes=datamodule.num_classes,
                class_names=datamodule.class_names,
                ignore_idx=0
            ),
            callbacks.SegmentationImagesVisualisation(
                freq=1
            )
        ],
        profiler='simple',
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        benchmark=True,
        enable_progress_bar=True
    )

    trainer.fit(
        model=module,
        datamodule=datamodule,
        ckpt_path=None
    )

if __name__ == "__main__":

    main()
