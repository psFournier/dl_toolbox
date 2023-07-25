import os
from datetime import datetime
from functools import partial
from pathlib import Path, PurePath
from random import shuffle

import segmentation_models_pytorch as smp
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

import dl_toolbox.callbacks as callbacks
import dl_toolbox.datamodules as datamodules
import dl_toolbox.modules as modules
import dl_toolbox.transforms as transforms


def main():
    """
    TODO
    """

    datamodule = datamodules.Flair(
        data_path="/data/FLAIR_1",
        bands=[1, 2, 3],
        batch_size=4,
        crop_size=512,
        train_tf=transforms.Compose([transforms.D4()]),
        val_tf=transforms.NoOp(),
        test_tf=transforms.NoOp(),
        merge="main13",
        num_workers=6,
        pin_memory=False,
        class_weights=None,
    )

    module = modules.Multilabel(
        network=partial(smp.Unet, encoder_name="efficientnet-b0"),
        optimizer=partial(torch.optim.SGD, lr=0.04, momentum=0.9, weight_decay=0.0001),
        scheduler=partial(torch.optim.lr_scheduler.StepLR, step_size=10, gamma=0.1),
        class_weights=datamodule.class_weights,
        in_channels=datamodule.in_channels,
        num_classes=datamodule.num_classes,
    )

    trainer = Trainer(
        max_steps=1000000,
        accelerator="gpu",
        devices=1,
        default_root_dir="/data/outputs/flair",
        limit_train_batches=1.0,
        limit_val_batches=1.0,
        logger=TensorBoardLogger(
            save_dir="/data/outputs/flair", version=f"{datetime.now():%d%b%y-%Hh%Mm%S}"
        ),
        callbacks=[
            ModelCheckpoint(monitor="loss/val", filename=f"epoch_{epoch:03d}"),
            callbacks.MetricsFromConfmat(
                num_classes=datamodule.num_classes,
                class_names=datamodule.class_names,
                ignore_idx=0,
            ),
            callbacks.SegmentationImagesVisualisation(freq=1),
        ],
        profiler="simple",
        num_sanity_val_steps=0,
        check_val_every_n_epoch=None,
        val_check_interval=1000,
    )

    trainer.fit(model=module, datamodule=datamodule, ckpt_path=None)


if __name__ == "__main__":
    main()
