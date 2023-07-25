import os
from datetime import datetime
from functools import partial
from pathlib import Path, PurePath
from random import shuffle

from pytorch_lightning import Trainer
from segmentation_models_pytorch import Unet

import dl_toolbox.transforms as transforms

from dl_toolbox.callbacks import MetricsFromConfmat
from dl_toolbox.datamodules import Flair
from dl_toolbox.modules import Multilabel


def main():
    datamodule = Flair(
        data_path=Path("/data/FLAIR_1"),
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

    module = Multilabel(
        network=partial(Unet, encoder_name="efficientnet-b0"),
        optimizer=None,
        scheduler=None,
        class_weights=[1.]*13,
        in_channels=3,
        num_classes=13,
    )
    
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        default_root_dir="/data/outputs/flair",
        limit_val_batches=1.,
        limit_test_batches=1.,
        limit_predict_batches=1.,
        callbacks=[
            MetricsFromConfmat(
                num_classes=datamodule.num_classes,
                class_names=datamodule.class_names,
                ignore_idx=0,
            ),
        ],
    )
    
    ckpt_path = Path('/data/outputs/flair/lightning_logs/23Jul23-11h11m48/checkpoints/epoch_epoch=018.ckpt')
    trainer.validate(model=module, datamodule=datamodule, ckpt_path=ckpt_path)

if __name__ == "__main__":
    main()
