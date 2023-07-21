from pytorch_lightning.callbacks import ModelCheckpoint#, DeviceStatsMonitor
from pytorch_lightning import Trainer
#from pytorch_lightning.profiler import SimpleProfiler
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


def main():
    """
    TODO
    """
    #data_path = Path('/data/toy_dataset_flair-one')
    
    
    

    datamodule = datamodules.Flair(
        data_path=Path('/data/flair_merged'),
        batch_size=32,
        crop_size=256,
        epoch_len=10000,
        labels='13',
        workers=8,
        use_metadata=False,
        #train_domains=train_domains,
        #val_domains=val_domains,
        #test_domains=None,
        unsup_train_idxs=None,
        img_aug='d4',
        unsup_img_aug=None,
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
        class_weights=[1]*13,
        #ttas,
        in_channels=3,
        num_classes=13,
    )

    trainer = Trainer(
        max_steps=100000,
        accelerator='gpu',
        devices=1,
        default_root_dir='/data/outputs/flair',
        limit_train_batches=1.,
        limit_val_batches=1.,
        logger=TensorBoardLogger(
            save_dir='/data/outputs/flair',
            #save_dir='/data/outputs/flair',
            #save_dir='/home/pfournie/ai4geo/ouputs/semcity',
            version=f'{datetime.now():%d%b%y-%Hh%Mm%S}'
        ),
        #profiler=SimpleProfiler(),
        callbacks=[
            ModelCheckpoint(),
            #DeviceStatsMonitor(),
        ],
        num_sanity_val_steps=0,
        check_val_every_n_epoch=1,
        benchmark=True,
        enable_progress_bar=True
    )

    #ckpt_path='/data/outputs/test_bce_resisc/version_2/checkpoints/epoch=49-step=14049.ckpt'
    trainer.fit(
        model=module,
        datamodule=datamodule,
        ckpt_path=None
    )

if __name__ == "__main__":

    main()
