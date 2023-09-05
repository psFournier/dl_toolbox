import os
from datetime import datetime
from functools import partial
from pathlib import Path, PurePath
from random import shuffle

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader
import torch.nn as nn

from dl_toolbox.datamodules.flair.utils import flair_gather_data
from dl_toolbox.datasets import DatasetFlair2
import dl_toolbox.transforms as tf
import dl_toolbox.utils as utils
from dl_toolbox.modules import Supervised
from dl_toolbox.networks import SmpUnet
from dl_toolbox.losses import DiceLoss
from dl_toolbox.callbacks import TiffPredsWriter


def main():
    
    domains = [Path("/data/FLAIR_1/train") / d for d in ["D007_2020"]]
    all_img, all_msk, all_mtd = flair_gather_data(
        domains, path_metadata=None, use_metadata=False, test_set=False
    ) 
    dataset = DatasetFlair2(
        all_img,
        all_msk,
        [1,2,3],
        merge='main13',
        transforms=tf.NoOp(),
    )
    dataloader = DataLoader(
        dataset=dataset,
        collate_fn=utils.CustomCollate(),
        batch_size=8,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
        drop_last=False,
    )
    module = Supervised(
        network=partial(SmpUnet, encoder_name="efficientnet-b0"),
        optimizer=None,
        scheduler=None,
        ce_loss=partial(nn.CrossEntropyLoss),
        dice_loss=partial(DiceLoss),
        class_weights=[1]*6,
        ce_weight=0,
        dice_weight=0,
        in_channels=3,
        num_classes=6,
    )
    ckpt = Path('/data/outputs/flair2_3_97/supervised_dummy/2023-09-05_102306/checkpoints/last.ckpt')
    pred_dir = str(ckpt.stem)+'_preds'
    pred_writer = TiffPredsWriter(
        out_path=ckpt.parent / pred_dir,
        base='/data'
    )
    trainer = Trainer(
        accelerator="gpu",
        devices=1,
        limit_predict_batches=5,
        callbacks=[pred_writer],
        logger=False
    )
    trainer.predict(model=module, dataloaders=dataloader, ckpt_path=ckpt)

if __name__ == "__main__":
    main()
