from functools import partial
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Subset
import torch.nn as nn
from itertools import product
from rasterio.windows import Window

import dl_toolbox.datasets as datasets
import dl_toolbox.datamodules as datamodules
import dl_toolbox.transforms as tf
import dl_toolbox.utils as utils
from dl_toolbox.modules import Supervised
from dl_toolbox.networks import SmpUnet
from dl_toolbox.losses import DiceLoss
import dl_toolbox.callbacks as cb


def main():
    
    module = Supervised(
        network=partial(SmpUnet, encoder_name='efficientnet-b0'),
        optimizer=None,
        scheduler=None,
        ce_loss=partial(nn.CrossEntropyLoss),
        dice_loss=partial(DiceLoss),
        class_weights=[1]*7,
        ce_weight=0,
        dice_weight=0,
        in_channels=3,
        num_classes=7,
        tf=tf.NoOp()
    )
    
    ckpt = Path("/data/outputs/digitanie/run_sup/2023-09-18_175546/checkpoints/last.ckpt")
    #pred_writer = cb.TiffPredsWriter(
    #    out_path=ckpt.parent/(str(ckpt.stem)+'_preds'),
    #    base='/data'
    #)
    merge_preds = cb.MergePreds((2048,2048))

    imgs = ['ARCACHON/ARCACHON_20180821_16bits_COG_0.tif']
    msks = ['ARCACHON/COS9/ARCACHON_0_mask.tif']
    base = Path('/data/DIGITANIE_v4')
    imgs = [base/img for img in imgs]
    msks = [base/msk for msk in msks]
    
    for img, msk in zip(imgs, msks):
        
        #accuracy = M.Accuracy(task='multiclass', num_classes=7)
        #cm = M.ConfusionMatrix(task="multiclass", num_classes=7)
        
        tiles = list(utils.get_tiles(2048, 2048, 512, step_w=256))
        ds = datasets.Digitanie(
            [img]*len(tiles),
            [msk]*len(tiles),
            tiles,
            [1,2,3],
            'main6',
            tf.StretchToMinmax([0,0,0], [2000, 2000, 2000])
        )
        
        loader = DataLoader(
            dataset=ds,
            collate_fn=utils.CustomCollate(),
            batch_size=8,
            num_workers=0,
            pin_memory=True,
            shuffle=False,
            drop_last=False
        )

        trainer = Trainer(
            accelerator="gpu",
            devices=1,
            limit_predict_batches=1.,
            callbacks=[merge_preds],
            logger=False
        )
        
        trainer.predict(model=module, dataloaders=loader, ckpt_path=ckpt)
        print(merge_preds.merged.shape)

if __name__ == "__main__":
    main()
