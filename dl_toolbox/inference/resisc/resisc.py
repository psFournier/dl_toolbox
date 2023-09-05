from functools import partial
from pathlib import Path

from pytorch_lightning import Trainer
from torch.utils.data import DataLoader, Subset
import torch.nn as nn

from dl_toolbox.datamodules import DatamoduleResisc1
import dl_toolbox.transforms as tf
import dl_toolbox.utils as utils
from dl_toolbox.modules import Supervised
from dl_toolbox.networks import EfficientNet_b0
from dl_toolbox.losses import DiceLoss
from dl_toolbox.callbacks import ClassifPredsWriter


def main():

    datamodule = DatamoduleResisc1(
        data_path=Path('/data'),
        merge='all45',
        prop=3,
        train_tf=tf.NoOp(),
        val_tf=tf.NoOp(),
        test_tf=tf.NoOp(),
        batch_size=4,
        num_workers=4,
        pin_memory=True,
        class_weights=None
    )
    
    module = Supervised(
        network=partial(EfficientNet_b0),
        optimizer=None,
        scheduler=None,
        ce_loss=partial(nn.CrossEntropyLoss),
        dice_loss=partial(DiceLoss),
        class_weights=[1]*45,
        ce_weight=0,
        dice_weight=0,
        in_channels=3,
        num_classes=45,
    )
    
    ckpt = Path('/data/outputs/resisc_3_80/sup/2023-09-05_154623/checkpoints/last.ckpt')
    pred_dir = str(ckpt.stem)+'_preds'
    pred_writer = ClassifPredsWriter(
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
    trainer.predict(model=module, datamodule=datamodule, ckpt_path=ckpt)

if __name__ == "__main__":
    main()
