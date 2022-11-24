from argparse import ArgumentParser
from pytorch_lightning.callbacks import ModelCheckpoint, DeviceStatsMonitor
from pytorch_lightning import Trainer 
from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from dl_toolbox.lightning_modules import CE
from dl_toolbox.networks import Vgg
from dl_toolbox.torch_datasets import ResiscDs
from torch.utils.data import DataLoader, Subset, ConcatDataset
from dl_toolbox.torch_collate import CustomCollate
from dl_toolbox.torch_sample import BalancedConcat
import os
from pathlib import Path
import torch

def main():

    sup_batch_size = 16
    num_workers = 4
    max_epochs = 5
    output_dir = '/d/pfournie/ai4geo/outputs'

    module = CE(
        ignore_index=-1,
        network='Vgg',
        weights=[],
        in_channels=3,
        out_channels=2
    )

    logger = TensorBoardLogger(
        output_dir,
        name='test_pl_seq'
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[
            ModelCheckpoint(),
        ],
        gpus=1
    )

    resisc = ResiscDs(
        data_path='/d/pfournie/ai4geo/data/NWPU-RESISC45',
        img_aug='d4',
    )

    train_set = Subset(
        dataset=resisc,
        indices=[700*i+j for i in range(2) for j in range(16)]
    )
    val_set = Subset(
        dataset=resisc,
        indices=[700*i+j for i in range(2) for j in range(16, 24)]
    )

    train_dataloader = DataLoader(
        dataset=train_set,
        batch_size=sup_batch_size,
        collate_fn=CustomCollate(),
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=True
    )

    val_dataloader = DataLoader(
        dataset=val_set,
        shuffle=False,
        collate_fn=CustomCollate(),
        batch_size=sup_batch_size,
        num_workers=num_workers,
        pin_memory=True
    )

    trainer.fit(
        model=module,
        train_dataloader=train_dataloader,
        val_dataloaders=val_dataloader
    )


    pred_set = Subset(
        dataset=resisc,
        indices=[700*i+j for i in range(2) for j in range(16, 48)]
    )

    pred_dataloader = DataLoader(
        dataset=pred_set,
        shuffle=False,
        collate_fn=CustomCollate(),
        batch_size=sup_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    pl_dir = os.path.join(
        logger.save_dir,
        logger.name,
        logger.log_dir,
        'NWPU-RESISC45'
    )

    counts = [0]*2

    for batch in pred_dataloader:

        logits = module.forward(batch['image'])
        probas = module._compute_probas(logits)
        _, preds = module._compute_conf_preds(probas)

        for i in range(sup_batch_size):

            pred = int(preds[i])
            cls_name = resisc.classes[pred]
            num = counts[pred]
            class_dir = Path(pl_dir) / cls_name
            class_dir.mkdir(parents=True, exist_ok=True)
            dst = class_dir / f'{cls_name}_{num:03}.jpg'
            os.symlink(
                batch['path'][i],
                dst
            )
            counts[preds[i]] += 1

    pl_set = ResiscDs(
        data_path=pl_dir,
        img_aug='d4_color-5'
    )

    pl_train_set = ConcatDataset([train_set, pl_set])

    pl_sampler = BalancedConcat(
        lengths=[len(train_set), len(pl_set)],
        num_samples=len(train_set)
    )

    pl_train_dataloader = DataLoader(
        dataset=pl_train_set,
        sampler=pl_sampler,
        collate_fn=CustomCollate(),
        batch_size=sup_batch_size,
        num_workers=num_workers,
        pin_memory=True,
    )

    pl_module = CE(
        ignore_index=-1,
        network='Vgg',
        weights=[],
        in_channels=3,
        out_channels=2
    )

    trainer = Trainer(
        max_epochs=max_epochs,
        logger=logger,
        callbacks=[
            ModelCheckpoint(),
        ],
        gpus=1
    )

    trainer.fit(
        model=pl_module,
        train_dataloader=pl_train_dataloader,
        val_dataloaders=val_dataloader
    )


if __name__ == "__main__":

    main()
