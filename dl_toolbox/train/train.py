from pytorch_lightning.callbacks import ModelCheckpoint#, DeviceStatsMonitor
from pytorch_lightning import Trainer
#from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from dl_toolbox.lightning_modules import *
from dl_toolbox.lightning_datamodules import *


def main():
    """
    TODO
    """

    trainer = Trainer(
        max_steps=100000,
        gpus=1,
        multiple_trainloader_mode='min_size',
        limit_train_batches=1.,
        limit_val_batches=1.,
        logger=TensorBoardLogger(
            save_dir='/data/outputs',
            name='resisc50',
            version='cemt_d4color3mixup',
            sub_dir='0'
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

    datamodule = FromFolderDataset(
        folder_dataset='Resisc',
        data_path='/data/NWPU-RESISC45',
        batch_size=16,
        workers=6,
        train_idxs=[700*i+j for i in range(45) for j in range(50)],
        test_idxs=[700*i+j for i in range(45) for j in range(650,700)],
        unsup_train_idxs=[700*i+j for i in range(45) for j in range(700)],
        img_aug='d4_color-3',
        unsup_img_aug=None
    )

    module = CE_MT(
        #ignore_index=-1,
        #no_pred_zero=False,
        mixup=0.4,
        network='Vgg',
        weights=[],
        in_channels=3,
        out_channels=45,
        initial_lr=0.001,
        final_lr=0.0005,
        plot_calib=True,
        class_names=datamodule.val_set.dataset.class_names,
        alphas=(0., 1.),
        ramp=(0, 40000),
        pseudo_threshold=0.9,
        consist_aug='d4_color-3',
        emas=(0.9, 0.999)
    )

    #ckpt_path='/data/outputs/test_bce_resisc/version_2/checkpoints/epoch=49-step=14049.ckpt'
    trainer.fit(
        model=module,
        datamodule=datamodule,
        ckpt_path=None
    )

if __name__ == "__main__":

    main()
