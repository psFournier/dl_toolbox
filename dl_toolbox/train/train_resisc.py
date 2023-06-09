from pytorch_lightning.callbacks import ModelCheckpoint#, DeviceStatsMonitor
from pytorch_lightning import Trainer
#from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from dl_toolbox.lightning_modules import *
from dl_toolbox.lightning_datamodules import *
from pathlib import Path, PurePath
from datetime import datetime
import os



"""
Changer data_path, labels, splitfile_path, epoch_len, save_dir... en fonction de l'expé.
"""
split='france'
#data_path=Path(os.environ['TMPDIR'])/'DIGITANIE'
data_path=Path('/work/OT/ai4geo/DATA/DATASETS/DIGITANIE')
#data_path=Path('/scratchf/AI4GEO/DIGITANIE')
#data_path=Path('/data/DIGITANIE')
#data_path=Path('/home/pfournie/ai4geo/data/SemCity-Toulouse-bench')
epoch_len=5000
labels='6mainFuseVege'
out_channels=6
max_steps=50000

datamodule = Splitfile(
    epoch_len=epoch_len,
    batch_size=16,
    workers=6,
    splitfile_path=Path.home() / f'dl_toolbox/dl_toolbox/lightning_datamodules/splits/digitanie/{split}.csv',
    test_folds=(8,9),
    train_folds=tuple(range(8)),
    data_path=data_path,
    crop_size=256,
    img_aug='d4',
    unsup_img_aug=None,
    labels=labels,
    unsup_train_folds=None
)

module = CE(
    ignore_index=0,
    #no_pred_zero=True,
    #mixup=0.4,
    #network='Vgg',
    network='SmpUnet',
    encoder='efficientnet-b4',
    pretrained=False,
    weights=[],
    in_channels=4,
    out_channels=out_channels,
    initial_lr=0.001,
    final_lr=0.0005,
    plot_calib=False,
    class_names=datamodule.class_names,
    #alphas=(0., 1.),
    #ramp=(0, 40000),
    #pseudo_threshold=0.9,
    #consist_aug='color-3',
    #emas=(0.9, 0.999)
)

trainer = Trainer(
    max_steps=max_steps,
    gpus=1,
    multiple_trainloader_mode='min_size',
    limit_train_batches=1.,
    limit_val_batches=1.,
    logger=TensorBoardLogger(
        #save_dir='/scratchl/pfournie/outputs/digitaniev2',
        save_dir='/work/OT/ai4usr/fournip/outputs/digitanie',
        #save_dir='/data/outputs/digitaniev2',
        #save_dir='/home/pfournie/ai4geo/ouputs/semcity',
        name=split,
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
    
    """
    datamodule = FromFolderDataset(
        folder_dataset='Resisc',
        #data_path='/data/NWPU-RESISC45',
        data_path='/scratchf/NWPU-RESISC45',
        batch_size=4,
        workers=6,
        train_idxs=[700*i+j for i in range(45) for j in range(50)],
        test_idxs=[700*i+j for i in range(45) for j in range(650,700)],
        unsup_train_idxs=[700*i+j for i in range(45) for j in range(700)],
        img_aug='d4_color-3',
        unsup_img_aug=None
    )
    """

#ckpt_path='/data/outputs/test_bce_resisc/version_2/checkpoints/epoch=49-step=14049.ckpt'
trainer.fit(
    model=module,
    datamodule=datamodule,
    ckpt_path=None
)
