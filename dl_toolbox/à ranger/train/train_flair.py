from pytorch_lightning.callbacks import ModelCheckpoint#, DeviceStatsMonitor
from pytorch_lightning import Trainer
#from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger
import dl_toolbox.lightning_modules as modules
import dl_toolbox.lightning_datamodules as datamodules
from dl_toolbox.lightning_modules import *
from dl_toolbox.lightning_datamodules import *
from pathlib import Path, PurePath
from datetime import datetime
import os
from random import shuffle


def main():
    """
    TODO
    """
    #data_path = Path('/data/toy_dataset_flair-one')
    data_path = Path('/scratchf/CHALLENGE_IGN')
    baseline_val_domains = ['D004_2021', 'D014_2020', 'D029_2021', 'D031_2019', 'D058_2020', 'D077_2021', 'D067_2021', 'D066_2021']
    baseline_train_domains = ['D033_2021', 'D055_2018', 'D072_2019', 'D044_2020', 'D017_2018', 'D086_2020', 'D049_2020', 'D016_2020', 'D063_2019', 'D091_2021', 'D070_2020', 'D013_2020', 'D023_2020', 'D074_2020', 'D021_2020', 'D080_2021', 'D078_2021', 'D032_2019', 'D081_2020', 'D046_2019', 'D052_2019', 'D051_2019', 'D038_2021', 'D009_2019', 'D034_2021', 'D006_2020', 'D008_2019', 'D041_2021', 'D035_2020', 'D007_2020', 'D060_2021', 'D030_2021']
    domains = baseline_val_domains + baseline_train_domains
    trainval_domains = [data_path / "train" / domain for domain in domains]
    #train_domains, val_domains = trainval_domains[:39], trainval_domains[39:]
    shuffle(trainval_domains)
    idx_split = int(len(trainval_domains) * 0.95)
    train_domains, val_domains = trainval_domains[:idx_split], trainval_domains[idx_split:] 

    datamodule = datamodules.Flair(
        #data_path,
        batch_size=32,
        crop_size=256,
        epoch_len=10000,
        labels='13',
        workers=12,
        use_metadata=False,
        train_domains=train_domains,
        val_domains=val_domains,
        test_domains=None,
        unsup_train_idxs=None,
        img_aug='d4',
        unsup_img_aug=None,
    )
    

    module = modules.CE(
        ignore_index=-1,
        #no_pred_zero=True,
        #mixup=0.4,
        #network='Vgg',
        network='EncodeThenUpsample',
        #encoder='efficientnet-b1',
        #pretrained=False,
        #bn=True,
        weights=[2.3, 2.3, 1.3, 5.1, 3.7, 7.0, 1.2, 2.7, 5.7, 1.0, 1.8, 5.3, 0.],
        in_channels=5,
        out_channels=13,
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
        max_steps=1000000,
        gpus=1,
        multiple_trainloader_mode='min_size',
        limit_train_batches=1.,
        limit_val_batches=1.,
        logger=TensorBoardLogger(
            save_dir='/d/pfournie/outputs/flair',
            #save_dir='/data/outputs/flair',
            #save_dir='/home/pfournie/ai4geo/ouputs/semcity',
            name='TBlogs_wce_d4',
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
