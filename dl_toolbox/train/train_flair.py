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

    trainer = Trainer(
        max_steps=1000000,
        gpus=1,
        multiple_trainloader_mode='min_size',
        limit_train_batches=1,
        limit_val_batches=1,
        logger=TensorBoardLogger(
            #save_dir='/scratchl/pfournie/outputs/digitaniev2',
            save_dir='/data/outputs/flair',
            #save_dir='/home/pfournie/ai4geo/ouputs/semcity',
            name='ce_d4',
            version=f'{datetime.now():%d%b%y-%Hh%M}'
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
    
    data_path = Path('/data/toy_dataset_flair-one')
    trainval_domains = [data_path / "train" / domain for domain in os.listdir(data_path / "train")]
    shuffle(trainval_domains)
    idx_split = int(len(trainval_domains) * 0.9)
    train_domains, val_domains = trainval_domains[:idx_split], trainval_domains[idx_split:] 
    test_domains = [data_path / "test" / domain for domain in os.listdir(data_path / "test")]

    datamodule = datamodules.Flair(
        #data_path,
        batch_size=4,
        labels='13',
        workers=4,
        use_metadata=False,
        train_domains=train_domains,
        val_domains=val_domains,
        test_domains=test_domains,
        unsup_train_idxs=None,
        img_aug='d4',
        unsup_img_aug=None,
    )
    

    module = modules.CE(
        ignore_index=-1,
        #no_pred_zero=True,
        #mixup=0.4,
        #network='Vgg',
        network='SmpUnet',
        encoder='efficientnet-b1',
        pretrained=False,
        weights=[],
        in_channels=5,
        out_channels=13,
        initial_lr=0.001,
        final_lr=0.0005,
        plot_calib=True,
        class_names=datamodule.class_names,
        #alphas=(0., 1.),
        #ramp=(0, 40000),
        #pseudo_threshold=0.9,
        #consist_aug='color-3',
        #emas=(0.9, 0.999)
    )

    #ckpt_path='/data/outputs/test_bce_resisc/version_2/checkpoints/epoch=49-step=14049.ckpt'
    trainer.fit(
        model=module,
        datamodule=datamodule,
        ckpt_path=None
    )

if __name__ == "__main__":

    main()
