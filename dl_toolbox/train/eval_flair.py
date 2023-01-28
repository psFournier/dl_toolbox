from pytorch_lightning.callbacks import ModelCheckpoint#, DeviceStatsMonitor
from pytorch_lightning import Trainer
#from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger
import dl_toolbox.lightning_modules as modules
import dl_toolbox.lightning_datamodules as datamodules
import dl_toolbox.callbacks as callbacks
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
    test_domains = [data_path / "test" / domain for domain in os.listdir(data_path / "test")]

    datamodule = datamodules.Flair(
        #data_path,
        batch_size=8,
        crop_size=320,
        epoch_len=1,
        labels='13',
        workers=16,
        use_metadata=False,
        train_domains=None,
        val_domains=None,
        test_domains=test_domains,
        img_aug='no',
    )
    

    module = modules.CE(
        ignore_index=-1,
        #no_pred_zero=True,
        #mixup=0.4,
        #network='Vgg',
        network='SmpUnet',
        encoder='efficientnet-b1',
        pretrained=False,
        bn=True,
        weights=[],
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
    
    output_dir = '/d/pfournie/outputs/flair/TBlogs_wce_d4/27Jan23-23h12m16'
    trainer = Trainer(
        gpus=1,
        callbacks=[
            callbacks.PredictionWriter(        
                #output_dir=os.path.join('/data/outputs/flair', "predictions"+"_ce_d4"),
                output_dir=os.path.join(output_dir, "preds"),
                write_interval="batch",
            )
        ],
        logger=None,
        enable_progress_bar=True
    )

    ckpt_path=os.path.join(output_dir, "checkpoints/epoch=208-step=261249.ckpt")
    trainer.predict(
        model=module,
        datamodule=datamodule,
        ckpt_path=ckpt_path,
        return_predictions=False
    )

if __name__ == "__main__":

    main()
