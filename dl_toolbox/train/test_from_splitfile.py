from pytorch_lightning.callbacks import ModelCheckpoint#, DeviceStatsMonitor
from pytorch_lightning import Trainer
#from pytorch_lightning.profiler import SimpleProfiler
from pytorch_lightning.loggers import TensorBoardLogger
from dl_toolbox.lightning_modules import *
from dl_toolbox.lightning_datamodules import *
import dl_toolbox.callbacks as callbacks

from pathlib import Path, PurePath
from datetime import datetime
import os
from argparse import ArgumentParser
from pathlib import Path
import csv
import ast 
from pytorch_lightning import LightningDataModule

from torch.utils.data import DataLoader, RandomSampler, ConcatDataset
from rasterio.windows import Window
from dl_toolbox.torch_datasets import *

#data_path=Path(os.environ['TMPDIR'])/'DIGITANIE'
#data_path=Path('/work/OT/ai4geo/DATA/DATASETS/DIGITANIE')
#data_path=Path('/scratchf/AI4GEO/DIGITANIE')
data_path=Path('/data/DIGITANIE')
#data_path=Path('/home/pfournie/ai4geo/data/SemCity-Toulouse-bench')

### Building training and validation datasets to concatenate from the splitfile lines
test_sets = []
splitfile_path=Path.home() / f'dl_toolbox/dl_toolbox/lightning_datamodules/splits/digitanie/arcachon.csv'
dataset_factory = DatasetFactory()
with open(splitfile_path, newline='') as splitfile:
    reader = csv.reader(splitfile)
    next(reader)
    for row in reader:
        ds_name, _, image_path, label_path, x0, y0, w, h, fold, mins, maxs = row[:11]
        if int(fold) in [8,9]:
            window = Window(
                col_off=int(x0),
                row_off=int(y0),
                width=int(w),
                height=int(h)
            )
            ds = dataset_factory.create(ds_name)(
                image_path=data_path/image_path,
                label_path=data_path/label_path if label_path else None,
                tile=window,
                crop_size=256,
                fixed_crops=True,
                img_aug=None,
                labels='6mainFuseVege',
                mins=ast.literal_eval(mins),
                maxs=ast.literal_eval(maxs)
            )
            test_sets.append(ds)

### Building dataloaders
test_set = ConcatDataset(test_sets)
dataloader = DataLoader(
    dataset=test_set,
    shuffle=False,
    collate_fn=CustomCollate(),
    batch_size=8,
    num_workers=8,
    pin_memory=True
)

### Building lightning module
module = CE(
    out_channels=6,
    ignore_zero=False,
    #no_pred_zero=True,
    #mixup=0.4,
    #network='Vgg',
    network='SmpUnet',
    encoder='efficientnet-b4',
    pretrained=False,
    weights=[],
    in_channels=4,
    initial_lr=0.001,
    final_lr=0.0005,
    plot_calib=False,
    ttas = ['vflip'],
    #no_pred_zero=True,
    #mixup=0.4,
    #network='Vgg',
    class_names=list(test_set.datasets[0].labels.keys()),
    #alphas=(0., 1.),
    #ramp=(0, 40000),
    #pseudo_threshold=0.9,
    #consist_aug='color-3',
    #emas=(0.9, 0.999)
)

#save_dir='/scratchl/pfournie/outputs/digitaniev2'
#save_dir='/work/OT/ai4usr/fournip/outputs/digitanie'
save_dir='/data/outputs/digitanie'
#save_dir='/home/pfournie/ai4geo/ouputs/semcity'

output_dir = Path(save_dir) / 'arcachon/01Mar23-09h56m08'

### Prediction writer callback
prediction_writer = callbacks.PredictionWriter(        
    out_path=output_dir / "preds",
    write_interval="batch",
)

### Trainer instance
trainer = Trainer(
    accelerator='gpu',
    devices=1,
    logger=None,
    callbacks=[prediction_writer]
)
    
#ckpt_path='/data/outputs/test_bce_resisc/version_2/checkpoints/epoch=49-step=14049.ckpt'
trainer.predict(
    model=module,
    dataloaders=dataloader,
    ckpt_path=output_dir / "checkpoints/epoch=29-step=18749.ckpt",
    return_predictions=False
)
