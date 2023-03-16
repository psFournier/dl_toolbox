import os
import torch
from pickletools import uint8
import numpy as np
from pathlib import Path
from pytorch_lightning.callbacks import BasePredictionWriter
try: 
    from pytorch_lightning.utilities.distributed import rank_zero_only 
except ImportError:
    from pytorch_lightning.utilities.rank_zero import rank_zero_only 
from PIL import Image
from copy import copy
import rasterio


class TiffPredictionWriter(BasePredictionWriter):

    def __init__(
        self,
        out_path,
        write_interval,
        mode
    ):
        super().__init__(write_interval)
        self.out_path = Path(out_path)
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.mode = mode

    def write_on_batch_end(
        self,
        trainer,
        pl_module,
        outputs,
        batch_indices,
        batch,
        batch_idx,
        dataloader_idx,
    ):
        
        logits = outputs.cpu()  # Move predictions to CPU    
        outputs = pl_module.logits2probas(logits)
        count = int(outputs.shape[1])
        if self.mode == 'pred':
            outputs = torch.unsqueeze(pl_module.probas2confpreds(outputs)[1], 1)
            print(outputs.shape)
            count = 1
        for output, path, crop, tf, crs in zip(outputs, batch['path'], batch['crop'], batch['crop_tf'], batch['crs']):
            os.makedirs(self.out_path/path.stem, exist_ok=True)
            out_file = self.out_path/path.stem/f'{crop.col_off}_{crop.row_off}.tif'
            meta = {
                'driver': 'GTiff',
                'height': crop.height,
                'width':crop.width,
                'count':count,
                'dtype':np.float32,
                'crs': crs,
                'transform': tf,
            }
            with rasterio.open(out_file, 'w+', **meta) as dst:
                dst.write(output.numpy())

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.interval.on_batch:
            return

        batch_indices = trainer.predict_loop.epoch_loop.current_batch_indices
        self.write_on_batch_end(
            trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx
        )
            
            
            
            
        
