import os
import torch
import numpy as np
from pathlib import Path
from pytorch_lightning.callbacks import BasePredictionWriter
import rasterio


class TiffPredsWriter(BasePredictionWriter):

    def __init__(
        self,
        out_path,
        write_mode
    ):
        super().__init__(write_interval='batch')
        
        self.out_path = Path(out_path)
        self.write_mode = write_mode
        self.out_path.mkdir(exist_ok=True, parents=True)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        logits = outputs.cpu()  # Move predictions to CPU    
        outputs = pl_module.logits2probas(logits)
        count = pl_module.num_classes
        if self.write_mode == 'pred':
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
            
            
            
        
