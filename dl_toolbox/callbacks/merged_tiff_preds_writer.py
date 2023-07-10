import os
import torch
import numpy as np
from pathlib import Path
from pytorch_lightning.callbacks import BasePredictionWriter
import rasterio
import rasterio.windows as windows 
from functools import partial


def dist_to_edge_mask(crop_size):
    
    def dist_to_edge(i, j, h, w):
        mi = np.minimum(i+1, h-i)
        mj = np.minimum(j+1, w-j)
        return np.minimum(mi, mj)
    
    crop_mask = np.fromfunction(
        function=partial(
            dist_to_edge,
            h=crop_size,
            w=crop_size
        ),
        shape=(crop_size, crop_size),
        dtype=int
    )
    crop_mask = torch.from_numpy(crop_mask).float()
    
    return crop_mask

class MergedTiffPredsWriter(BasePredictionWriter):

    def __init__(
        self,
        out_path,
        write_mode,
        data_src,
        crop_size,
        merge_mode
    ):
        super().__init__(write_interval='epoch')
        
        self.out_path = Path(out_path)
        self.write_mode = write_mode
        self.data_src = data_src
        self.merge_mode = merge_mode
        
        self.out_path.mkdir(exist_ok=True, parents=True)
        self.window = data_src.zone
        if merge_mode=='linear':
            self.crop_mask = dist_to_edge_mask(crop_size)
        elif merge_mode=='constant':
            self.crop_mask = torch.ones((crop_size, crop_size)).float()
        
    def on_predict_epoch_start(self, trainer, pl_module):
        
        h, w = self.window.height, self.window.width
        self.merged = torch.zeros((pl_module.num_classes, h, w))
        self.weights = torch.zeros((h, w))

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):

        logits = outputs.cpu()
        probas = pl_module.logits2probas(logits)
        crops = batch['crop']
        
        for proba, crop in zip(probas, crops):
            row_off = crop.row_off - self.window.row_off
            col_off = crop.col_off - self.window.col_off
            self.merged[
                :,
                row_off:row_off+crop.height,
                col_off:col_off+crop.width
            ] += proba * self.crop_mask
            self.weights[
                row_off:row_off+crop.height,
                col_off:col_off+crop.width
            ] += self.crop_mask
            
    def on_predict_epoch_end(self, trainer, pl_module):                
        
        self.merged = torch.div(self.merged, self.weights)
        
        if self.write_mode == 'pred':
            confs, preds = pl_module.probas2confpreds(self.merged.unsqueeze(dim=0))
            output = preds
            count = 1
        else:
            output = self.merged
            count = pl_module.num_classes
            
        path = self.data_src.image_path
        out_file = self.out_path/f'{path.stem}_{self.window.col_off}_{self.window.row_off}.tif'
        meta = {
            'driver': 'GTiff',
            'height': self.window.height,
            'width':self.window.width,
            'count':count,
            'dtype':np.float32,
            'crs': self.data_src.meta['crs'],
            'transform': self.data_src.meta['transform'],
        }
        with rasterio.open(out_file, 'w+', **meta) as dst:
            dst.write(output.numpy())