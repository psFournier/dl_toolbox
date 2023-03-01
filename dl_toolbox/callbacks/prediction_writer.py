import os
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


class PredictionWriter(BasePredictionWriter):

    def __init__(
        self,
        out_path,
        write_interval,
    ):
        super().__init__(write_interval)
        self.out_path = Path(out_path)
        self.out_path.mkdir(exist_ok=True, parents=True)

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
        
        logits = outputs.cpu().numpy()  # Move predictions to CPU        
        for logit, path, crop in zip(logits, batch['path'], batch['crop']):
            out_file = self.out_path / f'{path.stem}_{crop.col_off}_{crop.row_off}.tif'
            meta = {'driver': 'GTiff', 'height': 256, 'width':256, 'count':6, 'dtype':np.float32}
            with rasterio.open(out_file, 'w+', **meta) as dst:
                dst.write(logit)

    def on_predict_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        if not self.interval.on_batch:
            return

        batch_indices = trainer.predict_loop.epoch_loop.current_batch_indices
        self.write_on_batch_end(
            trainer, pl_module, outputs, batch_indices, batch, batch_idx, dataloader_idx
        )
