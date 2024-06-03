# Third-party libraries
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn

from dl_toolbox.utils import labels_to_rgb

def display_seg_batch(trainer, module, batch, prefix):
    x, tgt, p = batch
    logits = module.forward(x).cpu()
    prob = module.loss.prob(logits)
    conf, pred = module.loss.pred(prob)
    img = x.cpu()
    y = tgt["masks"].cpu()
    colors = trainer.datamodule.class_colors
    y_rgb = labels_to_rgb(y, colors=colors).transpose((0, 3, 1, 2))
    y_rgb = torch.from_numpy(y_rgb)
    pred_rgb = labels_to_rgb(pred, colors=colors).transpose((0, 3, 1, 2))
    pred_rgb = torch.from_numpy(pred_rgb)
    nb = min(4, x.shape[0])
    imgs = torchvision.utils.make_grid(img[:nb, ...], normalize=True)
    masks = torchvision.utils.make_grid(y_rgb[:nb, ...]).float() / 255.
    preds = torchvision.utils.make_grid(pred_rgb[:nb, ...]).float() / 255.
    grid = torch.cat([imgs, masks, preds], dim=1)
    step = trainer.global_step
    trainer.logger.experiment.add_image(f"{prefix} images", grid, step)

class SegmentationImagesVisualisation(pl.Callback):

    def __init__(self, freq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        
    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq>0 and trainer.current_epoch % self.freq == 0 and batch_idx <= 1:
            display_seg_batch(trainer, pl_module, batch["sup"], "Train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq>0 and trainer.current_epoch % self.freq == 0 and batch_idx <= 1:
            display_seg_batch(trainer, pl_module, batch, "Val")
            
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq>0 and batch_idx <= 1:
            display_seg_batch(trainer, pl_module, batch, "Test")
