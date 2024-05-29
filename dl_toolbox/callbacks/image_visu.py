# Third-party libraries
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn

from dl_toolbox.utils import labels_to_rgb


class SegmentationImagesVisualisation(pl.Callback):

    NB_COL: int = 4

    def __init__(self, freq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq

    def display_batch(self, colors, trainer, module, batch, prefix):
        x, tgt, p = batch
        logits = module.forward(x).cpu()
        prob = module.loss.prob(logits)
        conf, pred = module.loss.pred(prob)
        
        img = x.cpu()
        y = tgt["masks"].cpu()
        y_rgb = labels_to_rgb(y, colors=colors).transpose((0, 3, 1, 2))
        y_rgb = torch.from_numpy(y_rgb).float()
        pred_rgb = labels_to_rgb(pred, colors=colors).transpose((0, 3, 1, 2))
        pred_rgb = torch.from_numpy(pred_rgb).float()

        # Number of grids to log depends on the batch size
        quotient, remainder = divmod(x.shape[0], self.NB_COL)
        nb_grids = quotient + int(remainder > 0)

        for idx in range(nb_grids):
            start = self.NB_COL * idx
            if start + self.NB_COL <= x.shape[0]:
                end = start + self.NB_COL
            else:
                end = start + remainder

            img_grid = torchvision.utils.make_grid(
                img[start:end, :, :, :], padding=10, normalize=True
            )
            grids = [img_grid]
            mask_grid = torchvision.utils.make_grid(
                y_rgb[start:end, :, :, :], padding=10, normalize=True
            )
            grids.append(mask_grid)
            pred_grid = torchvision.utils.make_grid(
                pred_rgb[start:end, :, :, :], padding=10, normalize=True
            )
            grids.append(pred_grid)

            final_grid = torch.cat(grids, dim=1)
            trainer.logger.experiment.add_image(
                f"{prefix}_Images/batch0_part_{idx}",
                final_grid,
                global_step=trainer.global_step,
            )

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.freq == 0 and batch_idx <= 1:
            colors = trainer.datamodule.class_colors
            self.display_batch(colors, trainer, pl_module, batch["sup"], "Train")

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if trainer.current_epoch % self.freq == 0 and batch_idx <= 1:
            colors = trainer.datamodule.class_colors
            self.display_batch(colors, trainer, pl_module, batch, "Val")
            
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if batch_idx <= 1:
            colors = trainer.datamodule.class_colors
            self.display_batch(colors, trainer, pl_module, batch, "Test")
