# Third-party libraries
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn
import numpy as np

def display_batch(batch, visu_fn):

    nb_col = 8

    img = batch['image'].cpu()
    orig_img = batch['orig_image'].cpu()
    preds = batch['preds'].cpu()

    preds_rgb = visu_fn(preds).transpose((0,3,1,2))
    np_preds_rgb = torch.from_numpy(preds_rgb).float()

    if batch['mask'] is not None: 
        labels = batch['mask'].cpu()
        labels_rgb = visu_fn(labels).transpose((0,3,1,2))
        np_labels_rgb = torch.from_numpy(labels_rgb).float()

    # Number of grids to log depends on the batch size
    quotient, remainder = divmod(img.shape[0], nb_col)
    nb_grids = quotient + int(remainder > 0)

    final_grids = []
    for idx in range(nb_grids):

        start = nb_col * idx
        if start + nb_col <= img.shape[0]:
            end = start + nb_col
        else:
            end = start + remainder

        img_grid = torchvision.utils.make_grid(img[start:end, :, :, :], padding=10, normalize=True)
        orig_img_grid = torchvision.utils.make_grid(orig_img[start:end, :, :, :], padding=10, normalize=True)
        out_grid = torchvision.utils.make_grid(np_preds_rgb[start:end, :, :, :], padding=10, normalize=True)
        grids = [orig_img_grid, img_grid, out_grid]

        if batch['mask'] is not None:
            mask_grid = torchvision.utils.make_grid(np_labels_rgb[start:end, :, :, :], padding=10, normalize=True)
            grids.append(mask_grid)

        final_grids.append(torch.cat(grids, dim=1))

    return final_grids

def log_batch_images(batch, trainer, visu_fn, prefix):
    
    display_grids = display_batch(
        batch,
        visu_fn=visu_fn
    )
    for i, grid in enumerate(display_grids):
        trainer.logger.experiment.add_image(
            f'mages/{prefix}_batch_part_{i}',
            grid,
            global_step=trainer.global_step
        )

    

class SegmentationImagesVisualisation(pl.Callback):
    """Generate images based on classifier predictions and log a batch to predefined logger.

    .. warning:: This callback supports only tensorboard right now

    """

    NB_COL: int = 2

    def __init__(self, visu_fn, freq, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.visu_fn = visu_fn 
        self.freq = freq

#    def on_train_batch_end(
#            self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
#    ) -> None:

#        if trainer.current_epoch % self.freq == 0 and batch_idx == 0:
#
#            self.display_batch(
#                trainer,
#                outputs['batch'],
#                prefix='Train'
#        )
            
    def display_batch(self, trainer, batch, prefix):

        img = batch['image'].cpu()
        orig_img = batch['orig_image'].cpu()
        preds = batch['preds'].cpu()

        preds_rgb = self.visu_fn(preds).transpose((0,3,1,2))
        np_preds_rgb = torch.from_numpy(preds_rgb).float()

        if batch['mask'] is not None: 
            labels = batch['mask'].cpu()
            labels_rgb = self.visu_fn(labels).transpose((0,3,1,2))
            np_labels_rgb = torch.from_numpy(labels_rgb).float()

        # Number of grids to log depends on the batch size
        quotient, remainder = divmod(img.shape[0], self.NB_COL)
        nb_grids = quotient + int(remainder > 0)

        for idx in range(nb_grids):

            start = self.NB_COL * idx
            if start + self.NB_COL <= img.shape[0]:
                end = start + self.NB_COL
            else:
                end = start + remainder

            img_grid = torchvision.utils.make_grid(img[start:end, :, :, :], padding=10, normalize=True)
            orig_img_grid = torchvision.utils.make_grid(orig_img[start:end, :, :, :], padding=10, normalize=True)
            out_grid = torchvision.utils.make_grid(np_preds_rgb[start:end, :, :, :], padding=10, normalize=True)
            grids = [orig_img_grid, img_grid, out_grid]

            if batch['mask'] is not None:
                mask_grid = torchvision.utils.make_grid(np_labels_rgb[start:end, :, :, :], padding=10, normalize=True)
                grids.append(mask_grid)

            final_grid = torch.cat(grids, dim=1)

            trainer.logger.experiment.add_image(f'Images/{prefix}_batch_art_{idx}', final_grid, global_step=trainer.global_step)

    def on_validation_batch_end(
            self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ) -> None:
        """Called when the validation batch ends."""
 
        if trainer.current_epoch % self.freq == 0 and batch_idx == 0:
            self.display_batch(trainer, outputs['batch'], prefix='Val')

