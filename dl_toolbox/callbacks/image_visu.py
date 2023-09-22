# Third-party libraries
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn

from dl_toolbox.utils import labels_to_rgb


class SegmentationImagesVisualisation(pl.Callback):
    """Generate images based on classifier predictions and log a batch to predefined logger.

    .. warning:: This callback supports only tensorboard right now

    """

    NB_COL: int = 8

    def __init__(self, freq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq

    def display_batch(self, colors, trainer, module, batch, prefix):
        x = batch["image"]
        logits = module.forward(x).cpu()
        img = x.cpu()[:, :3, ...]
        label = batch["label"].cpu()
        label_is_img = (label is not None and label.ndim==3)
        if label_is_img:
            labels_rgb = labels_to_rgb(label, colors=colors).transpose((0, 3, 1, 2))
            labels_rgb = torch.from_numpy(labels_rgb).float()
        prob = module.logits2probas(logits)
        _, pred = module.probas2confpreds(prob)
        pred_is_img = (pred.ndim==3)
        if pred_is_img:
            pred_rgb = labels_to_rgb(pred, colors=colors).transpose((0, 3, 1, 2))
            pred_rgb = torch.from_numpy(pred_rgb).float()

        # Number of grids to log depends on the batch size
        quotient, remainder = divmod(img.shape[0], self.NB_COL)
        nb_grids = quotient + int(remainder > 0)

        for idx in range(nb_grids):
            start = self.NB_COL * idx
            if start + self.NB_COL <= img.shape[0]:
                end = start + self.NB_COL
            else:
                end = start + remainder

            img_grid = torchvision.utils.make_grid(
                img[start:end, :, :, :], padding=10, normalize=True
            )
            grids = [img_grid]
            if label_is_img:
                mask_grid = torchvision.utils.make_grid(
                    labels_rgb[start:end, :, :, :], padding=10, normalize=True
                )
                grids.append(mask_grid)
            if pred_is_img:
                pred_grid = torchvision.utils.make_grid(
                    pred_rgb[start:end, :, :, :], padding=10, normalize=True
                )
                grids.append(pred_grid)

            final_grid = torch.cat(grids, dim=1)

            trainer.logger.experiment.add_image(
                f"Images/{prefix}_batch0_part_{idx}",
                final_grid,
                global_step=trainer.global_step,
            )

            break

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.current_epoch % self.freq == 0 and batch_idx <= 1:
            self.display_batch(
                trainer.datamodule.class_colors, trainer, pl_module, batch["sup"], prefix="Train"
            )

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        """Called when the validation batch ends."""

        if trainer.current_epoch % self.freq == 0 and batch_idx <= 1:
            self.display_batch(
                trainer.datamodule.class_colors, trainer, pl_module, batch, prefix="Val"
            )
