# Third-party libraries
import pytorch_lightning as pl
import torch
import torchvision
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.utilities import rank_zero_warn
import numpy as np
from dl_toolbox.utils import labels_to_rgb


class SegmentationImagesVisualisation(pl.Callback):
    """Generate images based on classifier predictions and log a batch to predefined logger.

    .. warning:: This callback supports only tensorboard right now

    """

    NB_COL: int = 8

    def __init__(self, freq, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq

    def display_batch(self, colors, trainer, batch, prefix):
        img = batch["image"].cpu()[:, :3, ...]
        if batch["label"] is not None:
            labels = batch["label"].cpu()
            labels_rgb = labels_to_rgb(labels, colors=colors).transpose((0, 3, 1, 2))
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

            img_grid = torchvision.utils.make_grid(
                img[start:end, :, :, :], padding=10, normalize=True
            )
            grids = [img_grid]
            if batch["label"] is not None:
                mask_grid = torchvision.utils.make_grid(
                    np_labels_rgb[start:end, :, :, :], padding=10, normalize=True
                )
                grids.append(mask_grid)

            final_grid = torch.cat(grids, dim=1)

            trainer.logger.experiment.add_image(
                f"Images/{prefix}_batch0_part_{idx}",
                final_grid,
                global_step=trainer.global_step,
            )

            break

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx) -> None:
        if trainer.current_epoch % self.freq == 0 and batch_idx == 0:
            self.display_batch(
                trainer.datamodule.class_colors, trainer, batch["sup"], prefix="Train"
            )

    def on_validation_batch_end(
        self, trainer: pl.Trainer, pl_module, outputs, batch, batch_idx
    ) -> None:
        """Called when the validation batch ends."""

        if trainer.current_epoch % self.freq == 0 and batch_idx == 0:
            self.display_batch(
                trainer.datamodule.class_colors, trainer, batch, prefix="Val"
            )
