import pytorch_lightning as pl
import torch
from torchmetrics import CalibrationError
import matplotlib.pyplot as plt
import numpy as np
import itertools
from torchmetrics.functional.classification.calibration_error import _binning_bucketize
from torchmetrics.utilities.data import dim_zero_cat
import seaborn as sns

# Necessary for imshow to run on machines with no graphical interface.
plt.switch_backend("agg")

def plot_hist_cls(class_distrib):
    
    figure = plt.figure(figsize=(8,8))
    plt.bar(
        x=range(len(class_distrib)),
        height=class_distrib
    )
    plt.xlabel("Class")
    plt.ylabel("Counts")

    return figure

class ClassDistribLogger(pl.Callback):

    def __init__(self, freq, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.freq = freq

    def on_fit_start(self, trainer, pl_module):

        self.class_distrib_train = [0]*pl_module.num_classes
        self.class_distrib_val = [0]*pl_module.num_classes
        
    def on_train_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        if trainer.current_epoch % self.freq == 0:
            labels = batch['mask'].cpu().flatten()
            for i in range(pl_module.num_classes):
                cls_filter = torch.nonzero(labels==i, as_tuple=True)
                self.class_distrib_train[i] += len(cls_filter[0])

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        if trainer.current_epoch % self.freq == 0:
            labels = batch['mask'].cpu().flatten()
            for i in range(pl_module.num_classes):
                cls_filter = torch.nonzero(labels == i, as_tuple=True)
                self.class_distrib_val[i] += len(cls_filter[0])
    
    def on_train_epoch_end(self, trainer, pl_module):
        
        if trainer.current_epoch % self.freq == 0:
            fig_distrib = plot_hist_cls(self.class_distrib_train)
            self.class_distrib_train = [0]*pl_module.num_classes
            trainer.logger.experiment.add_figure(
                f"Train set class distribution",
                fig_distrib,
                global_step=trainer.global_step
            )
            
    def on_validation_epoch_end(self, trainer, pl_module):
        
        if trainer.current_epoch % self.freq == 0:
            fig_distrib = plot_hist_cls(self.class_distrib_val)
            self.class_distrib_val = [0]*pl_module.num_classes
            trainer.logger.experiment.add_figure(
                f"Val set class distribution",
                fig_distrib,
                global_step=trainer.global_step
            )

