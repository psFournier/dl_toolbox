import pytorch_lightning as pl
import torch
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import itertools
from dl_toolbox.utils import plot_confusion_matrix, plot_ious
# Necessary for imshow to run on machines with no graphical interface.
plt.switch_backend("agg")


def compute_conf_mat(labels, preds, num_classes, ignore_idx=None):

    if ignore_idx is not None:
        idx = labels != ignore_idx
        preds = preds[idx]
        labels = labels[idx]

    unique_mapping = (labels * num_classes + preds).to(torch.long)
    bins = torch.bincount(unique_mapping, minlength=num_classes**2)
    conf_mat = bins.reshape(num_classes, num_classes)

    return conf_mat
    

class MetricsFromConfmat(pl.Callback):

    def __init__(
        self,
        num_classes,
        class_names,
        ignore_idx=None,
        plot_iou=False,
        plot_confmat=True,
        *args,
        **kwargs
    ):

        super().__init__(*args, **kwargs)
        self.num_classes = num_classes
        self.class_names = class_names
        self.plot_iou = plot_iou
        self.plot_confmat = plot_confmat
        self.ignore_idx = ignore_idx

    def on_validation_epoch_start(self, trainer, pl_module):
        
        self.confmat = torch.zeros((self.num_classes, self.num_classes))
        
    def on_test_epoch_start(self, trainer, pl_module):
        
        self.confmat = torch.zeros((self.num_classes, self.num_classes))    
        
    def compute_conf_mat(self, outputs, pl_module, batch):
        
        logits = outputs.cpu()
        probas = pl_module.logits2probas(logits)
        confidences, preds = pl_module.probas2confpreds(probas)
        labels = batch['label'].cpu()
        conf_mat = compute_conf_mat(
            labels.flatten(),
            preds.flatten(),
            self.num_classes,
            ignore_idx=self.ignore_idx
        )
        
        return conf_mat

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        
        self.confmat += self.compute_conf_mat(
            outputs,
            pl_module,
            batch
        )
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        
        self.confmat += self.compute_conf_mat(
            outputs,
            pl_module,
            batch
        )
        
    def log_from_confmat(self, trainer, pl_module):
        
        cm_array = self.confmat.numpy()
        m = np.nan
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ious = np.diag(cm_array) / (cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array))
            f1s = 2 * np.diag(cm_array) / (cm_array.sum(0) + cm_array.sum(1))
            idx = np.array(range(self.num_classes)) != self.ignore_idx
            ious = ious[idx]
            f1s = f1s[idx]
            #precisions = np.diag(cm_array) / cm_array.sum(0)
            #recalls = np.diag(cm_array) / cm_array.sum(1)
            
        mIou = np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()
        mf1 = np.nansum(f1s) / (np.logical_not(np.isnan(f1s))).sum()
        self.log('miou/val', mIou.astype(float))
        self.log('mf1/val', mf1.astype(float))
        
        if self.plot_iou:
            trainer.logger.experiment.add_figure(
                "Class IoUs",
                plot_ious(
                    ious,
                    class_names=self.class_names
                ),
                global_step=trainer.global_step
            )
            
        if self.plot_confmat:
            trainer.logger.experiment.add_figure(
                "Precision matrix", 
                plot_confusion_matrix(
                    self.confmat,
                    class_names=self.class_names,
                    norm='precision'
                ), 
                global_step=trainer.global_step
            )
            trainer.logger.experiment.add_figure(
                "Recall matrix", 
                plot_confusion_matrix(
                    self.confmat,
                    class_names=self.class_names,
                    norm='recall'
                ), 
                global_step=trainer.global_step
            )
            
    def on_validation_epoch_end(self, trainer, pl_module):
        
        self.log_from_confmat(trainer, pl_module)
        
    def on_test_epoch_end(self, trainer, pl_module):
        
        self.log_from_confmat(trainer, pl_module)
