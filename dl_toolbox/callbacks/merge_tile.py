import pytorch_lightning as pl
import torch
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import rasterio.windows as windows 
from functools import partial

def dist_to_edge(i, j, h, w):

    mi = np.minimum(i+1, h-i)
    mj = np.minimum(j+1, w-j)
    return np.minimum(mi, mj)


class MergePretiledRasterPreds(pl.Callback):

    def __init__(self, dataset, mode, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.data_src = dataset.data_src
        self.window = self.data_src.zone
        assert isinstance(self.window, windows.Window)
        crop_size = dataset.crop_size
        if mode=='linear':
            crop_mask = np.fromfunction(
                function=partial(
                    dist_to_edge,
                    h=crop_size,
                    w=crop_size
                ),
                shape=(crop_size, crop_size),
                dtype=int
            )
            self.crop_mask = torch.from_numpy(crop_mask).float()
        elif mode=='constant':
            self.crop_mask = torch.ones((crop_size, crop_size)).float()
        
    def on_test_epoch_start(self, trainer, pl_module):
        
        h, w = self.window.height, self.window.width
        self.merged = torch.zeros((pl_module.num_classes, h, w))
        self.weights = torch.zeros((h, w))
        
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx):
        
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
                :,
                row_off:row_off+crop.height,
                col_off:col_off+crop.width
            ] += self.crop_mask
        
    def on_test_epoch_end(self, trainer, pl_module):                
        
        probas = torch.div(self.merged, self.weights)
        confs, preds = pl_module.probas2confpreds(probas.unsqueeze(dim=0))
        preds = preds.squeeze().numpy()
        
        labels = self.data_src.read_label(self.window)
        conf_mat = compute_conf_mat(
            labels.flatten(),
            preds.flatten(),
            pl_module.num_classes,
            ignore_idx=None
        )
        
        
        
    
    def on_predict_epoch_end(self, trainer, pl_module):
        
        probas = torch.div(self.merged, self.weights)

        

        
    def log_from_confmat(self, trainer, pl_module):
        
        cm_array = self.confmat.numpy()
        m = np.nan
        
        with np.errstate(divide='ignore', invalid='ignore'):
            ious = np.diag(cm_array) / (cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array))
            f1s = 2 * np.diag(cm_array) / (cm_array.sum(0) + cm_array.sum(1))
            #precisions = np.diag(cm_array) / cm_array.sum(0)
            #recalls = np.diag(cm_array) / cm_array.sum(1)
            
        mIou = np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()
        mf1 = np.nansum(f1s) / (np.logical_not(np.isnan(f1s))).sum()
        self.log('Val_miou', mIou.astype(float))
        self.log('Val_mf1', mf1.astype(float))
        
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
