from argparse import ArgumentParser
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch
import torchmetrics.functional as torchmetrics
from torchmetrics import MeanMetric, JaccardIndex
import matplotlib.pyplot as plt

from dl_toolbox.callbacks import plot_confusion_matrix, plot_calib, compute_calibration_bins, compute_conf_mat, log_batch_images
import numpy as np
from dl_toolbox.networks import NetworkFactory
from dl_toolbox.torch_datasets.utils import aug_dict, anti_aug_dict


def calc_miou(cm_array):
    m = np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        ious = np.diag(cm_array) / (cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array))
    m = np.nansum(ious[:-1]) / (np.logical_not(np.isnan(ious[:-1]))).sum()
    return m.astype(float), ious[:-1]  

def plot_ious(ious, class_names, baseline=None):
    
    y_pos = np.arange(len(ious)) #- .2
    #y_pos_2 = np.arange(len(classes)-1) + .2
    ious = [round(i, 4) for i in ious]
    if baseline is None:
        baseline = [0.]*len(ious)
    diff = [round(i-b, 4) for i,b in zip(ious, baseline)]

    bar_width = .8

    fig, axs = plt.subplots(1,2, width_ratios=[2, 1])
    #fig.subplots_adjust(wspace=0, top=1, right=1, left=0, bottom=0)

    axs[1].barh(y_pos[::-1], ious, bar_width,  align='center', alpha=0.4)
    #axs[1].barh(y_pos_2[::-1], baseline, bar_width,  align='center', alpha=0.4, color=lut_colors[:-1])

    axs[1].set_yticks([])
    axs[1].set_xlabel('IoU')
    axs[1].set_ylim(0 - .4, (len(ious)) + .4)
    axs[1].set_xlim(0, 1)

    cell_text = list(zip(ious, baseline, diff))
    c = ['r' if d<0 else 'g' for d in diff]
    cellColours = list(zip(['white']*12, ['white']*12, c))

    column_labels = ['iou', 'baseline', 'difference']

    axs[0].axis('off')

    the_table = axs[0].table(cellText=cell_text,
                             cellColours=cellColours,
                         rowLabels=class_names,
                         colLabels=column_labels,
                         bbox=[0.4, 0.0, 0.6, 1.0])

    #the_table.auto_set_font_size(False)
    #the_table.set_fontsize(12)
    fig.tight_layout()
    
    return fig

class BaseModule(pl.LightningModule):

    # Validation step common to all modules if possible

    def __init__(self,
                 initial_lr,
                 class_names,
                 ttas,
                 plot_calib,
                 *args,
                 **kwargs):

        super().__init__()
        self.net_factory = NetworkFactory()
        self.initial_lr = initial_lr
        self.class_names = class_names
        self.ttas = [(aug_dict[t](p=1), anti_aug_dict[t](p=1)) for t in ttas]
        self.plot_calib = plot_calib
        
    def on_validation_epoch_start(self):
        
        self.confmat = torch.zeros((self.num_classes, self.num_classes))
        if self.plot_calib: 
            self.conf_bins = torch.zeros(100, dtype=torch.float)
            self.acc_bins = torch.zeros(100, dtype=torch.float)
            self.count_bins = torch.zeros(100, dtype=torch.float)
    
    def configure_optimizers(self):

        self.optimizer = Adam(self.parameters(), lr=self.initial_lr)

        return self.optimizer
    
    def predict_step(self, batch, batch_idx):
        
        inputs = batch['image']
        logits = self.forward(inputs)
        
        if self.ttas:
            for tta, reverse in self.ttas:
                aux, _ = tta(img=inputs)
                aux_logits = self.forward(aux)
                tta_logits, _ = reverse(img=aux_logits)
                logits = torch.stack([logits, tta_logits])
            logits = logits.mean(dim=0)
         
        #probas = self.logits2probas(logits)
        #confidences, preds = self.probas2confpreds(probas)
        
        return logits
    
    def validation_step(self, batch, batch_idx):
        
        inputs = batch['image']
        labels = batch['label']
        logits = self.forward(inputs)
        
        probas = self.logits2probas(logits)
        confidences, preds = self.probas2confpreds(probas)
        
        labels_flat = labels.cpu().flatten()
        preds_flat = preds.cpu().flatten()
        
        conf_mat = compute_conf_mat(
            labels_flat,
            preds_flat,
            self.num_classes,
            ignore_idx=None
        )
        self.confmat += conf_mat

        if self.plot_calib:
            conf_flat = confidences.cpu().flatten()
            acc_bins, conf_bins, count_bins = compute_calibration_bins(
                torch.linspace(0, 1, 100 + 1),
                labels_flat,
                conf_flat,
                preds_flat,
                ignore_idx=None
            )
            self.count_bins += count_bins
            self.acc_bins += torch.mul(acc_bins, count_bins)
            self.conf_bins += torch.mul(conf_bins, count_bins)

        return logits

    def validation_epoch_end(self, outs):

        cm_array = self.confmat.numpy()
        m = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            ious = np.diag(cm_array) / (cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array))
        if self.ignore_zero: ious = ious[1:]
        mIou = np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()
        self.log('Val_miou', mIou.astype(float))
        
        self.trainer.logger.experiment.add_figure(
            "Class IoUs",
            plot_ious(
                ious,
                class_names=self.class_names
            ),
            global_step=self.trainer.global_step
        )
        self.trainer.logger.experiment.add_figure(
            "Precision matrix", 
            plot_confusion_matrix(
                self.confmat,
                class_names=self.class_names,
                norm='precision'
            ), 
            global_step=self.trainer.global_step
        )
        self.trainer.logger.experiment.add_figure(
            "Recall matrix", 
            plot_confusion_matrix(
                self.confmat,
                class_names=self.class_names,
                norm='recall'
            ), 
            global_step=self.trainer.global_step
        )
        
        if self.plot_calib:
            
            self.acc_bins = torch.div(self.acc_bins, self.count_bins)
            self.conf_bins = torch.div(self.conf_bins, self.count_bins)
            self.trainer.logger.experiment.add_figure(
                f"Calibration",
                plot_calib(
                    self.count_bins.numpy(),
                    self.acc_bins.numpy(),
                    self.conf_bins.numpy(),
                    max_points=10000
                ),
                global_step=self.trainer.global_step
            )

    def on_train_epoch_end(self):
        for param_group in self.optimizer.param_groups:
            self.log(f'learning_rate', param_group['lr'])