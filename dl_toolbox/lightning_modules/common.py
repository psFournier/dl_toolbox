from argparse import ArgumentParser
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam
from torch.optim.lr_scheduler import MultiStepLR
import torch
import torchmetrics.functional as torchmetrics
from torchmetrics import MeanMetric, JaccardIndex
import matplotlib.pyplot as plt

import numpy as np
from dl_toolbox.networks import NetworkFactory
from dl_toolbox.torch_datasets.utils import aug_dict, anti_aug_dict




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
        
        #self.confmat = torch.zeros((self.num_classes, self.num_classes))
        if self.plot_calib: 
            self.conf_bins = torch.zeros(100, dtype=torch.float)
            self.acc_bins = torch.zeros(100, dtype=torch.float)
            self.count_bins = torch.zeros(100, dtype=torch.float)
    
    def configure_optimizers(self):

        self.optimizer = Adam(self.parameters(), lr=self.initial_lr)

        return self.optimizer
    
#    def predict_step(self, batch, batch_idx):
#        
#        inputs = batch['image']
#        logits = self.forward(inputs)
#        
#        if self.ttas:
#            for tta, reverse in self.ttas:
#                aux, _ = tta(img=inputs)
#                aux_logits = self.forward(aux)
#                tta_logits, _ = reverse(img=aux_logits)
#                logits = torch.stack([logits, tta_logits])
#            logits = logits.mean(dim=0)
#         
#        #probas = self.logits2probas(logits)
#        #confidences, preds = self.probas2confpreds(probas)
#        
#        return logits
    #
    #def validation_step(self, batch, batch_idx):
    #    
    #    inputs = batch['image']
    #    labels = batch['label']
    #    logits = self.forward(inputs)
    #    
    #    probas = self.logits2probas(logits)
    #    confidences, preds = self.probas2confpreds(probas)
    #    
    #    labels_flat = labels.cpu().flatten()
    #    preds_flat = preds.cpu().flatten()
    #    
    #    conf_mat = compute_conf_mat(
    #        labels_flat,
    #        preds_flat,
    #        self.num_classes,
    #        ignore_idx=None
    #    )
    #    self.confmat += conf_mat
#
    #    if self.plot_calib:
    #        conf_flat = confidences.cpu().flatten()
    #        acc_bins, conf_bins, count_bins = compute_calibration_bins(
    #            torch.linspace(0, 1, 100 + 1),
    #            labels_flat,
    #            conf_flat,
    #            preds_flat,
    #            ignore_idx=None
    #        )
    #        self.count_bins += count_bins
    #        self.acc_bins += torch.mul(acc_bins, count_bins)
    #        self.conf_bins += torch.mul(conf_bins, count_bins)
#
    #    return logits

#    def validation_epoch_end(self, outs):
#
#        cm_array = self.confmat.numpy()
#        m = np.nan
#        with np.errstate(divide='ignore', invalid='ignore'):
#            ious = np.diag(cm_array) / (cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array))
#        if self.ignore_zero: ious = ious[1:]
#        mIou = np.nansum(ious) / (np.logical_not(np.isnan(ious))).sum()
#        self.log('Val_miou', mIou.astype(float))
#        
#        self.trainer.logger.experiment.add_figure(
#            "Class IoUs",
#            plot_ious(
#                ious,
#                class_names=self.class_names
#            ),
#            global_step=self.trainer.global_step
#        )
#        self.trainer.logger.experiment.add_figure(
#            "Precision matrix", 
#            plot_confusion_matrix(
#                self.confmat,
#                class_names=self.class_names,
#                norm='precision'
#            ), 
#            global_step=self.trainer.global_step
#        )
#        self.trainer.logger.experiment.add_figure(
#            "Recall matrix", 
#            plot_confusion_matrix(
#                self.confmat,
#                class_names=self.class_names,
#                norm='recall'
#            ), 
#            global_step=self.trainer.global_step
#        )
#        
#        if self.plot_calib:
#            
#            self.acc_bins = torch.div(self.acc_bins, self.count_bins)
#            self.conf_bins = torch.div(self.conf_bins, self.count_bins)
#            self.trainer.logger.experiment.add_figure(
#                f"Calibration",
#                plot_calib(
#                    self.count_bins.numpy(),
#                    self.acc_bins.numpy(),
#                    self.conf_bins.numpy(),
#                    max_points=10000
#                ),
#                global_step=self.trainer.global_step
#            )
#
#    def on_train_epoch_end(self):
#        for param_group in self.optimizer.param_groups:
#            self.log(f'learning_rate', param_group['lr'])