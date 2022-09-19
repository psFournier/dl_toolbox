from argparse import ArgumentParser
import segmentation_models_pytorch as smp
import torch.nn as nn
import pytorch_lightning as pl
from torch.optim import Adam, SGD
from torch.optim.lr_scheduler import MultiStepLR, LambdaLR
import torch
import torchmetrics.functional as torchmetrics
from dl_toolbox.losses import DiceLoss
from copy import deepcopy
import torch.nn.functional as F
from dl_toolbox.callbacks import plot_confusion_matrix, plot_calib, compute_calibration_bins

from dl_toolbox.lightning_modules.utils import *
import numpy as np

class BaseModule(pl.LightningModule):

    # Validation step common to all modules if possible

    def __init__(self,
                 num_classes,
                 ignore_index,
                 weights,
                 *args,
                 **kwargs):

        super().__init__()

        self.num_classes = num_classes
        self.weights = list(weights) if len(weights)>0 else [1]*self.num_classes
        self.ignore_index = ignore_index

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--weights", type=float, nargs="+", default=())
        parser.add_argument("--ignore_index", type=int)

        return parser
    
    def configure_optimizers(self):

        self.optimizer = Adam(self.parameters(), lr=self.initial_lr)
        scheduler = MultiStepLR(
            self.optimizer,
            milestones=self.lr_milestones,
            gamma=0.2
        )

        return [self.optimizer], [scheduler]

    def validation_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['mask']
        logits = self.forward(inputs)
        probas = logits.softmax(dim=1)
        confidences, preds = torch.max(probas, dim=1)

        batch['probas'] = probas.detach()
        batch['confs'] = confidences.detach()
        batch['preds'] = preds.detach()
        batch['logits'] = logits.detach()
        batch['mask'] = labels.detach()
        batch['image'] = inputs.detach()

        stat_scores = torchmetrics.stat_scores(
            preds,
            labels,
            ignore_index=self.ignore_index if self.ignore_index >= 0 else None,
            mdmc_reduce='global',
            reduce='macro',
            num_classes=self.num_classes
        )

        unique_mapping = (labels.flatten() * self.num_classes + preds.flatten()).to(torch.long)
        bins = torch.bincount(unique_mapping, minlength=self.num_classes**2)
        conf_mat = bins.reshape(self.num_classes, self.num_classes)
        
        loss1 = self.loss1(logits, labels)
        #loss2 = self.loss2(logits, labels)
        loss2=0
        loss = loss1 + loss2
        self.log('Val_CE', loss1)
        self.log('Val_Dice', loss2)
        self.log('Val_loss', loss)

        return {'batch': batch,
                'stat_scores': stat_scores.detach(),
                'conf_mat': conf_mat.detach()
                }

    def validation_epoch_end(self, outs):
        
        stat_scores = [out['stat_scores'] for out in outs]

        class_stat_scores = torch.sum(torch.stack(stat_scores), dim=0)
        f1_sum = 0
        tp_sum = 0
        supp_sum = 0
        nc = 0
        num_classes = 2 if self.num_classes == 1 else self.num_classes
        for i in range(num_classes):
            if i != self.ignore_index:
                tp, fp, tn, fn, supp = class_stat_scores[i, :]
                if supp > 0:
                    nc += 1
                    f1 = tp / (tp + 0.5 * (fp + fn))
                    #self.log(f'Val_f1_{i}', f1)
                    f1_sum += f1
                    tp_sum += tp
                    supp_sum += supp
        
        self.log('Val_acc', tp_sum / supp_sum)
        self.log('Val_f1', f1_sum / nc) 

        conf_mats = [out['conf_mat'] for out in outs]

        cm = torch.stack(conf_mats, dim=0).sum(dim=0)
        cm = cm.cpu().numpy()
        cm_recall = cm/np.sum(cm,axis=1, keepdims=True)
        cm_precision = cm/np.sum(cm,axis=0,keepdims=True)
        
        self.trainer.logger.experiment.add_figure(
            "Confusion matrix recall", 
            plot_confusion_matrix(cm_recall, class_names=[str(i) for i in range(num_classes)]), 
            global_step=self.trainer.global_step
        )
        self.trainer.logger.experiment.add_figure(
            "Confusion matrix precision", 
            plot_confusion_matrix(cm_precision, class_names=[str(i) for i in range(num_classes)]), 
            global_step=self.trainer.global_step
        )

        labels = [out['batch']['mask'].flatten() for out in outs]
        preds = [out['batch']['preds'].flatten() for out in outs]
        confs = [out['batch']['confs'].flatten() for out in outs]

        acc_bins, conf_bins, count_bins = compute_calibration_bins(
            torch.linspace(0, 1, 100 + 1).to(self.device),
            torch.cat(labels, dim=0),
            torch.cat(confs, dim=0),
            torch.cat(preds, dim=0)
        )

        figure = plot_calib(
            count_bins.cpu().numpy(),
            acc_bins.cpu().numpy(),
            conf_bins.cpu().numpy(),
            max_points=1000
        )

        self.trainer.logger.experiment.add_figure(
            f"Calibration",
            figure,
            global_step=self.trainer.global_step
        )


    def on_train_epoch_end(self):
        for param_group in self.optimizer.param_groups:
            self.log(f'learning_rate', param_group['lr'])
            break
