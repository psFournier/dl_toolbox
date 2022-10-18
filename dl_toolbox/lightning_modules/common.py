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
from dl_toolbox.callbacks import plot_confusion_matrix, plot_calib, compute_calibration_bins, compute_conf_mat

from dl_toolbox.lightning_modules.utils import *
import numpy as np

class BaseModule(pl.LightningModule):

    # Validation step common to all modules if possible

    def __init__(self,
                 num_classes,
                 weights,
                 *args,
                 **kwargs):

        super().__init__()

        self.num_classes = num_classes
        self.out_channels = self.num_classes
        self.weights = list(weights) if len(weights)>0 else [1]*self.num_classes

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_classes", type=int)
        parser.add_argument("--weights", type=float, nargs="+", default=())
                            
        return parser
    
    def configure_optimizers(self):

        self.optimizer = Adam(self.parameters(), lr=self.initial_lr)
        scheduler = MultiStepLR(
            self.optimizer,
            milestones=self.lr_milestones,
            gamma=0.2
        )

        return [self.optimizer], [scheduler]

    def _compute_probas(self, logits):

        return logits.softmax(dim=1)
    
    def _compute_conf_preds(self, probas):
        
        return torch.max(probas, dim=1)

    def validation_step(self, batch, batch_idx):

        inputs = batch['image']
        labels = batch['mask']
        logits = self.forward(inputs)
        probas = self._compute_probas(logits)
        confidences, preds = self._compute_conf_preds(probas)

        ignore_idx = None
        stat_scores = torchmetrics.stat_scores(
            preds,
            labels,
            ignore_index=ignore_idx,
            mdmc_reduce='global',
            reduce='macro',
            num_classes=self.num_classes
        )
        
        conf_mat = compute_conf_mat(
            labels.flatten(),
            preds.flatten(),
            self.num_classes,
            ignore_idx=None
        )
        
        acc_bins, conf_bins, count_bins = compute_calibration_bins(
            torch.linspace(0, 1, 100 + 1).to(self.device),
            labels.flatten(),
            confidences.flatten(),
            preds.flatten(),
            ignore_idx=ignore_idx
        )
        

        return {'acc_bins': acc_bins.detach(),
                'conf_bins': conf_bins.detach(),
                'count_bins': count_bins.detach(),
                'stat_scores': stat_scores.detach(),
                'conf_mat': conf_mat.detach(),
                'logits': logits.detach()
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

        cm = torch.stack(conf_mats, dim=0).sum(dim=0).cpu()
        
        #sum_col = torch.sum(cm,dim=1, keepdim=True)
        #sum_lin = torch.sum(cm,dim=0, keepdim=True)
        #if self.ignore_index >= 0: sum_lin -= cm[self.ignore_index,:]
        #cm_recall = torch.nan_to_num(cm/sum_col, nan=0., posinf=0., neginf=0.)
        #cm_precision = torch.nan_to_num(cm/sum_lin, nan=0., posinf=0., neginf=0.)
        #cm_recall = np.divide(cm, sum_col, out=np.zeros_like(cm), where=sum_col!=0)
        #cm_precision = np.divide(cm, sum_lin, out=np.zeros_like(cm), where=sum_lin!=0)
        
        self.trainer.logger.experiment.add_figure(
            "Confusion matrices", 
            plot_confusion_matrix(cm, class_names=[str(i) for i in range(num_classes)]), 
            global_step=self.trainer.global_step
        )


        count_bins = torch.stack([out['count_bins'] for out in outs])
        conf_bins = torch.stack([out['conf_bins'] for out in outs])
        acc_bins = torch.stack([out['acc_bins'] for out in outs])
        
        counts = torch.sum(count_bins, dim=0)
        accs = torch.sum(torch.mul(acc_bins, count_bins), dim=0)
        accs = torch.div(accs, counts)
        confs = torch.sum(torch.mul(conf_bins, count_bins), dim=0)
        confs = torch.div(confs, counts)

        figure = plot_calib(
            counts.cpu().numpy(),
            accs.cpu().numpy(),
            confs.cpu().numpy(),
            max_points=10000
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
