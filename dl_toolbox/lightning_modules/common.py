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


def calc_miou(cm_array):
    m = np.nan
    with np.errstate(divide='ignore', invalid='ignore'):
        ious = np.diag(cm_array) / (cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array))
    m = np.nansum(ious[:-1]) / (np.logical_not(np.isnan(ious[:-1]))).sum()
    return m.astype(float), ious[:-1]  

def plot_ious(ious, class_names):
    
    y_pos = np.arange(len(ious)) #- .2
    #y_pos_2 = np.arange(len(classes)-1) + .2
    ious = [round(i, 4) for i in ious]
    baseline = [0.8004, 0.4739, 0.7078, 0.3995, 0.7818, 0.3213, 0.6537, 0.3127, 0.8016, 0.5109, 0.4520, 0.3315]
    diff = [round(i-b, 4) for i,b in zip(ious, baseline)]

    bar_width = .8

    fig, axs = plt.subplots(1,2, width_ratios=[2, 1])
    #fig.subplots_adjust(wspace=0, top=1, right=1, left=0, bottom=0)

    axs[1].barh(y_pos[::-1], ious, bar_width,  align='center', alpha=0.4)
    #axs[1].barh(y_pos_2[::-1], baseline, bar_width,  align='center', alpha=0.4, color=lut_colors[:-1])

    axs[1].set_yticks([])
    axs[1].set_xlabel('submission IoU')
    axs[1].set_ylim(0 - .4, (len(ious)) + .4)
    axs[1].set_xlim(0, 1)

    cell_text = list(zip(ious, baseline, diff))
    c = ['r' if d<0 else 'g' for d in diff]
    cellColours = list(zip(['white']*12, ['white']*12, c))

    column_labels = ['iou', 'baseline', 'difference']

    axs[0].axis('off')

    the_table = axs[0].table(cellText=cell_text,
                             cellColours=cellColours,
                         rowLabels=class_names[:-1],
                         colLabels=column_labels,
                         bbox=[0.4, 0.0, 0.6, 1.0])

    #the_table.auto_set_font_size(False)
    #the_table.set_fontsize(12)
    fig.tight_layout()
    
    return fig

class BaseModule(pl.LightningModule):

    # Validation step common to all modules if possible

    def __init__(self,
                 initial_lr=0.05,
                 final_lr=0.001,
                 lr_milestones=(0.5,0.9),
                 plot_calib=False,
                 class_names=None,
                 *args,
                 **kwargs):

        super().__init__()
        self.net_factory = NetworkFactory()
        self.initial_lr = initial_lr
        self.final_lr = final_lr
        self.lr_milestones = list(lr_milestones)
        self.plot_calib = plot_calib
        self.class_names = class_names if class_names else [str(i) for i in range(self.num_classes)]
        
        #self.val_metrics = JaccardIndex(
        #    num_classes=len(self.class_names),
        #    absent_score=1.0,
        #    reduction='elementwise_mean'
        #)
        #self.val_loss = MeanMetric()

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--initial_lr", type=float)
        parser.add_argument("--final_lr", type=float)
        parser.add_argument("--lr_milestones", nargs='+', type=float)
        parser.add_argument("--plot_calib", action='store_true')

        return parser
    
    def configure_optimizers(self):

        self.optimizer = Adam(self.parameters(), lr=self.initial_lr)
        scheduler = MultiStepLR(
            self.optimizer,
            milestones=self.lr_milestones,
            gamma=0.2
        )

        return [self.optimizer], [scheduler]
    
    def predict_step(self, batch, batch_idx):
        
        inputs = batch['image']
        logits = self.forward(inputs)
        probas = self._compute_probas(logits)
        confidences, preds = self._compute_conf_preds(probas)
        batch['preds'] = preds
        
        return batch
    
    def validation_step(self, batch, batch_idx):
        
        inputs = batch['image']
        labels = batch['mask']
        logits = self.forward(inputs)
        
        probas = self._compute_probas(logits)
        confidences, preds = self._compute_conf_preds(probas)
        #batch['preds'] = preds
        
        #self.val_loss.update(loss)
        #self.val_metrics(preds=preds, target=labels)
        
        #if self.trainer.current_epoch % 10 == 0 and batch_idx == 0:
        #    log_batch_images(
        #        batch,
        #        self.trainer,
        #        prefix='Val'
        #    )
        
        #stat_scores = torchmetrics.stat_scores(
        #    preds,
        #    labels,
        #    ignore_index=None,
        #    mdmc_reduce='global',
        #    reduce='micro',
        #    num_classes=self.num_classes
        #)
        
        labels_flat = labels.cpu().flatten()
        preds_flat = preds.cpu().flatten()
        
        conf_mat = compute_conf_mat(
            labels_flat,
            preds_flat,
            self.num_classes,
            ignore_idx=None
        )

        res = {
            #'stat_scores': stat_scores,
            'conf_mat': conf_mat,
            'logits': logits
        }

        if self.plot_calib:
            acc_bins, conf_bins, count_bins = compute_calibration_bins(
                torch.linspace(0, 1, 100 + 1).to(self.device),
                labels.flatten(),
                confidences.flatten(),
                preds.flatten(),
                ignore_idx=None
            )
            res['acc_bins'] = acc_bins
            res['conf_bins'] = conf_bins
            res['count_bins'] = count_bins

        return res

    def validation_epoch_end(self, outs):
        
        #self.val_epoch_loss = self.val_loss.compute()
        #self.val_epoch_metrics = self.val_metrics.compute()
        #self.log(
        #    "val_loss",
        #    self.val_epoch_loss,
        #    on_step=False,
        #    on_epoch=True,
        #    prog_bar=True,
        #    logger=True,
        #    rank_zero_only=True)
        #self.log(
        #    "val_miou",
        #    self.val_epoch_metrics,
        #    on_step=False,
        #    on_epoch=True,
        #    prog_bar=True,
        #    logger=True,
        #    rank_zero_only=True)
        #self.val_loss.reset()
        #self.val_metrics.reset()
        
        #stat_scores = [out['stat_scores'] for out in outs]
        #class_stat_scores = torch.sum(torch.stack(stat_scores), dim=0)
        #
        #f1_sum = 0
        #iou_sum = 0
        #tp_sum = 0
        #supp_sum = 0
        #nc = 0
        #for i in range(self.num_classes):
        #    if i != self.ignore_index:
        #        tp, fp, tn, fn, supp = class_stat_scores[i, :]
        #        if supp > 0:
        #            nc += 1
        #            f1 = tp / (tp + 0.5 * (fp + fn))
        #            iou = tp / (tp + fn + fp)
        #            #self.log(f'Val_f1_{i}', f1)
        #            f1_sum += f1
        #            iou_sum += iou
        #            tp_sum += tp
        #            supp_sum += supp
        #
        #self.log('Val_acc', tp_sum / supp_sum)
        #self.log('Val_f1', f1_sum / nc) 
        #self.log('Val_iou', iou_sum / nc)

        conf_mats = [out['conf_mat'] for out in outs]
        cm = torch.stack(conf_mats, dim=0).sum(dim=0)

        #sum_col = torch.sum(cm,dim=1, keepdim=True)
        #sum_lin = torch.sum(cm,dim=0, keepdim=True)
        #if self.ignore_index >= 0: sum_lin -= cm[self.ignore_index,:]
        #cm_recall = torch.nan_to_num(cm/sum_col, nan=0., posinf=0., neginf=0.)
        #cm_precision = torch.nan_to_num(cm/sum_lin, nan=0., posinf=0., neginf=0.)
        #cm_recall = np.divide(cm, sum_col, out=np.zeros_like(cm), where=sum_col!=0)
        #cm_precision = np.divide(cm, sum_lin, out=np.zeros_like(cm), where=sum_lin!=0)
              
        self.trainer.logger.experiment.add_figure(
            "Precision matrix", 
            plot_confusion_matrix(
                cm,
                class_names=self.class_names,
                norm='precision'
            ), 
            global_step=self.trainer.global_step
        )
        self.trainer.logger.experiment.add_figure(
            "Recall matrix", 
            plot_confusion_matrix(
                cm,
                class_names=self.class_names,
                norm='recall'
            ), 
            global_step=self.trainer.global_step
        )
        
        cm_array = cm.numpy()
        m = np.nan
        with np.errstate(divide='ignore', invalid='ignore'):
            ious = np.diag(cm_array) / (cm_array.sum(0) + cm_array.sum(1) - np.diag(cm_array))
        mIou = np.nansum(ious[1:]) / (np.logical_not(np.isnan(ious[1:]))).sum()
        self.log('Val_miou', mIou.astype(float))
        self.trainer.logger.experiment.add_figure(
            "Class IoUs",
            plot_ious(
                ious[1:],
                class_names=self.class_names
            ),
            global_step=self.trainer.global_step
        )

        if self.plot_calib:
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
