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

def plot_reliability_diagram(acc_bin, conf_bin):
    """

    """
    figure = plt.figure(figsize=(8, 8))
    plt.plot([0,1], [0,1], "k:", label="Perfectly calibrated")
    plt.plot(conf_bin, acc_bin, "s-", label="Model")
    plt.xlabel("Confidence")
    plt.ylabel("Accuracy")
    plt.legend(loc="lower right")
    plt.title("Calibration curve")
    plt.show()

    return figure

def plot_calib(count_bins, acc_bins, conf_bins, max_points):

    acc, conf = [], []
    total_count = torch.sum(count_bins)
    k = min(max_points / total_count, 1)
    for i,c in enumerate(count_bins):
        c = int(c * k)
        if c > 0:
            for _ in range(int(c)):
                acc.append(acc_bins[i])
                conf.append(conf_bins[i])

    #fig = plt.figure()
    #plt.scatter(x=np.asarray(conf), y=np.asarray(acc))
    g = sns.jointplot(
        x=np.asarray(conf),
        y=np.asarray(acc),
        kind='kde',
        fill=True,
        color='b',
        xlim=(None, 1),
        ylim=(None, 1)
    )
    sns.lineplot(x=[0,1], y=[0,1], ax=g.ax_joint, dashes=True)
    g.ax_joint.set_xlabel('average confidence')
    g.ax_joint.set_ylabel('average accuracy')

    return g.figure

def compute_calibration_bins(bin_boundaries, labels, confs, preds):

    """
    All inputs must be flattened torch tensors.
    """
      
    accus = preds.eq(labels).float()

    indices = torch.bucketize(confs, bin_boundaries) - 1
    n_bins = len(bin_boundaries) - 1

    count_bins = torch.zeros(n_bins, dtype=confs.dtype)
    count_bins.scatter_add_(dim=0, index=indices, src=torch.ones_like(confs))
    
    conf_bins = torch.zeros(n_bins, dtype=confs.dtype)
    conf_bins.scatter_add_(dim=0, index=indices, src=confs)
    conf_bins = torch.nan_to_num(conf_bins / count_bins)

    acc_bins = torch.zeros(n_bins, dtype=accus.dtype)
    acc_bins.scatter_add_(dim=0, index=indices, src=accus)
    acc_bins = torch.nan_to_num(acc_bins / count_bins)

    return acc_bins, conf_bins, count_bins
   

class CalibrationLogger(pl.Callback):

    def __init__(self, freq, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.freq = freq

    def on_fit_start(self, trainer, pl_module):

        self.n_bins = 100
        self.bin_boundaries = torch.linspace(0, 1, self.n_bins + 1)
        self.acc_bins, self.conf_bins, self.count_bins = [], [], []
        for i in range(pl_module.num_classes):
            if i != pl_module.ignore_index:
                self.acc_bins.append(torch.zeros(self.n_bins))
                self.conf_bins.append(torch.zeros(self.n_bins))
                self.count_bins.append(torch.zeros(self.n_bins))
        self.nb_step = 0

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):

        batch = outputs['batch']

        if trainer.current_epoch % self.freq == 0:
            
            labels = batch['mask'].cpu().flatten()
            
            for i in range(pl_module.num_classes):
                cls_filter = torch.nonzero(labels == i, as_tuple=True)
                if i != pl_module.ignore_index:                    
                    cls_labels = labels[cls_filter]
                    cls_confs = batch['confs'].cpu().flatten()[cls_filter]
                    cls_preds = batch['preds'].cpu().flatten()[cls_filter]
                    acc_bins, conf_bins, count_bins = compute_calibration_bins(
                        self.bin_boundaries,
                        cls_labels,
                        cls_confs,
                        cls_preds
                    )
                    self.acc_bins[i] += acc_bins
                    self.conf_bins[i] += conf_bins
                    self.count_bins[i] += count_bins

            self.nb_step += 1
            
    def on_validation_epoch_end(self, trainer, pl_module):
        
        if trainer.current_epoch % self.freq == 0:
            
            acc_bins = torch.zeros(self.n_bins)
            conf_bins = torch.zeros(self.n_bins)
            count_bins = torch.zeros(self.n_bins)
            tot_steps = 0

            for i in range(pl_module.num_classes):
                if i != pl_module.ignore_index:
                    
                    acc_bins += self.acc_bins[i]
                    self.acc_bins[i] /= self.nb_step
                    conf_bins += self.conf_bins[i]
                    self.conf_bins[i] /= self.nb_step
                    count_bins += self.count_bins[i]
                    tot_steps += self.nb_step

                    figure = plot_calib(
                        self.count_bins[i],
                        self.acc_bins[i],
                        self.conf_bins[i],
                        max_points=10000
                    )
                    trainer.logger.experiment.add_figure(
                        f"Calibration for class {i}",
                        figure,
                        global_step=trainer.global_step
                    )
                    self.acc_bins[i] = torch.zeros(self.n_bins)
                    self.conf_bins[i] = torch.zeros(self.n_bins)
                    self.count_bins[i] = torch.zeros(self.n_bins)

            self.nb_step = 0

