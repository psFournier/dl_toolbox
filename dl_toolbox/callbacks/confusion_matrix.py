import pytorch_lightning as pl
import torch
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Necessary for imshow to run on machines with no graphical interface.
plt.switch_backend("agg")

# Taken from https://www.tensorflow.org/tensorboard/image_summaries
def plot_confusion_matrix(cm, class_names):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    figure = plt.figure(figsize=(15,15))
    plt.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    plt.title("Confusion matrix")
    plt.colorbar()
    tick_marks = np.arange(len(class_names))
    plt.xticks(tick_marks, class_names, rotation=90)
    plt.yticks(tick_marks, class_names)

    # Compute the labels from the normalized confusion matrix.
    # labels = np.around(cm.astype("float"), decimals=2)
    labels = np.around(cm.astype('float') * 100).astype('int')

    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        plt.text(j, i, labels[i, j], fontsize=10, horizontalalignment="center", color=color)

    plt.tight_layout()
    plt.ylabel("True label")
    plt.xlabel("Predicted label")
    return figure

def compute_conf_mat(labels, preds, num_classes, ignore_idx=None):

    if ignore_idx is not None:
        idx = labels != ignore_idx
        preds = preds[idx]
        labels = labels[idx]

    unique_mapping = (labels * num_classes + preds).to(torch.long)
    bins = torch.bincount(unique_mapping, minlength=num_classes**2)
    conf_mat = bins.reshape(num_classes, num_classes)

    return conf_mat
    

class ConfMatLogger(pl.Callback):

    def __init__(self, labels, freq, *args, **kwargs):

        super().__init__(*args, **kwargs)
        self.labels = labels
        self.freq = freq

    def on_fit_start(self, trainer, pl_module):

        self.conf_mat = ConfusionMatrix(
            num_classes=len(self.labels),
            normalize=None,
            compute_on_step=False
        )

    def on_validation_batch_end(
            self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx
    ):
        
        if trainer.current_epoch % self.freq == 0:
            
            batch = outputs['batch']
            labels = batch['mask'].cpu() # labels shape B,H,W, values in {0, C-1}
            preds = batch['preds'].cpu() 
            self.conf_mat(preds, labels)

    def on_validation_epoch_end(self, trainer, pl_module):
        
        if trainer.current_epoch % self.freq == 0:

            cm = self.conf_mat.compute().numpy()
            cm_recall = cm/np.sum(cm,axis=1, keepdims=True)
            cm_precision = cm/np.sum(cm,axis=0,keepdims=True)
            
            trainer.logger.experiment.add_figure(
                "Confusion matrix recall", 
                plot_confusion_matrix(cm_recall, class_names=self.labels), 
                global_step=trainer.global_step
            )
            trainer.logger.experiment.add_figure(
                "Confusion matrix precision", 
                plot_confusion_matrix(cm_precision, class_names=self.labels), 
                global_step=trainer.global_step
            )
