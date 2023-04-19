import pytorch_lightning as pl
import torch
from torchmetrics import ConfusionMatrix
import matplotlib.pyplot as plt
import numpy as np
import itertools

# Taken from https://www.tensorflow.org/tensorboard/image_summaries
def plot_confusion_matrix(cm, class_names, norm):
    """
    Returns a matplotlib figure containing the plotted confusion matrix.

    Args:
      cm (array, shape = [n, n]): a confusion matrix of integer classes
      class_names (array, shape = [n]): String names of the integer classes
    """
    tick_marks = np.arange(len(class_names))
    
    fig, ax = plt.subplots(1, 1)
    
    if norm=='recall':

        sum_col = torch.sum(cm,dim=1, keepdim=True)
        cm = torch.nan_to_num(cm/sum_col, nan=0., posinf=0., neginf=0.).numpy()
        ax.set_title("Recall matrix")
    
    elif norm=='precision':

        sum_lin = torch.sum(cm,dim=0, keepdim=True)
        cm = torch.nan_to_num(cm/sum_lin, nan=0., posinf=0., neginf=0.).numpy()
        ax.set_title("Precision matrix")

    ax.imshow(cm, interpolation="nearest", cmap=plt.cm.Blues)
    ax.set_xticks(tick_marks, class_names, rotation=90, fontsize=4)
    ax.set_yticks(tick_marks, class_names, fontsize=4)
    # Compute the labels from the normalized confusion matrix.
    labels = np.around(cm.astype('float') * 100).astype('int')
    # Use white text if squares are dark; otherwise black.
    threshold = cm.max() / 2.0
    for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
        color = "white" if cm[i, j] > threshold else "black"
        ax.text(j, i, labels[i, j], fontsize=4, horizontalalignment="center", color=color)
    ax.set_ylabel("True label", fontsize=4)
    ax.set_xlabel("Predicted label", fontsize=4)
    fig.tight_layout()
    
    return fig

def plot_ious(ious, class_names, baseline=None):
    
    num_class = len(ious)
    
    y_pos = np.arange(num_class) #- .2
    #y_pos_2 = np.arange(len(classes)-1) + .2
    ious = [round(i, 4) for i in ious]
    if baseline is None:
        baseline = [0.]*num_class
    diff = [round(i-b, 4) for i,b in zip(ious, baseline)]

    bar_width = .8

    fig, axs = plt.subplots(1,2, width_ratios=[2, 1])
    #fig.subplots_adjust(wspace=0, top=1, right=1, left=0, bottom=0)

    axs[1].barh(y_pos[::-1], ious, bar_width,  align='center', alpha=0.4)
    #axs[1].barh(y_pos_2[::-1], baseline, bar_width,  align='center', alpha=0.4, color=lut_colors[:-1])

    axs[1].set_yticks([])
    axs[1].set_xlabel('IoU')
    axs[1].set_ylim(0 - .4, (num_class) + .4)
    axs[1].set_xlim(0, 1)

    cell_text = list(zip(ious, baseline, diff))
    c = ['r' if d<0 else 'g' for d in diff]
    cellColours = list(zip(['white']*num_class, ['white']*num_class, c))

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