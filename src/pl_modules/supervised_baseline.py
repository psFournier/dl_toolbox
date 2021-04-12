from argparse import ArgumentParser
import segmentation_models_pytorch as smp

import numpy as np
import torch.nn as nn
import pytorch_lightning as pl
import torch.nn.functional as F
from torch import rot90, no_grad
from torch.optim import Adam
from torch.optim.lr_scheduler import (
    ExponentialLR,
    CyclicLR,
    MultiStepLR,
    CosineAnnealingLR,
    CosineAnnealingWarmRestarts,
)
import copy
import pytorch_lightning.metrics as M
from metrics import MyMetricCollection
from callbacks import ArrayValLogger, ConfMatLogger
from pytorch_toolbelt import losses
from common_utils.scheduler import get_scheduler


class SupervisedBaseline(pl.LightningModule):

    def __init__(self,
                 encoder,
                 pretrained,
                 in_channels,
                 num_classes,
                 inplaceBN,
                 learning_rate,
                 max_epochs,
                 *args,
                 **kwargs):

        super().__init__()

        network = smp.UnetPlusPlus(
            encoder_name=encoder,
            encoder_weights='imagenet' if pretrained else None,
            in_channels=in_channels,
            classes=num_classes,
            decoder_use_batchnorm='inplace' if inplaceBN else True
        )

        self.network = network
        self.save_hyperparameters()

        self.learning_rate = learning_rate
        self.max_epochs = max_epochs

        self.train_metrics = None
        self.val_metrics = None
        self.init_metrics(num_classes)

        self.callbacks = None
        self.init_callbacks(num_classes)

        self.loss = losses.JointLoss(
            nn.CrossEntropyLoss(),
            losses.DiceLoss(mode='multiclass', ignore_index=0)
        )

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--num_classes", type=int, default=2)
        parser.add_argument("--in_channels", type=int, default=3)
        parser.add_argument("--pretrained", action='store_true')
        parser.add_argument("--encoder", type=str, default='timm-regnetx_002')
        parser.add_argument("-lr", "--learning-rate", type=float, default=1e-3,
                            help="Initial learning rate")
        parser.add_argument("--inplaceBN", action='store_true' )

        return parser

    def init_metrics(self, num_classes):

        # Scalar metrics are separated because lightning can deal with logging
        # them automatically.
        accuracy = M.Accuracy(top_k=1, subset_accuracy=False)
        global_precision = M.Precision(
            num_classes=num_classes, mdmc_average="global", average="macro"
        )
        iou = M.IoU(num_classes=num_classes, reduction="elementwise_mean")

        scalar_metrics_dict = {
            "acc": accuracy,
            "global_precision": global_precision,
            "IoU": iou,
        }

        # Two things here:
        # 1. MyMetricCollection adds a prefix to metrics names, and should be
        # included in future versions of lightning
        # 2. The metric objects keep statistics during runs, and deepcopy should be
        # necessary to ensure these stats do not overlap
        self.train_metrics = MyMetricCollection(
            scalar_metrics_dict,
            "train_"
        )
        self.val_metrics = MyMetricCollection(
            copy.deepcopy(scalar_metrics_dict),
            "val_"
        )

    def init_callbacks(self, num_classes):

        # Non-scalar metrics are bundled in callbacks that deal with logging them
        per_class_precision = M.Precision(
            num_classes=num_classes, mdmc_average="global", average="none"
        )
        per_class_precision_logger = ArrayValLogger(
            array_metric=per_class_precision, name="per_class_precision"
        )
        per_class_F1 = M.F1(num_classes=num_classes, average="none")
        per_class_F1_logger = ArrayValLogger(
            array_metric=per_class_F1, name="per_class_F1"
        )
        cm = ConfMatLogger(num_classes=num_classes)

        self.callbacks = [
            cm, per_class_F1_logger, per_class_precision_logger
        ]

    def forward(self, x):

        return self.network(x)

    def configure_optimizers(self):

        optimizer = Adam(self.parameters(), lr=self.learning_rate)
        scheduler = MultiStepLR(
            optimizer,
            milestones=[
                int(self.max_epochs * 0.5),
                int(self.max_epochs * 0.7),
                int(self.max_epochs * 0.9)],
            gamma=0.3
        )

        return [optimizer], [scheduler]

    def training_step(self, batch, batch_idx):

        train_inputs, train_labels = batch
        outputs = self.network(train_inputs)
        train_loss = self.loss(outputs, train_labels)

        self.train_metrics(outputs.softmax(dim=1), train_labels)
        self.log("supervised_loss", train_loss)
        self.log_dict(self.train_metrics)

        return {"loss": train_loss}

    def validation_step(self, batch, batch_idx):

        val_inputs, val_labels = batch
        outputs = self.network(val_inputs)
        val_loss = self.loss(outputs, val_labels)

        self.val_metrics(outputs.softmax(dim=1), val_labels)
        self.log("val_sup_loss", val_loss)
        self.log_dict(self.val_metrics)