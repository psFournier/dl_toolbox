import torch
import torch.nn as nn
from dl_toolbox.lightning_modules import BaseModule
import dl_toolbox.augmentations as aug
from dl_toolbox.utils import TorchOneHot
from dl_toolbox.callbacks import log_batch_images


class CE(BaseModule):

    def __init__(self,
                 network,
                 weights,
                 mixup=0.,
                 ignore_index=-1,
                 *args,
                 **kwargs):

        super().__init__(*args, **kwargs)

        net_cls = self.net_factory.create(network)
        self.network = net_cls(*args, **kwargs)
        self.num_classes = self.network.out_channels
        out_dim = self.network.out_dim
        self.weights = list(weights) if len(weights)>0 else [1]*self.num_classes
        self.ignore_index = ignore_index
        self.loss = nn.CrossEntropyLoss(
            ignore_index=self.ignore_index,
            weight=torch.Tensor(self.weights)
            #weight=torch.Tensor(self.weights).reshape(1,-1,*out_dim)
        )
        self.onehot = TorchOneHot(
            range(self.num_classes)
        )
        self.mixup = aug.Mixup(alpha=mixup) if mixup > 0. else None
        #self.save_hyperparameters('network', 'weights', 'mixup', 'ignore_index')
        self.save_hyperparameters()

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = super().add_model_specific_args(parent_parser)
        parser.add_argument("--ignore_index", type=int, default=-1)
        parser.add_argument("--network", type=str)
        parser.add_argument("--weights", type=float, nargs="+", default=())
        parser.add_argument("--mixup", type=float, default=0.)

        return parser

    def forward(self, x):
        
        return self.network(x)

    @classmethod
    def _compute_probas(cls, logits):

        return logits.softmax(dim=1)
    
    @classmethod
    def _compute_conf_preds(cls, probas):
        
        return torch.max(probas, dim=1)

    def training_step(self, batch, batch_idx):

        batch = batch["sup"]
        inputs = batch['image']
        labels = batch['mask']
        if self.mixup:
            labels = self.onehot(labels).float()
            inputs, labels = self.mixup(inputs, labels)
            #batch['image'] = inputs
        logits = self.network(inputs)
        loss = self.loss(logits, labels)
        self.log('Train_sup_CE', loss)
        batch['logits'] = logits.detach()
        
        #if self.trainer.current_epoch % 10 == 0 and batch_idx == 0:
        #    log_batch_images(
        #        batch,
        #        self.trainer,
        #        prefix='Train'
        #    )

        return {'batch': batch, "loss": loss}

    def validation_step(self, batch, batch_idx):

        outs = super().validation_step(batch, batch_idx)
        labels = batch['mask']
        logits = outs['logits']
        loss = self.loss(logits, labels)
        self.log('Val_CE', loss)

        return outs    
