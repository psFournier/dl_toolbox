# Third-party libraries
import numpy as np
import pytorch_lightning as pl
import torch
import torchvision

from dl_toolbox.utils import show_classifications, show_segmentations, show_detections

def display_batch(trainer, module, batch, prefix, mode):
    x = batch['image']
    nb = min(4, len(x))
    
    imgs = list(x.detach().cpu()[:nb])
    mins = [i.view(3, -1).min(1)[0].view(3, 1, 1) for i in imgs]
    maxs = [i.view(3, -1).max(1)[0].view(3, 1, 1) for i in imgs]
    unnorm_imgs = [(i-m)/(M-m) for i,m,M in zip(imgs, mins, maxs)]
    int_imgs = [img.mul(255.).to(torch.uint8) for img in unnorm_imgs]
    
    class_list = trainer.datamodule.class_list
    
    if mode=='classification':
        logits_x = module.forward(x)
        probs = module.logits_to_probas(logits_x)
        pred_probs, preds = torch.max(probs, dim=1)
        preds = list(preds.detach().cpu()[:nb])
        labels = list(batch['target'])[:nb]
        fig = show_classifications(int_imgs, preds, class_list, labels)
    elif mode=='segmentation':
        logits_x = module.forward(x)
        probs = module.logits_to_probas(logits_x)
        pred_probs, preds = torch.max(probs, dim=1)
        preds = list(preds.detach().cpu()[:nb])
        fig = show_segmentations(int_imgs, preds, class_list, alpha=0.5)
    elif mode=='detection':
        outputs = module.forward(x)
        outputs.update({k: v.detach().cpu() for k, v in outputs.items()})
        preds = module.post_process(outputs, x)
        fig = show_detections(int_imgs, preds, class_list)    
    
    trainer.logger.experiment.add_figure(
        f"{prefix} predictions", fig, global_step=trainer.global_step
    )

class PredictionsVisu(pl.Callback):

    def __init__(self, freq, mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        assert mode in ['detection', 'classification', 'segmentation']
        self.mode = mode

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq>0 and trainer.current_epoch % self.freq == 0 and batch_idx <= 1:
            display_batch(trainer, pl_module, batch["sup"], "Train", mode=self.mode)

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq>0 and trainer.current_epoch % self.freq == 0 and batch_idx <= 1:
            display_batch(trainer, pl_module, batch, "Val", mode=self.mode)
            
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq>0 and batch_idx <= 1:
            display_batch(trainer, pl_module, batch, "Test", mode=self.mode)
