# Third-party libraries
import pytorch_lightning as pl
import torch

from dl_toolbox.utils import show_classifications, show_segmentations, show_detections, unnormalize

def display_batch(module, batch, class_list, mode):
    x = batch['image']
    int_imgs = unnormalize(x.detach().cpu())
    preds = module(x)
    
    if mode=='classification':
        logits_x = module.forward(x)
        probs = module.logits_to_probas(logits_x)
        pred_probs, preds = torch.max(probs, dim=1)
        preds = list(preds.detach().cpu())
        labels = list(batch['target'])
        fig = show_classifications(int_imgs, preds, class_list, labels)
    elif mode=='segmentation':
        logits_x = module.forward(x)
        probs = module.logits_to_probas(logits_x)
        pred_probs, preds = torch.max(probs, dim=1)
        preds = list(preds.detach().cpu())
        fig = show_segmentations(int_imgs, preds, class_list, alpha=0.5)
    elif mode=='detection':
        preds = [{k: v.detach().cpu() for k, v in p.items()} for p in preds]
        fig = show_detections(int_imgs, preds, class_list)    
    return fig
    

class PredictionsVisu(pl.Callback):

    def __init__(self, freq, mode, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.freq = freq
        assert mode in ['detection', 'classification', 'segmentation']
        self.mode = mode

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq>0 and trainer.current_epoch % self.freq == 0 and batch_idx <= 1:
            fig = display_batch(pl_module, batch["sup"], trainer.datamodule.class_list, mode=self.mode)
            trainer.logger.experiment.add_figure(
                f"Train predictions", fig, global_step=trainer.global_step
            )

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq>0 and trainer.current_epoch % self.freq == 0 and batch_idx <= 1:
            fig = display_batch(pl_module, batch, trainer.datamodule.class_list, mode=self.mode)
            trainer.logger.experiment.add_figure(
                f"Val predictions", fig, global_step=trainer.global_step
            )
            
    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        if self.freq>0 and batch_idx <= 1:
            fig = display_batch(pl_module, batch, trainer.datamodule.class_list, mode=self.mode)
            trainer.logger.experiment.add_figure(
                f"Test predictions", fig, global_step=trainer.global_step
            )
