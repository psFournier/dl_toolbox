import torchmetrics as M
from pytorch_lightning.callbacks import StochasticWeightAveraging


class Swa(StochasticWeightAveraging):
    def __init__(self, *args, **kwargs):
        super(Swa, self).__init__(*args, **kwargs)
        
    def on_fit_start(self, trainer, pl_module):
        super().on_fit_start(trainer, pl_module)
        n = pl_module.num_classes
        d = pl_module.device
        self.acc = M.Accuracy(task='multiclass', num_classes=n).to(d)
        #print(list(pl_module.modules())[:10])
        print(self._model_contains_batch_norm)
        
#    def on_train_batch_start(self, trainer, pl_module, batch, batch_idx):
#        super().on_train_batch_start(trainer, pl_module, batch, batch_idx)
#        if (self.swa_start <= trainer.current_epoch <= self.swa_end):
#            batch = batch['sup']
#            x, y = batch["image"], batch['label']
#            logits = pl_module(x)
#            swa_logits = self._average_model(x)
#            print(f'batch {batch_idx}:', torch.mean(swa_logits - logits))

    def on_train_end(self, trainer, pl_module):
        super().on_train_end(trainer,pl_module)
        

    def on_validation_epoch_end(self, trainer, pl_module):
        super().on_validation_epoch_end(trainer, pl_module)
        if trainer.current_epoch >= self.swa_start :
            acc = self.acc.compute()
            print('swa acc :', acc)
            self.log("accuracy/swa_val", acc)
            self.acc.reset()          

    def on_validation_batch_end(self, trainer, pl_module, outputs, batch, batch_idx):
        super().on_validation_batch_end(trainer, pl_module, outputs, batch, batch_idx)
        if self._model_contains_batch_norm and trainer.current_epoch - 1 == self.swa_end + 1:
            x, y = batch["image"], batch['label']
            #logits = pl_module(x)
            swa_logits = self._average_model(x)
            swa_preds = swa_logits.argmax(dim=1)
            self.acc.update(swa_preds, y)