from pytorch_lightning.callbacks import BaseFinetuning
            
class FeatureFt(BaseFinetuning):
    def __init__(self, do_finetune=False, unfreeze_at_epoch=10, train_bn=True):
        super().__init__()
        self.do_finetune = do_finetune
        self._unfreeze_at_epoch = unfreeze_at_epoch
        self.train_bn = train_bn

    def freeze_before_training(self, pl_module):
        # freeze any module you want
        # Here, we are freezing `feature_extractor`
        if self.do_finetune:
            self.freeze(pl_module.network.feature_extractor, train_bn=self.train_bn)

    def finetune_function(self, pl_module, current_epoch, optimizer):
        # When `current_epoch` is 10, feature_extractor will start training.
        if self.do_finetune and current_epoch == self._unfreeze_at_epoch:
            self.unfreeze_and_add_param_group(
                modules=pl_module.network.feature_extractor,
                optimizer=optimizer,
                lr=None,
                initial_denom_lr=10.,
                train_bn=self.train_bn,
            )