from torch.utils.data import DataLoader, RandomSampler
from pathlib import Path
from pytorch_lightning import LightningDataModule
from functools import partial
from pytorch_lightning.utilities import CombinedLoader

class Base(LightningDataModule):
        
    def __init__(
        self,
        data_path,
        merge,
        train_tf,
        val_tf,
        test_tf,
        batch_size,
        epoch_steps,
        num_workers,
        pin_memory,
        *args,
        **kwargs
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.merge = merge
        self.train_tf = train_tf
        self.val_tf = val_tf
        self.test_tf = test_tf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.dataloader = partial(
            DataLoader,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
        self.epoch_steps = epoch_steps
     
    def collate(self, batch, train):
        pass
    
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.dataloader(
            dataset=self.train_set,
            sampler=RandomSampler(
                self.train_set,
                replacement=True,
                num_samples=self.epoch_steps*self.batch_size
            ),
            drop_last=True,
            collate_fn=partial(self.collate, train=True)
        )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        return self.dataloader(
            dataset=self.val_set,
            shuffle=False,
            drop_last=False,
            collate_fn=partial(self.collate, train=False)
        )
    
    def predict_dataloader(self):
        return self.dataloader(
            dataset=self.pred_set,
            shuffle=False,
            drop_last=False,
            collate_fn=partial(self.collate, train=False)
        )