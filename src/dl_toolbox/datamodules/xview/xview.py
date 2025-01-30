from torch.utils.data import DataLoader, Subset
from pathlib import Path
from pytorch_lightning import LightningDataModule
from functools import partial
import dl_toolbox.datasets as datasets
import torch
from pytorch_lightning.utilities import CombinedLoader
from dl_toolbox.utils import list_of_dicts_to_dict_of_lists
import copy

class xView(LightningDataModule):
    
    def __init__(
        self,
        data_path,
        merge,
        train_tf,
        test_tf,
        batch_size,
        num_workers,
        pin_memory,
        *args,
        **kwargs
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.merge=merge
        self.train_tf = train_tf
        self.test_tf = test_tf
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.class_list = datasets.xView1.classes[merge].value
        self.dataloader = partial(
            DataLoader,
            batch_size=batch_size,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    def setup(self, stage=None):
        path = self.data_path/'XVIEW1'
        xview = datasets.xView1(
            merge=self.merge,
            root=path/'train_images',
            annFile=path/'xView_train.json'
        )
        L = len(xview)
        if stage in {'fit'}:
            train_set = copy.deepcopy(xview)
            train_set.init_tf(self.train_tf)
            self.train_set = Subset(train_set, range(int(0.7*L)))
        if stage in {'fit', 'validate'}:
            val_set = copy.deepcopy(xview)
            val_set.init_tf(self.test_tf)
            self.val_set = Subset(val_set, range(int(0.7*L), int(0.8*L)))
    
    def collate(self, batch, train):
        batch = list_of_dicts_to_dict_of_lists(batch)
        batch['image'] = torch.stack(batch['image'])
        # don't stack targets because each batch elem may not have the same nb of bb
        return batch
        #images_b, targets_b, paths_b = tuple(zip(*batch))
        # ignore batch_tf for detection 
        # don't stack bb because each batch elem may not have the same nb of bb
        #return torch.stack(images_b), targets_b, paths_b 
    
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.dataloader(
            dataset=self.train_set,
            shuffle=True,
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