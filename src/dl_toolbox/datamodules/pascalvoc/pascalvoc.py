from torch.utils.data import DataLoader, RandomSampler
from pathlib import Path
import pandas as pd
import numpy as np
from pytorch_lightning import LightningDataModule
from functools import partial
import dl_toolbox.datasets as datasets
import torch


class PascalVOC(LightningDataModule):
    
    def __init__(
        self,
        data_path,
        train_tf,
        test_tf,
        merge,
        batch_size_s,
        steps_per_epoch,
        num_workers,
        pin_memory,
        *args,
        **kwargs
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.train_tf = train_tf
        self.test_tf = test_tf
        self.batch_size_s = batch_size_s
        self.steps_per_epoch = steps_per_epoch
        self.in_channels = 3
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.classes = datasets.PascalVOC.classes[merge]
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
    
    def prepare_data(self):
        img_dir = self.data_path/"PASCALVOC/VOCdevkit/VOC2012/JPEGImages"
        self.instances = []
        for _, row in pd.read_pickle(self.data_path/"PASCALVOC/voc_combined.csv").iterrows():
            img_path = row["filename"]
            labels_ = row["labels"]
            image_path = f"{img_dir}/{img_path}"
            labels_ = [[self.class_names.index(l)] for l in labels_]
            targets_ = np.concatenate([row["bboxes"], labels_],
                                      axis=-1).tolist()
            self.instances.append({"image_path": image_path, "target": targets_})
    
    def setup(self, stage=None):
        split = int(0.95*len(self.instances))
        train_data = self.instances[:split]
        val_data = self.instances[split:]
        self.train_s_set = datasets.PascalVOC(train_data, transforms=self.train_tf)
        self.val_set = datasets.PascalVOC(val_data, transforms=self.test_tf)
    
    @staticmethod
    def _collate(batch):
        images_b, bboxes_b, labels_b, image_paths_b = list(zip(*batch))
        # don't stack bb because each batch elem may not have the same nb of bb
        return torch.stack(images_b), bboxes_b, labels_b, image_paths_b 
                
    def _dataloader(self, dataset):
        return partial(
            DataLoader,
            dataset=dataset,
            collate_fn=self._collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
                       
    def train_dataloader(self):
        return self._dataloader(self.train_s_set)(
            sampler=RandomSampler(
                self.train_s_set,
                replacement=True,
                num_samples=self.steps_per_epoch*self.batch_size_s
            ),
            drop_last=True,
            batch_size=self.batch_size_s
        )
    
    def val_dataloader(self):
        return self._dataloader(self.val_set)(
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size_s
        )