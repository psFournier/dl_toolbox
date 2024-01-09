from pathlib import Path
import os
from functools import partial
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from torch.utils.data import DataLoader, RandomSampler, ConcatDataset

import dl_toolbox.datasets as datasets
from dl_toolbox.utils import CustomCollate
from dl_toolbox.transforms import Compose, NoOp, RandomCrop2


class Cityscapes(LightningDataModule):
    def __init__(
        self,
        data_path,
        merge,
        sup,
        unsup,
        to_0_1,
        train_tf,
        test_tf,
        batch_size_s,
        batch_size_u,
        steps_per_epoch,
        num_workers,
        pin_memory,
        class_weights=None,
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.merge = merge
        self.sup = sup
        self.unsup = unsup
        self.to_0_1 = to_0_1
        self.train_tf = train_tf
        self.test_tf = test_tf
        self.batch_size_s = batch_size_s
        self.batch_size_u = batch_size_u
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers
        self.pin_memory = pin_memory
        self.in_channels = 3
        self.classes = datasets.Cityscapes.classes[merge].value
        self.num_classes = len(self.classes)
        self.class_names = [l.name for l in self.classes]
        self.class_colors = [(i, l.color) for i, l in enumerate(self.classes)]
        self.class_weights = (
            [1.0] * self.num_classes if class_weights is None else class_weights
        )
        
    def prepare_data(self):
        self.dict_train = {"IMG": [], "MSK": [], "WIN": []}
        self.dict_train_unlabeled = {"IMG": [], "MSK": [], "WIN": []}
        self.dict_val = {"IMG": [], "MSK": [], "WIN": []}
        def get_split_data(split):
            imgs = []
            msks = []
            img_dir = self.data_path/'Cityscapes'/'leftImg8bit'/split
            msk_dir = self.data_path/'Cityscapes'/'gtFine'/split
            for city in os.listdir(img_dir):
                for file_name in os.listdir(img_dir/city):
                    target_name = "{}_{}".format(
                        file_name.split("_leftImg8bit")[0], "gtFine_labelIds.png"
                    )
                    imgs.append(img_dir/city/file_name)
                    msks.append(msk_dir/city/target_name)
            return zip(imgs, msks)
        for i, (img, msk) in enumerate(get_split_data('train')):
            if self.sup <= i%100 < self.sup + self.unsup:
                self.dict_train_unlabeled["IMG"].append(img)
            if i%100 < self.sup:
                self.dict_train["IMG"].append(img)
                self.dict_train["MSK"].append(msk)
        for img, msk in get_split_data('val'):
            self.dict_val["IMG"].append(img)
            self.dict_val["MSK"].append(msk)

    def setup(self, stage):
        if stage in ("fit", "validate"):
            self.train_s_set = datasets.Cityscapes(
                self.dict_train['IMG'],
                self.dict_train['MSK'],
                merge=self.merge,
                transforms=Compose([self.to_0_1, self.train_tf])
            )

            self.val_set = datasets.Cityscapes(
                self.dict_val['IMG'],
                self.dict_val['MSK'],
                merge=self.merge,
                transforms=Compose([self.to_0_1, self.test_tf]),
            )
            if self.unsup > 0:
                self.train_u_set = datasets.Cityscapes(
                    self.dict_train_unlabeled["IMG"],
                    [],
                    self.merge,
                    transforms=Compose([self.to_0_1, self.train_tf])
                )

    def dataloader(self, dataset):
        return partial(
            DataLoader,
            dataset=dataset,
            collate_fn=CustomCollate(),
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
                       
    def train_dataloader(self):
        train_dataloaders = {}
        train_dataloaders["sup"] = self.dataloader(self.train_s_set)(
            sampler=RandomSampler(
                self.train_s_set,
                replacement=True,
                num_samples=self.steps_per_epoch*self.batch_size_s
            ),
            drop_last=True,
            batch_size=self.batch_size_s
        )
        if self.unsup != -1:
            train_dataloaders["unsup"] = self.dataloader(self.train_u_set)(
                sampler=RandomSampler(
                    self.train_u_set,
                    replacement=True,
                    num_samples=self.steps_per_epoch*self.batch_size_u
                ),
                drop_last=True,
                batch_size=self.batch_size_u
            )
        return CombinedLoader(train_dataloaders, mode="max_size_cycle")
    
    def val_dataloader(self):
        return self.dataloader(self.val_set)(
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size_s
        )
    
    def test_dataloader(self):
        return self.dataloader(self.test_set)(
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size_s
        )
