from dl_toolbox.datamodules import Base
from dl_toolbox.datasets import PennFudanDataset
from dl_toolbox.utils import label
import torch


class PennFudan(Base):
    def __init__(
        self,
        *args,
        **kwargs
    ):    
        super().__init__(*args, **kwargs)
        self.class_list = [label("person", (0, 255, 0), {1})]

    def setup(self, stage):
        dataset_train = PennFudanDataset('/data/PennFudanPed', self.train_tf)
        dataset_test = PennFudanDataset('/data/PennFudanPed', self.val_tf)
        # split the dataset in train and test set + make hyp that train/test are in same order
        indices = torch.randperm(len(dataset_train)).tolist()
        self.train_set = torch.utils.data.Subset(dataset_train, indices[:-50])
        self.val_set = torch.utils.data.Subset(dataset_test, indices[-50:])
        
    def collate(self, batch, train):
        return PennFudanDataset.collate(batch)