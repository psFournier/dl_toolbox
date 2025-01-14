import dl_toolbox.datasets as datasets
from dl_toolbox.utils import list_of_dicts_to_dict_of_lists
import torch

from .common import Base

class Coco(Base):
        
    def __init__(
        self,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.class_list = datasets.Coco.classes[self.merge].value
    
    def setup(self, stage=None):
        path = self.data_path/'coco'
        self.train_set = datasets.Coco(
            path/"train2017",
            path/"annotations/instances_train2017.json",
            self.train_tf,
            merge=self.merge
        )
        self.val_set = datasets.Coco(
            path/"val2017",
            path/"annotations/instances_val2017.json",
            self.val_tf,
            merge=self.merge
        )
        self.pred_set = datasets.Coco(
            path/"val2017",
            path/"annotations/instances_val2017.json",
            self.test_tf,
            merge=self.merge
        )
        
    def collate(self, batch, train):
        batch = list_of_dicts_to_dict_of_lists(batch)
        batch['image'] = torch.stack(batch['image'])
        # don't stack targets because each batch elem may not have the same nb of bb
        return batch