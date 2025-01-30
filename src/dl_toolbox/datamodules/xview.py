import copy

from torch.utils.data import Subset

from dl_toolbox.datasets import xView1
from dl_toolbox.datamodules import Base


class xView(Base):
    
    def __init__(
        self,
        *args,
        **kwargs
    ):    
        super().__init__(*args, **kwargs)
        self.class_list = xView1.classes[self.merge].value        
    
    def setup(self, stage=None):
        path = self.data_path/'XVIEW1'
        xview = xView1(
            merge=self.merge,
            root=path/'train_images',
            annFile=path/'xView_train.json'
        )
        L = len(xview)
        print(L)
        train_set = copy.deepcopy(xview)
        train_set.init_tf(self.train_tf)
        self.train_set = Subset(train_set, range(int(0.7*L)))
        val_set = copy.deepcopy(xview)
        val_set.init_tf(self.test_tf)
        self.val_set = Subset(val_set, range(int(0.7*L), int(0.8*L)))
            
    def collate(self, batch, train):
        return xView1.collate(batch)