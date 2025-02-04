import copy

from torch.utils.data import Subset
from pycocotools.coco import COCO
import pandas as pd

from dl_toolbox.datasets import xView1
from dl_toolbox.datamodules import Base
from dl_toolbox.utils import get_tiles


class xView(Base):
    
    def __init__(
        self,
        *args,
        **kwargs
    ):    
        super().__init__(*args, **kwargs)
        self.class_list = xView1.classes[self.merge].value   
        
    def prepare_data(self):        
        coco = COCO(self.data_path/'XVIEW1'/'xView_train.json')
        df = pd.read_csv(self.data_path/'XVIEW1'/'df.csv', index_col=0)
        filtered = df.loc[df['Building'] != 0.][['image_id', 'left', 'top', 'width', 'height', 'Building']] # A changer pour merge
        ids_tiles = []
        num_obj_per_tile = []
        for _, [img_id, l, t, w, h, num] in filtered.iterrows():
            ids_tiles.append((img_id, (l,t,w,h)))
            num_obj_per_tile.append(num)
        nb_imgs = len(ids_tiles)
        val_start, test_start = int(0.8*nb_imgs), int(0.85*nb_imgs)
        train_ids, val_ids = ids_tiles[:val_start], ids_tiles[val_start:test_start]
        self.train_ids_and_windows = ids_tiles[:val_start]
        self.val_ids_and_windows = ids_tiles[val_start:test_start]
        self.coco = coco
    
    def setup(self, stage=None):
        self.train_set = xView1(
            merge=self.merge,
            coco_dataset=self.coco,
            root=self.data_path/'XVIEW1'/'train_images',
            ids=self.train_ids_and_windows
        )
        self.val_set = xView1(
            merge=self.merge,
            coco_dataset=self.coco,
            root=self.data_path/'XVIEW1'/'train_images',
            ids=self.val_ids_and_windows[::5]
        )
        self.train_set.init_tf(self.train_tf)
        self.val_set.init_tf(self.test_tf)
            
    def collate(self, batch, train):
        return xView1.collate(batch)