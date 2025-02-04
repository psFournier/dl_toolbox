import copy

from torch.utils.data import Subset
from pycocotools.coco import COCO

from dl_toolbox.datasets import xView1
from dl_toolbox.datamodules import Base
from dl_toolbox.utils import get_tiles


class xView(Base):
    
    def __init__(
        self,
        tile_size,
        *args,
        **kwargs
    ):    
        super().__init__(*args, **kwargs)
        self.tile_size = tile_size # w
        self.class_list = xView1.classes[self.merge].value   
        
    def prepare_data(self):        
        coco = COCO(self.data_path/'XVIEW1'/'xView_train.json')
        ids = []
        merges = [list(l.values) for l in self.class_list]
        for merge in merges:
            # It seems that xview annotations geojson contain bboxes for images not in train_images nor val (test set not avail?)
            ids += [id for id in coco.getImgIds(catIds=merge) if id in coco.imgs.keys()]
        nb_imgs = len(ids)
        val_start, test_start = int(0.8*nb_imgs), int(0.85*nb_imgs)
        train_ids, val_ids = ids[:val_start], ids[val_start:test_start]
        self.train_ids_and_windows = []
        for id in train_ids:
            img = coco.imgs[id]
            tiles = get_tiles(
                img['width'],
                img['height'],
                self.tile_size,
                step_w=256
            )
            self.train_ids_and_windows += [(id, w) for w in tiles]
        self.val_ids_and_windows = []
        for id in val_ids:
            img = coco.imgs[id]
            tiles = get_tiles(
                img['width'],
                img['height'],
                self.tile_size,
                step_w=self.tile_size
            )
            self.val_ids_and_windows += [(id, w) for w in tiles]
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