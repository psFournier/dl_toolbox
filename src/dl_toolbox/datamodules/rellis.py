import dl_toolbox.datasets as datasets
from dl_toolbox.utils import list_of_dicts_to_dict_of_lists
import torch

from .common import Base

class Rellis(Base):
    
    sequences = [
        "00000",
        "00001",
        "00002",
        "00003",
        "00004",
    ]
        
    def __init__(
        self,
        *args,
        **kwargs
    ):    
        super().__init__(*args, **kwargs)
        self.class_list = datasets.Rellis3d.all_class_lists[merge].value
        self.num_frames = num_frames

    def setup(self, stage):
        
        def get_imgs_msks(seqs, start, end):
            imgs = []
            msks = []
            for s in seqs:
                img_dir = self.data_path/'Rellis-3D'/s/'pylon_camera_node'
                msk_dir = self.data_path/'Rellis-3D'/s/'pylon_camera_node_label_id'
                for msk_name in sorted(os.listdir(msk_dir))[start:end]:
                    img_name = "{}.{}".format(msk_name.split('.')[0], "jpg")
                    imgs.append(img_dir/img_name)
                    msks.append(msk_dir/msk_name)
            return imgs, msks
        
        train_imgs, train_msks = get_imgs_msks(self.sequences, 0, self.num_frames)
        train_set = datasets.Rellis3d(train_imgs, train_msks, self.merge, self.train_tf)
        self.train_set = Subset(train_set, indices=list(range(0, len(train_set), 1)))   
        
        val_imgs, val_msks = get_imgs_msks(self.sequences, 700, -1)
        val_set = datasets.Rellis3d(val_imgs, val_msks, self.merge, self.val_tf)
        self.val_set = Subset(val_set, indices=list(range(0, len(val_set), 1)))
        
        pred_set = datasets.Rellis3d(val_imgs, val_msks, self.merge, self.test_tf)
        self.pred_set = Subset(pred_set, indices=list(range(0, len(pred_set), 10)))
        
    def collate(self, batch, train):
        batch = list_of_dicts_to_dict_of_lists(batch)
        batch['image'] = torch.stack(batch['image'])
        batch['target'] = torch.stack(batch['target'])
        return batch