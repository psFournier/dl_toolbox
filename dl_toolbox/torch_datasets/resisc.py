from torch.utils.data import Dataset, DataLoader, Subset
from torchvision.datasets.folder import DatasetFolder
from argparse import ArgumentParser
from PIL import Image
from dl_toolbox.torch_datasets.utils import *
import matplotlib.pyplot as plt
import os
import torch

from dl_toolbox.utils import MergeLabels, OneHot
from dl_toolbox.torch_collate import CustomCollate

cls_names = [
    'airplane',
    'bridge',
    'commercial_area',
    'golf_course',
    'island',
    'mountain',
    'railway_station',
    'sea_ice',
    'storage_tank',
    'airport',
    'chaparral',
    'dense_residential',
    'ground_track_field',
    'lake', 
    'overpass',
    'rectangular_farmland',
    'ship',
    'tennis_court',
    'baseball_diamond',
    'church',
    'desert',
    'harbor',
    'meadow',
    'palace',
    'river',
    'snowberg',
    'terrace',
    'basketball_court',
    'circular_farmland',
    'forest',
    'industrial_area',
    'medium_residential',
    'parking_lot',
    'roundabout',
    'sparse_residential',
    'thermal_power_station',
    'beach',
    'cloud',
    'freeway',
    'intersection',
    'mobile_home_park',
    'railway',
    'runway',
    'stadium',
    'wetland'
]

resisc_labels = {
    'base': {key: {} for key in cls_names},
    'small': {key: {} for key in ['stadium', 'wetland']}
}

def pil_to_torch_loader(path: str):

    with open(path, "rb") as f:
        img = np.array(Image.open(f))
        img = torch.from_numpy(img).permute(2,0,1).float() / 255.
        return img

class ResiscDs(DatasetFolder):

    def __init__(
        self,
        data_path,
        img_aug,
    ):
        super().__init__(
            root=data_path,
            loader=pil_to_torch_loader,
            extensions=('jpg',)
        )
        self.img_aug = get_transforms(img_aug)
        self.class_names = self.classes

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_path", type=str)
        parser.add_argument("--img_aug", type=str)

        return parser

    def __getitem__(self, idx):

        path, label = self.samples[idx]
        image = self.loader(path)
        end_image, _ = self.img_aug(img=image)

        return {
            'orig_image': image,
            'image': end_image,
            'mask': torch.tensor(label),
            'path': path
        }





#class ResiscDs2(Dataset):
#
#    def __init__(
#        self,
#        data_path,
#        idxs,
#        img_aug,
#        labels,
#        *args,
#        **kwargs
#    ):
#        
#        self.data_path = data_path
#        self.idxs = idxs
#        self.img_aug = get_transforms(img_aug)
#        self.labels = resisc_labels[labels]
#        self.cls_names = list(self.labels.keys())
#
#    @classmethod
#    def add_model_specific_args(cls, parent_parser):
#
#        parser = ArgumentParser(parents=[parent_parser], add_help=False)
#        parser.add_argument("--data_path", type=str)
#        parser.add_argument("--img_aug", type=str)
#        parser.add_argument("--labels", type=str)
#
#        return parser
#
#    def __len__(self):
#        
#        return len(self.cls_names) * len(self.idxs)
#
#    def __getitem__(self, idx):
#        
#        cls_idx = idx // len(self.idxs)
#        cls_name = self.cls_names[cls_idx]
#        img_idx = self.idxs[idx % len(self.idxs)]
#        path = os.path.join(self.data_path, cls_name, f'{cls_name}_{img_idx:03}.jpg')
#        img_array = np.asarray(Image.open(path))
#        image = torch.from_numpy(img_array).permute(2,0,1).float() / 255.
#        label = torch.tensor(cls_idx)
#        if self.img_aug is not None:
#            end_image, _ = self.img_aug(img=image)
#        else:
#            end_image = image
#
#        return {
#            'orig_image': image,
#            'image': end_image,
#            'mask': label,
#            'path': path
#        }
        
def main():

    dataset = ResiscDs(
        data_path='/d/pfournie/ai4geo/data/NWPU-RESISC45',
        img_aug='d4'
    )
    train_set = Subset(
        dataset=dataset,
        indices=[700*i+j for i in range(45) for j in range(50)]
    )
    print(len(train_set))

    dataloader = DataLoader(
        dataset=train_set,
        shuffle=True,
        collate_fn=CustomCollate(),
        batch_size=4,
        num_workers=1,
        drop_last=True
    )
    #for i, batch in enumerate(dataloader):
    #    f, ax = plt.subplots(4, 2, figsize=(20, 10))
    #    for j in range(4):
    #        ax[j, 0].imshow(batch['orig_image'][j].numpy().transpose(1,2,0))
    #        ax[j, 1].imshow(batch['image'][j].numpy().transpose(1,2,0))
    #        ax[j,0].set_title(batch['path'][j])
    #        ax[j, 1].set_title(int(batch['mask'][j]))
    #    plt.show()  

if __name__ == '__main__':
    main()
