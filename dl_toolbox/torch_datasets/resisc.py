from torch.utils.data import Dataset
from argparse import ArgumentParser
from PIL import Image
from dl_toolbox.torch_datasets.utils import *
import matplotlib.pyplot as plt
import os
import torch

from dl_toolbox.utils import MergeLabels, OneHot

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
    'lake', 'overpass',
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
    'base': {key: {} for key in cls_names}
}

resisc_label_mergers = {
    'base': [[i] for i in range(len(cls_names))]
}

class ResiscDs(Dataset):

    def __init__(
        self,
        data_path,
        idxs,
        img_aug,
        labels,
        *args,
        **kwargs
    ):
        
        self.data_path = data_path
        self.idxs = idxs
        self.img_aug = get_transforms(img_aug)
        self.labels = resisc_labels[labels]
        self.cls_names = list(self.labels.keys())
        self.label_merger = MergeLabels(resisc_label_mergers[labels])

    @classmethod
    def add_model_specific_args(cls, parent_parser):

        parser = ArgumentParser(parents=[parent_parser], add_help=False)
        parser.add_argument("--data_path", type=str)
        parser.add_argument("--img_aug", type=str)
        parser.add_argument("--labels", type=str)

        return parser

    def __len__(self):
        
        return len(self.cls_names) * len(self.idxs)

    def __getitem__(self, idx):
        
        cls_idx = idx // len(self.idxs)
        cls_name = self.cls_names[cls_idx]
        img_idx = self.idxs[idx % len(self.idxs)]
        path = os.path.join(self.data_path, cls_name, f'{cls_name}_{img_idx:03}.jpg')
        img_array = np.asarray(Image.open(path))
        image = torch.from_numpy(img_array).permute(2,0,1).float() / 255.

        label = torch.tensor(cls_idx)

        if self.img_aug is not None:
            end_image, _ = self.img_aug(img=image)
        else:
            end_image = image

        return {
            'orig_image': image,
            'image': end_image,
            'mask': label
        }
        
def main():

    dataset = ResiscDs(
        datadir='/d/pfournie/ai4geo/data/NWPU-RESISC45',
        n_img=3,
        img_aug='d4'
    )

    print(len(dataset))
    res = dataset[3]
    print(res['image'].shape)
    orig_image = res['orig_image'].numpy().transpose(1,2,0)
    image = res['image'].numpy().transpose(1,2,0)
    f, ax = plt.subplots(1, 2, figsize=(10, 10))
    ax[0].imshow(orig_image)
    ax[1].imshow(image)
    ax[0].set_title(dataset.cls_list[int(res['mask'])])
    plt.show()


if __name__ == '__main__':
    main()
        


