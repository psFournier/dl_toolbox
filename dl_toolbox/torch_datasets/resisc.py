from torch.utils.data import Dataset
from PIL import Image
from dl_toolbox.torch_datasets.utils import *
import matplotlib.pyplot as plt
import os
import torch



class ResiscDs(Dataset):

    cls_names = ['airplane', 'bridge', 'commercial_area', 'golf_course', 'island', 'mountain', 'railway_station', 'sea_ice', 'storage_tank', 'airport', 'chaparral', 'dense_residential', 'ground_track_field', 'lake', 'overpass', 'rectangular_farmland', 'ship', 'tennis_court', 'baseball_diamond', 'church', 'desert', 'harbor', 'meadow', 'palace', 'river', 'snowberg', 'terrace', 'basketball_court', 'circular_farmland', 'forest', 'industrial_area', 'medium_residential', 'parking_lot', 'roundabout', 'sparse_residential', 'thermal_power_station', 'beach', 'cloud', 'freeway', 'intersection', 'mobile_home_park', 'railway', 'runway', 'stadium', 'wetland']

    def __init__(
        self,
        datadir,
        idxs,
        img_aug,
        *args,
        **kwargs
    ):
        
        self.datadir = datadir
        self.idxs = idxs
        self.img_aug = get_transforms(img_aug)

    def __len__(self):
        
        return len(self.cls_names) * len(self.idxs)

    def __getitem__(self, idx):
        
        cls_idx = idx // len(self.idxs)
        cls_name = self.cls_names[cls_idx]
        img_idx = self.idxs[idx % len(self.idxs)]
        path = os.path.join(self.datadir, cls_name, f'{cls_name}_{img_idx:03}.jpg')
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
        idxs=[1, 3, 4],
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
        


