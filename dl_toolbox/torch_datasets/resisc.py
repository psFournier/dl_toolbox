from torch.utils.data import Dataset
from PIL import Image
from dl_toolbox.torch_datasets.utils import *
import matplotlib.pyplot as plt
import os
import torch

class ResiscDs(Dataset):

    def __init__(
        self,
        datadir,
        cls_list,
        idxs,
        img_aug,
        *args,
        **kwargs
    ):
        
        self.datadir = datadir
        self.cls_list = cls_list
        self.idxs = idxs
        self.img_aug = get_transforms(img_aug)

    def __len__(self):
        
        return len(self.cls_list) * len(self.idxs)

    def __getitem__(self, idx):
        
        cls_name = self.cls_list[idx // len(self.idxs)]
        img_idx = self.idxs[idx % len(self.idxs)]
        path = os.path.join(self.datadir, cls_name, f'{cls_name}_{img_idx:03}.jpg')
        img_array = np.asarray(Image.open(path))
        image = torch.from_numpy(img_array).permute(2,0,1).float() / 255.

        label = torch.tensor(idx // len(self.idxs))

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
        datadir='/home/pfournie/ai4geo/data/NWPU-RESISC45',
        cls_list=['forest', 'harbor'],
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
        


