import os
import numpy as np
import rasterio
from rasterio.windows import Window

import torch
from torch.utils.data import Dataset
import matplotlib.colors as colors
from dl_toolbox.utils import MergeLabels, OneHot, LabelsToRGB, RGBToLabels
import dl_toolbox.transforms as transforms


lut_colors = {
1   : '#db0e9a',
2   : '#938e7b',
3   : '#f80c00',
4   : '#a97101',
5   : '#1553ae',
6   : '#194a26',
7   : '#46e483',
8   : '#f3a60d',
9   : '#660082',
10  : '#55ff00',
11  : '#fff30d',
12  : '#e4df7c',
13  : '#3de6eb',
14  : '#ffffff',
15  : '#8ab3a0',
16  : '#6b714f',
17  : '#c5dc42',
18  : '#9999ff',
19  : '#000000'}

lut_classes = {
1   : 'building',
2   : 'pervious',
3   : 'impervious',
4   : 'bare soil',
5   : 'water',
6   : 'coniferous',
7   : 'deciduous',
8   : 'brushwood',
9   : 'vineyard',
10  : 'herbaceous',
11  : 'agricultural',
12  : 'plowed land',
13  : 'swimmingpool',
14  : 'snow',
15  : 'clear cut',
16  : 'mixed',
17  : 'ligneous',
18  : 'greenhouse',
19  : 'other'}

def hex2color(hex):
    return tuple([int(z * 255) for z in colors.hex2color(hex)])

labels_dict = {
    'base': {lut_classes[i]: {'color': hex2color(lut_colors[i])} for i in range(1,20)},
    '13': {
        **{lut_classes[i]: {'color': hex2color(lut_colors[i])} for i in range(1,13)},
        **{'other': {'color': hex2color(lut_colors[19])}}
    }
}

mergers = {
    'base' : [[i] for i in range(1, 20)],
    '13' : [[i] for i in range(1, 13)] + [[13, 14, 15, 16, 17, 18, 19]]
}


class Flair(Dataset):

    def __init__(self,
                 dict_files,
                 labels,
                 crop_size,
                 #num_classes=13, 
                 use_metadata=False,
                 img_aug=None,
                 ):

        self.list_imgs = np.array(dict_files["IMG"])
        self.list_msks = np.array(dict_files["MSK"])
        self.labels = labels_dict[labels]
        self.crop_size = crop_size
        self.label_merger = MergeLabels(mergers[labels])
        self.use_metadata = use_metadata
        if use_metadata == True:
            self.list_metadata = np.array(dict_files["MTD"])
        self.img_aug = transforms.get_transforms(img_aug)
        #self.num_classes = num_classes
        self.labels_to_rgb = LabelsToRGB(self.labels)
        self.rgb_to_labels = RGBToLabels(self.labels)

    def read_image(self, image_path, window):
        
        with rasterio.open(image_path) as image_file:
            image = image_file.read(window=window, out_dtype=np.float32)[:3]
            
        return image
    
    def read_label(self, label_path, window):
    
        with rasterio.open(label_path) as label_file:
            label = label_file.read(window=window, out_dtype=np.float32)
            
        label = np.squeeze(label)
        label = self.label_merger(label)
        # attention on a enelvé le onehot encoding par rapport au code falir initial
        
        return label
        
    def __len__(self):
        return len(self.list_imgs)

    def __getitem__(self, index):
        
        image_file = self.list_imgs[index]
        cx = np.random.randint(0, 512 - self.crop_size + 1)
        cy = np.random.randint(0, 512 - self.crop_size + 1)
        window = Window(cx, cy, self.crop_size, self.crop_size)
        image = self.read_image(image_file, window)
        image = torch.from_numpy(image).float().contiguous()
        
        label = None
        if self.list_msks.size: # a changer
            mask_file = self.list_msks[index] 
            label = self.read_label(mask_file, window)
            label = torch.from_numpy(label).long().contiguous()            

        if self.img_aug is not None:
            end_image, end_mask = self.img_aug(img=image, label=label)
        else:
            end_image, end_mask = image, label
        
        # todo : gerer metadata
        
        return {
            #'orig_image':image,
            #'orig_mask':label,
            'image':end_image,
            'label':end_mask,
            'path': '/'.join(image_file.split('/')[-4:])
        }

#        if self.use_metadata == True:
#            mtd = self.list_metadata[index]
#            return {"img": torch.as_tensor(img, dtype=torch.float), 
#                    "mtd": torch.as_tensor(mtd, dtype=torch.float),
#                    "msk": torch.as_tensor(msk, dtype=torch.float)}
     

    
    
    
    
    

#class Predict_Dataset(Dataset):
#
#    def __init__(self,
#                 dict_files,
#                 num_classes=13, use_metadata=True
#                 ):
#        self.list_imgs = np.array(dict_files["IMG"])
#        self.num_classes = num_classes
#        self.use_metadata = use_metadata
#        if use_metadata == True:
#            self.list_metadata = np.array(dict_files["MTD"])
#
#    def read_img(self, raster_file: str) -> np.ndarray:
#        with rasterio.open(raster_file) as src_img:
#            array = src_img.read()
#            return array
#        
#    def __len__(self):
#        return len(self.list_imgs)
#
#    def __getitem__(self, index):
#        image_file = self.list_imgs[index]
#        img = self.read_img(raster_file=image_file)
#        img = img_as_float(img)
#
#        if self.use_metadata == True:
#            mtd = self.list_metadata[index]
#            return {"img": torch.as_tensor(img, dtype=torch.float), 
#                    "mtd": torch.as_tensor(mtd, dtype=torch.float),
#                    "id": '/'.join(image_file.split('/')[-4:])}
#        else:
#           
#            return {"img": torch.as_tensor(img, dtype=torch.float),
#                    "id": '/'.join(image_file.split('/')[-4:])}  
