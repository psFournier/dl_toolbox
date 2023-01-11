import torch
import rasterio
import numpy as np

from dl_toolbox.utils import minmax, MergeLabels
from dl_toolbox.torch_datasets import RasterDs


labels_dict = {
    
    'base':{
        'nodata': {'color': (0, 0, 0)},
        'bare_ground': {'color':(100, 50, 0)},
        'low_vegetation': {'color':(0, 250, 50)},
        'water': {'color':(0, 50, 250)},
        'building': {'color':(250, 50, 50)},
        'high_vegetation': {'color':(0, 100, 50)},
        'parking': {'color':(200, 200, 200)},
        'road': {'color':(100, 100, 100)},
        'railways': {'color':(200, 100, 200)},
        'swimmingpool': {'color':(50, 150, 250)}
    },

    '6': {
        'other': {'color': (0, 0, 0)},
        'low_vegetation': {'color':(0, 250, 50)},
        'water': {'color':(0, 50, 250)},
        'building': {'color':(250, 50, 50)},
        'high_vegetation': {'color':(0, 100, 50)},
        'road': {'color':(100, 100, 100)}
    },
    '5': {
        'nodata': {'color': (0, 0, 0)},
        'other': {'color': (250, 250, 250)},
        'vegetation': {'color':(0, 250, 50)},
        'water': {'color':(0, 50, 250)},
        'building': {'color':(250, 50, 50)},
        'road': {'color':(100, 100, 100)}
    }

}

mergers = {
    'base' : [[0], [1], [2], [3], [4], [5], [6], [7], [8], [9], [10]],
    '6' : [[0, 1, 6, 8, 9], [2], [3], [4], [5], [7]],
    '5' : [[0], [1, 6, 8, 9], [2, 5], [3], [4], [7]]
}

class DigitanieV2(RasterDs):

    def __init__(self, labels, *args, **kwargs):
 
        self.labels = labels_dict[labels]
        super().__init__(*args, **kwargs)
        self.label_merger = MergeLabels(mergers[labels])

    def read_image(self, image_path, window):
        
        with rasterio.open(image_path) as image_file:
            image = image_file.read(window=window, out_dtype=np.float32)
            
        mins = np.array([stat.min for stat in self.info['stats']])
        maxs = np.array([stat.max for stat in self.info['stats']])
        image = minmax(image[:3], mins[:3], maxs[:3])

        return image

    def read_label(self, label_path, window):
    
        with rasterio.open(label_path) as label_file:
            label = label_file.read(window=window, out_dtype=np.float32)
            
        label = np.squeeze(label)
        label = self.label_merger(label)
        
        return label



def main():
    
    image_path = '/work/OT/ai4geo/DATA/DATASETS/DIGITANIE/Biarritz/Biarritz_EPSG32630_0.tif'
    label_path = '/work/OT/ai4geo/DATA/DATASETS/DIGITANIE/Biarritz/COS9/Biarritz_0-v4.tif'

    dataset = DigitanieV2(
        image_path=image_path,
        label_path=label_path,
        crop_size=1024,
        crop_step=1024,
        img_aug='no',
        tile=Window(col_off=0, row_off=0, width=1024, height=1024),
        fixed_crops=False,
        one_hot=False
    )
    img = dataset[0]['mask']
    print(img.shape)
    img = DigitanieToulouse2Ds.labels_to_rgb(img.numpy())
    #img = img.numpy().transpose((1,2,0))
    img = plt.imsave('digitanie_ds_test.jpg', img)


if __name__ == '__main__':
    main()
