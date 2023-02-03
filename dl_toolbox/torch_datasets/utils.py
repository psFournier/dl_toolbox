import numpy as np
import rasterio
#import gdal
import dl_toolbox.augmentations as aug


#def read_window_basic_gdal(window, path):
#    ds = gdal.Open(path)
#    image = ds.ReadAsArray(
#        xoff=window.col_off,
#        yoff=window.row_off,
#        xsize=window.width,
#        ysize=window.height
#    ).astype(np.float32)
#    ds = None
#    return image
#
#def read_window_from_big_raster_gdal(window, path, raster_path):
#
#    with rasterio.open(path) as image_file:
#        with rasterio.open(raster_path) as raster_file:
#            left, bottom, right, top = rasterio.windows.bounds(
#                window, 
#                transform=image_file.transform
#            )
#            rw = rasterio.windows.from_bounds(
#                left=left, bottom=bottom, right=right, top=top, 
#                transform=raster_file.transform
#            )
#    ds = gdal.Open(raster_path)
#    image = ds.ReadAsArray(
#        xoff=int(rw.col_off),
#        yoff=int(rw.row_off),
#        xsize=int(rw.width),
#        ysize=int(rw.height))
#
#    image = image.astype(np.float32)
#    return image

aug_dict = {
    'no': aug.NoOp,
    'imagenet': aug.ImagenetNormalize, 
    'd4': aug.D4,
    'hflip': aug.Hflip,
    'vflip': aug.Vflip,
    'd1flip': aug.Transpose1,
    'd2flip': aug.Transpose2,
    'rot90': aug.Rot90,
    'rot180': aug.Rot180,
    'rot270': aug.Rot270,
    'saturation': aug.Saturation,
    'sharpness': aug.Sharpness,
    'contrast': aug.Contrast,
    'gamma': aug.Gamma,
    'brightness': aug.Brightness,
    'color': aug.Color,
    'cutmix': aug.Cutmix,
    'mixup': aug.Mixup
}

anti_aug_dict = {
    'no': aug.NoOp,
    'imagenet': aug.NoOp, 
    'hflip': aug.Hflip,
    'vflip': aug.Vflip,
    'd1flip': aug.Transpose1,
    'd2flip': aug.Transpose2,
    'rot90': aug.Rot270,
    'rot180': aug.Rot180,
    'rot270': aug.rot90,
    'saturation': aug.NoOp,
    'sharpness': aug.NoOp,
    'contrast': aug.NoOp,
    'gamma': aug.NoOp,
    'brightness': aug.NoOp,
}

def get_transforms(name):
    
    if name:
        parts = name.split('_')
        aug_list = []
        for part in parts:
            if part.startswith('color'):
                bounds = part.split('-')[-1]
                augment = aug.Color(bound=0.1*int(bounds))
            elif part.startswith('cutmix2'):
                alpha = part.split('-')[-1]
                augment = aug.Cutmix(alpha=0.1*int(alpha))
            else:
                augment = aug_dict[part]()
            aug_list.append(augment)
        return aug.Compose(aug_list)
    else:
        return aug.NoOp()

