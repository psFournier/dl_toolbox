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



