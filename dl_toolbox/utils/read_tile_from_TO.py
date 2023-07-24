import fiona
import rasterio
import imagesize
import numpy as np
from rasterio.features import rasterize
from fiona.transform import transform as f_transform

tile = rasterio.open("data/DIGITANIE/Toulouse/toulouse_tuile_1_img_normalized.tif")
print(tile.crs)
big_raster = rasterio.open("data/DIGITANIE/Toulouse/normalized_mergedTO.tif")
print(big_raster.crs)
window = rasterio.windows.Window(0, 0, 1000, 1000)
img = tile.read(window=window, out_dtype=np.float32)[:3, ...]
print(np.min(img), np.max(img))
minx, miny, maxx, maxy = rasterio.windows.bounds(window, transform=tile.transform)
print(minx)
(minx, maxx), (miny, maxy) = f_transform(
    dict(tile.crs), dict(big_raster.crs), [minx, maxx], [miny, maxy]
)
print(minx)
new_window = rasterio.windows.from_bounds(
    minx, miny, maxx, maxy, transform=big_raster.transform
)
print(new_window)
new_img = big_raster.read(window=new_window, out_dtype=np.float32)[:3, ...]
print(np.min(new_img), np.max(new_img))
