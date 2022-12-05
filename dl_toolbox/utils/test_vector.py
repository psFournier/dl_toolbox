import fiona
import rasterio
import imagesize
import numpy as np
from rasterio.features import rasterize
from fiona.transform import transform as f_transform

tile_shp = fiona.open("data/DIGITANIE/Toulouse/shapefile/tlse_arenes.shp")
big_raster = rasterio.open('data/DIGITANIE/Toulouse/normalized_mergedTO.tif')
minx, miny, maxx, maxy = tile_shp.bounds
(minx, maxx), (miny, maxy) = f_transform(tile_shp.crs, dict(big_raster.crs), [minx, maxx], [miny, maxy])
w = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=big_raster.transform)
x = np.random.randint(w.col_off, w.col_off+w.width-2000+1)
y = np.random.randint(w.row_off, w.row_off+w.height-2000+1)
chip = rasterio.windows.Window(x, y, 2000, 2000)
minx, miny, maxx, maxy = rasterio.windows.bounds(chip, transform=big_raster.transform)
t = rasterio.transform.from_bounds(minx, miny, maxx, maxy, 2000, 2000)
(minx, maxx), (miny, maxy) = f_transform(dict(big_raster.crs), tile_shp.crs, [minx, maxx], [miny, maxy])

profile = big_raster.profile
profile.update(dtype=rasterio.uint8, count=3, width=2000, height=2000, transform=t, compress='lzw')
with rasterio.Env():
    with rasterio.open('data/DIGITANIE/Toulouse/tlse_arenes_patch.tif', 'w', **profile) as dst:
        for i, att in enumerate(['num_class']):
            shapes_for_raster = [(s['geometry'], s['properties'][att]) for s in tile_shp.filter(bbox=(minx, miny, maxx, maxy))]
            masks = rasterize(shapes_for_raster, out_shape=(2000, 2000), transform=t)
            dst.write(masks.astype(rasterio.uint8), i+1)

