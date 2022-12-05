import fiona
import rasterio
import imagesize
import numpy as np
from rasterio.features import rasterize
from fiona.transform import transform as f_transform

tile_shp = fiona.open("data/DIGITANIE/Toulouse/shapefile/tlse_arenes.shp")
big_raster = rasterio.open('data/DIGITANIE/Toulouse/normalized_mergedTO.tif')
# les boundaries du shapefile exprimées dans le crs du shapefile
minx, miny, maxx, maxy = tile_shp.bounds
# les boundaries du shapefile exprimées dans le crs du raster
(minx, maxx), (miny, maxy) = f_transform(tile_shp.crs, dict(big_raster.crs), [minx, maxx], [miny, maxy])
# on crée la window dans le raster correspondant au boundaries du shapefile
w = rasterio.windows.from_bounds(minx, miny, maxx, maxy, transform=big_raster.transform)
# on extrait une sous-tuile de la window
x = np.random.randint(w.col_off, w.col_off+w.width-2000+1)
y = np.random.randint(w.row_off, w.row_off+w.height-2000+1)
chip = rasterio.windows.Window(x, y, 2000, 2000)
# les boundaries de la sous-tuile dans le crs du raster
minx, miny, maxx, maxy = rasterio.windows.bounds(chip, transform=big_raster.transform)
# les boundaries de la sous-tuile dans le crs du shapefile
(minx, maxx), (miny, maxy) = f_transform(dict(big_raster.crs), tile_shp.crs, [minx, maxx], [miny, maxy])

profile = big_raster.profile
profile.update(dtype=rasterio.uint8, count=1, width=2000, height=2000, transform=t, compress='lzw')
with rasterio.Env():
    
    t = rasterio.transform.from_bounds(minx, miny, maxx, maxy, 2000, 2000)
    shapes = tile_shp.filter(bbox=(minx, miny, maxx, maxy))
    shapes_for_raster = [(s['geometry'], s['properties']['num_class']) for s in
                         shapes]
    masks = rasterize(shapes_for_raster, out_shape=(2000, 2000), transform=t)
    masks2 = rasterize(shapes_for_raster, out_shape=(2000, 2000), transform=t,
                       all_touched=True)
    with rasterio.open('data/DIGITANIE/Toulouse/tlse_arenes_patch.tif', 'w', **profile) as dst:
        dst.write(masks.astype(rasterio.uint8), 1)
    with rasterio.open('data/DIGITANIE/Toulouse/tlse_arenes_edges.tif', 'w',
                       **profile) as dst2:


