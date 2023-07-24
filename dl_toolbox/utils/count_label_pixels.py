import numpy as np
import rasterio

# src = rasterio.open('/d/pfournie/Documents/ai4geo/data/SemcityTLS_DL/GT_3_4_7_8.tif')
src = rasterio.open(
    Path("/work/OT/ai4geo/DATA/DATASETS/DIGITANIE/Toulouse/COS9/Toulouse_4.tif")
)
img = src.read(out_dtype=np.uint8)
print(np.unique(img.reshape(img.shape[0], -1), axis=1, return_counts=True))
