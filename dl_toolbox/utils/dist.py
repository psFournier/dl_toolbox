import numpy as np
import matplotlib.pyplot as plt
import rasterio
from scipy import ndimage
from skimage import color
from rasterio.windows import Window

with rasterio.open('data/SemcityTLS_DL/train/TLS_GT_08.tif') as f:
    label = np.transpose(f.read(window=Window(0,0,128,128)), axes=(1,2,0))

with rasterio.open('data/SemcityTLS_DL/train/TLS_BDSD_M_08.tif') as f:
    img = np.transpose(f.read(window=Window(0,0,128,128))[[3,2,1],...], axes=(1,2,0))

p2, p98 = np.percentile(img, [2,98])
img = (img - p2)/(p98 - p2)
gray = color.rgb2gray(label)*255

kernel = np.array([[1, 1, 1], [1, -8, 1], [1, 1, 1]], dtype=np.float32)
edges = ndimage.convolve(gray, weights=kernel)
edges = np.clip(edges, 0, 255).astype(np.uint8)

binary = edges
binary[edges<=20] = 1
binary[edges>20] = 0
dist = ndimage.distance_transform_edt(binary)
dist[dist <= 2] = 0
dist[dist > 2] = 1
#dist = (1 + np.tanh(5*(dist - np.mean(dist))))/2

fig = plt.figure(figsize=(20,20))
ax1 = fig.add_subplot(221)
ax2 = fig.add_subplot(222)
ax3 = fig.add_subplot(223)
ax4 = fig.add_subplot(224)
ax1.imshow(img)
ax2.imshow(gray, cmap='gray')
ax3.imshow(edges, cmap='gray')
ax4.imshow(dist, cmap='gray')
plt.show(block=True)

