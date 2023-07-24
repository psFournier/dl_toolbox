from argparse import ArgumentParser

import dltoolbox.inference as dl_inf
import matplotlib.pyplot as plt
import numpy as np
import rasterio
from scipy import ndimage
from skimage import color

from dl_toolbox.utils import MergeLabels

parser = ArgumentParser()
parser.add_argument("--label_path", type=str)
args = parser.parse_args()

with rasterio.open(args.label_path) as f:
    rgb_label = f.read(out_dtype=np.uint8)
label = dl_inf.rgb_to_labels(rgb_label, dataset="semcity")


merges = [[0], [1, 2, 3, 5, 6, 7, 8], [4]]
label_merger = MergeLabels(merges)

label = label_merger(label)

fig = plt.figure(figsize=(20, 20))
ax1 = fig.add_subplot(111)
ax1.imshow(label, cmap="gray")
plt.show(block=True)
