import numpy as np
import torch
import rasterio
    
def read_window_from_big_raster(window, path, raster_path):
    
    with rasterio.open(path) as image_file:
        with rasterio.open(raster_path) as raster_file:
            left, bottom, right, top = rasterio.windows.bounds(
                window, 
                transform=image_file.transform
            )
            rw = rasterio.windows.from_bounds(
                left, bottom, right, top, 
                transform=raster_file.transform
            )
            image = raster_file.read(
                window=rw, 
                out_dtype=np.float32
            )
        
    return image

def minmax(image, m, M):

    if isinstance(image, np.ndarray):
        return np.clip((image - np.reshape(m, (-1, 1, 1))) / np.reshape(M - m, (-1, 1, 1)), 0, 1)
    elif isinstance(image, torch.Tensor):
        return torch.clip((image - torch.reshape(m, (-1, 1, 1))) / torch.reshape(M - m, (-1, 1, 1)), 0, 1)
    else:
        assert False

class MergeLabels:

    def __init__(self, labels, label_names=None):

        self.labels = labels

    def __call__(self, L):
        """
        If self.labels is [[0,1,2],[3,4,5]], then all pixels in classes [0,1,2]
        will be set to label 0 and all pixels in classes [3,4,5] will be set to
        label 1.
        :param L:
        :return:
        """
        ret = np.zeros(L.shape, dtype=L.dtype)
        for i, lab in enumerate(self.labels):
            for j in lab:
                ret[L == j] = i

        return ret

class TorchOneHot:

    def __init__(self, labels):

        self.labels = labels

    def __call__(self, L):

        onehot_masks = [
            (L==val).squeeze() for val in self.labels
        ]

        return torch.stack(onehot_masks, axis=1)

class OneHot:

    def __init__(self, labels):

        self.labels = labels

    def __call__(self, L):

        onehot_masks = [(L==val).astype(int).squeeze() for val in self.labels]

        return np.stack(onehot_masks, axis=0)

class LabelsToRGB:
    # Inputs shape : B,H,W or H,W
    # Outputs shape : B,H,W,3 or H,W,3

    def __init__(self, labels):

        self.labels = labels

    def __call__(self, labels):
        rgb = np.zeros(shape=(*labels.shape, 3), dtype=np.uint8)
        for label, key in enumerate(self.labels):
            mask = np.array(labels == label)
            rgb[mask] = np.array(self.labels[key]['color'])

        return rgb

class RGBToLabels:
    # Inputs shape : B,H,W,3 or H,W,3
    # Outputs shape : B,H,W or H,W
    def __init__(self, labels):

        self.labels = labels

    def __call__(self, rgb):

        labels = np.zeros(shape=rgb.shape[:-1], dtype=np.uint8)
        for label, key in enumerate(self.labels):
            c = self.labels[key]['color']
            d = rgb[..., 0] == c[0]
            d = np.logical_and(d, (rgb[..., 1] == c[1]))
            d = np.logical_and(d, (rgb[..., 2] == c[2]))
            labels[d] = label

        return labels
