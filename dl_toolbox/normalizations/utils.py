import numpy as np

def stretch_to_minmax(img, mins, maxs):
    
    res = (img - mins) / (maxs - mins)
    
    return np.clip(res, 0, 1)
