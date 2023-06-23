import numpy as np
import torch

def stretch_to_minmax(img, minval, maxval):
    
    return (img - minval) / (maxval - minval)
