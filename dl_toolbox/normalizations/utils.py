import numpy as np
import torch

def stretch_to_minmax(img, mins, maxs):
    
    return (img - mins) / (maxs - mins)
