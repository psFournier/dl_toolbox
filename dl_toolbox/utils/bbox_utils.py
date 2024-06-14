import torchvision.transforms.v2.functional as v2F
from torchvision import tv_tensors

def to_xyxy(tv_bb, old_format=None):
    return v2F.convert_bounding_box_format(
        tv_bb, 
        new_format=tv_tensors.BoundingBoxFormat.XYXY,
        old_format=old_format,
        inplace=True
    )

def to_xywh(tv_bb, old_format=None):
    return v2F.convert_bounding_box_format(
        tv_bb, 
        new_format=tv_tensors.BoundingBoxFormat.XYWH,
        old_format=old_format,
        inplace=True
    )       

def norm(xyxy):
    h, w = xyxy.canvas_size
    xyxy[..., 0::2].div_(w)
    xyxy[..., 1::2].div_(h)
    return xyxy

def unnorm(xyxy):
    h, w = xyxy.canvas_size
    xyxy[..., 0::2].mul_(w)
    xyxy[..., 1::2].mul_(h) 
    return xyxy