from torchvision.transforms.v2 import functional as F, Transform
from typing import Optional, Tuple, Dict, Any
from torchvision import tv_tensors
import torch

def _norm_bounding_boxes(
    bounding_boxes: torch.Tensor, format: tv_tensors.BoundingBoxFormat, canvas_size: Tuple[int, int]
) -> torch.Tensor:
    in_dtype = bounding_boxes.dtype
    bounding_boxes = bounding_boxes.clone() if bounding_boxes.is_floating_point() else bounding_boxes.float()
    whwh = torch.Tensor(canvas_size).repeat(2).flip(dims=(0,)) # canvas_size is H,W hence the flip to WHWH
    out_boxes = bounding_boxes/whwh 
    return out_boxes.to(in_dtype)

def norm_bounding_boxes(
    inpt: torch.Tensor,
    format: Optional[tv_tensors.BoundingBoxFormat] = None,
    canvas_size: Optional[Tuple[int, int]] = None,
) -> torch.Tensor:
    """See :func:`~torchvision.transforms.v2.ClampBoundingBoxes` for details."""
    if F.is_pure_tensor(inpt):
        if format is None or canvas_size is None:
            raise ValueError("For pure tensor inputs, `format` and `canvas_size` have to be passed.")
        return _norm_bounding_boxes(inpt, format=format, canvas_size=canvas_size)
    elif isinstance(inpt, tv_tensors.BoundingBoxes):
        if format is not None or canvas_size is not None:
            raise ValueError("For bounding box tv_tensor inputs, `format` and `canvas_size` must not be passed.")
        output = _norm_bounding_boxes(inpt.as_subclass(torch.Tensor), format=inpt.format, canvas_size=inpt.canvas_size)
        return tv_tensors.wrap(output, like=inpt)
    else:
        raise TypeError(
            f"Input can either be a plain tensor or a bounding box tv_tensor, but got {type(inpt)} instead."
        )
        
class NormalizeBB(Transform):
    """Normalize bounding boxes according to their corresponding image dimensions.

    The normalization is done according to the bounding boxes' ``canvas_size`` meta-data.

    """

    _transformed_types = (tv_tensors.BoundingBoxes,)

    def _transform(self, inpt: tv_tensors.BoundingBoxes, params: Dict[str, Any]) -> tv_tensors.BoundingBoxes:
        return norm_bounding_boxes(inpt)  # type: ignore[return-value]