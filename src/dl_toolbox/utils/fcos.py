import torch
import torch.nn as nn
import torchvision

INF = 100000000

def get_fm_anchors(h, w, s):
    """
    Args:
        h, w: height, width of the feat map
        s: stride of the featmap = size reduction factor relative to image
    Returns:
        Tensor NumAnchorsInFeatMap x 2, ordered by column (TODO: check why)
    """
    locs_x = [s / 2 + x * s for x in range(w)]
    locs_y = [s / 2 + y * s for y in range(h)]
    locs = [(y, x) for x in locs_x for y in locs_y] # order !
    return torch.tensor(locs)

def get_all_anchors_bb_sizes(fm_sizes, fm_strides, bb_sizes):
    """
    Args:
        fm_sizes: seq of feature_maps sizes
        fm_strides: seq of corresponding strides
        bb_sizes: seq of bbox sizes feature maps are associated with, len = len(fm) + 1
    Returns:
        anchors: list of num_featmaps elem, where each elem indicates the tensor of anchors of size Nx2 in the original image corresponding to each location in the feature map at this level
        anchors_bb_sizes: sizes of the bbox each anchor is authorized/supposed to detect
    """
    bb_sizes = [-1] + bb_sizes + [INF]
    anchors, anchors_bb_sizes = [], []
    for l, ((h,w), s) in enumerate(zip(fm_sizes, fm_strides)):
        fm_anchors = get_fm_anchors(h, w, s)
        sizes = torch.tensor([bb_sizes[l], bb_sizes[l+1]], dtype=torch.float32)
        sizes = sizes.repeat(len(fm_anchors)).view(len(fm_anchors), 2)
        anchors.append(fm_anchors)
        anchors_bb_sizes.append(sizes)
    return torch.cat(anchors, 0), torch.cat(anchors_bb_sizes, 0)

def _calculate_reg_targets(anchors, bbox):
    """
    Args:
        anchors: Lx2, anchors coordinates
        bbox: tensor of bbox Tx4, format should be xywh
    Returns:
        reg_tgt: l,t,r,b values to regress for each pair (anchor, bbox)
        anchor_in_box: whether anchor is in bbox for each pair (anchor, bbox)
    """
    xs, ys = anchors[:, 0], anchors[:, 1] # L & L, x & y reversed ??
    bbox[:, 2] += bbox[:, 0]
    bbox[:, 3] += bbox[:, 1]
    l = xs[:, None] - bbox[:, 0][None] # Lx1 - 1xT -> LxT
    t = ys[:, None] - bbox[:, 1][None]
    r = bbox[:, 2][None] - xs[:, None]
    b = bbox[:, 3][None] - ys[:, None]
    reg_tgt = torch.stack([l, t, r, b], dim=2) # LxTx4
    anchor_in_box = reg_tgt.min(dim=2)[0] > 0 # LxT
    return reg_tgt, anchor_in_box

def _apply_distance_constraints(reg_targets, anchor_bb_sizes):
    """
    Args:
        reg_targets: LxTx4
        anchor_bb_sizes: Lx2
    Returns:
        A LxT tensor where value at (anchor, bbox) is true if the max value to regress at this anchor for this bbox is inside the bounds associated to this anchor
        If other values to regress than the max are negatives, it is dealt with anchor_in_boxes.
    """
    max_reg_targets, _ = reg_targets.max(dim=2)
    return torch.logical_and(
        max_reg_targets >= anchor_bb_sizes[:, None, 0],
        max_reg_targets <= anchor_bb_sizes[:, None, 1]
    )

def anchor_bbox_area(bbox, anchors, is_in_boxes, fits_to_feature_level):
    """
    Args:
    Returns: 
        Tensor LxT where value at (anchor, bbox) is the area of bbox if anchor is in bbox and anchor is associated with bbox of that size
        Else INF.
    """
    #bbox_areas = _calc_bbox_area(bbox_targets) # T
    bbox_areas = torchvision.ops.box_area(bbox) # compared to above, does not deal with 0dim bb
    # area of each target bbox repeated for each loc with inf where the the loc is not 
    # in the target bbox or if the loc is not at the right level for this bbox size
    anchor_bbox_area = bbox_areas[None].repeat(len(anchors), 1) # LxT
    anchor_bbox_area[is_in_boxes == 0] = INF
    anchor_bbox_area[fits_to_feature_level == 0] = INF
    return anchor_bbox_area

def associate_targets_to_anchors(targets_batch, anchors, anchors_bb_sizes):
    """
    Associate one target cls/bbox to regress ONLY to each anchor: among the bboxes that contain the anchor and have the right size, pick that of min area.
    If no tgt exists for an anchor, the tgt class is 0.
    inputs:
        targets_batch: list of dict of tv_tensors {'labels':, 'boxes':}; boxes should be in XYWH format
        anchors: 
        anchor_bb_sizes:
    outputs:
        all class targets: BxNumAnchors
        all bbox targets: BxNumAnchorsx4
    """
    all_reg_targets, all_cls_targets = [], []
    for targets in targets_batch:
        bbox_targets = targets['boxes'] # Tx4, format XYWH
        cls_targets = targets['labels'] # T
        reg_targets, anchor_in_box = _calculate_reg_targets(
            anchors, bbox_targets) # LxTx4, LxT
        fits_to_feature_level = _apply_distance_constraints(
            reg_targets, anchors_bb_sizes) # LxT
        locations_to_gt_area = anchor_bbox_area(
            bbox_targets, anchors, anchor_in_box, fits_to_feature_level)
        # Core of the anchor/target association
        if cls_targets.shape[0]>0:
            loc_min_area, loc_min_idxs = locations_to_gt_area.min(dim=1) #L,idx in [0,T-1],T must be>0
            reg_targets = reg_targets[range(len(anchors)), loc_min_idxs] # Lx4
            cls_targets = cls_targets[loc_min_idxs] # L
            cls_targets[loc_min_area == INF] = 0 # 0 is no-obj category
        else:
            cls_targets = cls_targets.new_zeros((len(anchors),))
            reg_targets = reg_targets.new_zeros((len(anchors),4))
        all_cls_targets.append(cls_targets)
        all_reg_targets.append(reg_targets)
    # BxL & BxLx4
    return torch.stack(all_cls_targets), torch.stack(all_reg_targets)