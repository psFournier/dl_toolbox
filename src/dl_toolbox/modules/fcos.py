import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision


## MATCHING 

INF = 100000000

def get_fm_anchors(h, w, s):
    """
    Args:
        h, w: height, width of the feat map
        s: stride of the featmap = size reduction factor relative to image
    Returns:
        Tensor NumAnchorsInFeatMap x 2, ordered by column 
        TODO: check why: DONE: it corresponds to how locs are computed in 
        https://github.com/tianzhi0549/FCOS/blob/master/fcos_core/modeling/rpn/fcos/fcos.py
        When flattening feat maps, we see first the line at H(=y) fixed and W(=x) moving
        
    """
    locs_x = [s / 2 + x * s for x in range(w)]
    locs_y = [s / 2 + y * s for y in range(h)]
    locs = [(x, y) for y in locs_y for x in locs_x] # order !
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
    
def calculate_reg_targets(anchors, bbox):
    """
    Args:
        anchors: Lx2, anchors coordinates
        bbox: tensor of bbox Tx4, format should be xywh
    Returns:
        reg_tgt: l,t,r,b values to regress for each pair (anchor, bbox)
        anchor_in_box: whether anchor is in bbox for each pair (anchor, bbox)
    """
    xs, ys = anchors[:, 0], anchors[:, 1] # L & L, x & y reversed ?? x means position on x-axis
    l = xs[:, None] - bbox[:, 0][None] # Lx1 - 1xT -> LxT
    t = ys[:, None] - bbox[:, 1][None]
    r = bbox[:, 2][None] + bbox[:, 0][None] - xs[:, None]
    b = bbox[:, 3][None] + bbox[:, 1][None] - ys[:, None]  
    #print(xs[0], ys[0], l[0], t[0], r[0], b[0])
    return torch.stack([l, t, r, b], dim=2) # LxTx4

def apply_distance_constraints(reg_targets, anchor_sizes):
    """
    Args:
        reg_targets: LxTx4
        anchor_bb_sizes: Lx2
    Returns:
        A LxT tensor where value at (anchor, bbox) is true if the max value to regress at this anchor for this bbox is inside the bounds associated to this anchor
        If other values to regress than the max are negatives, it is dealt with anchor_in_boxes.
    """
    max_reg_targets, _ = reg_targets.max(dim=2) # LxT
    min_reg_targets, _ = reg_targets.min(dim=2) # LxT
    dist_constraints = torch.stack([
        min_reg_targets > 0,
        max_reg_targets >= anchor_sizes[:, None, 0],
        max_reg_targets <= anchor_sizes[:, None, 1]
    ])
    return torch.all(dist_constraints, dim=0)

def anchor_bbox_area(bbox, anchors, fits_to_feature_level):
    """
    Args: bbox is XYWH
    Returns: 
        Tensor LxT where value at (anchor, bbox) is the area of bbox if anchor is in bbox and anchor is associated with bbox of that size
        Else INF.
    """
    #bbox_areas = _calc_bbox_area(bbox_targets) # T
    bbox_areas = bbox[:, 2] * bbox[:, 3] # T
    # area of each target bbox repeated for each loc with inf where the the loc is not 
    # in the target bbox or if the loc is not at the right level for this bbox size
    anchor_bbox_area = bbox_areas[None].repeat(len(anchors), 1) # LxT
    anchor_bbox_area[~fits_to_feature_level] = INF
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
        reg_targets = calculate_reg_targets(
            anchors, bbox_targets) # LxTx4, LxT
        fits_to_feature_level = apply_distance_constraints(
            reg_targets, anchors_bb_sizes) # LxT
        locations_to_gt_area = anchor_bbox_area(
            bbox_targets, anchors, fits_to_feature_level)
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

## FCOS lightning module

class FCOS(pl.LightningModule):
    def __init__(
        self,
        class_list,
        model,
        loss,
        optimizer,
        scheduler,
        #scheduler_interval,
        input_size,
        det_size_bounds,
        pre_nms_thresh=0.3,
        pre_nms_top_n=100000,
        nms_thresh=0.45,
        fpn_post_nms_top_n=50,
        min_size=0,
        *args,
        **kwargs
    ):
        super().__init__()
        self.class_list = class_list
        self.num_classes = len(class_list)
        self.model = model(num_classes=self.num_classes)
        self.loss = loss(num_classes=self.num_classes)
        self.optimizer = optimizer
        self.scheduler = scheduler
        #self.scheduler_interval = scheduler_interval
        self.map_metric = MeanAveragePrecision(
            box_format='xywh', # make sure your dataset outputs target in xywh format
            backend='faster_coco_eval'
        )
        feature_maps_sizes = [input_size/s for s in self.model.strides]
        assert all(map(lambda x: x.is_integer(), feature_maps_sizes))
        feature_maps_sizes = [(int(fms), int(fms)) for fms in feature_maps_sizes]
        anchors, anchor_sizes = get_all_anchors_bb_sizes(
            fm_sizes=feature_maps_sizes,
            fm_strides=self.model.strides,
            bb_sizes=det_size_bounds
        )
        self.register_buffer('anchors', anchors) # Lx2
        self.register_buffer('anchor_sizes', anchor_sizes) # Lx2
                
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
    
    def configure_optimizers(self):
        train_params = list(filter(lambda p: p[1].requires_grad, self.named_parameters()))
        nb_train = sum([int(torch.numel(p[1])) for p in train_params])
        nb_tot = sum([int(torch.numel(p)) for p in self.parameters()])
        print(f"Training {nb_train} params out of {nb_tot}")
        optimizer = self.optimizer(params=[p[1] for p in train_params])
        scheduler = self.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }

    def forward(self, x):
        return self.model(x)    

    def post_process_batch(
        self,
        cls_preds, # B x L x C 
        reg_preds, # B x L x 4
        cness_preds, # B x L x 1
        input_size
    ): 
        preds = []
        for logits, ltrb, cness in zip(cls_preds, reg_preds, cness_preds):
            boxes, scores, classes = self.post_process(logits, ltrb, cness, input_size)
            preds.append({'boxes': boxes, 'scores': scores, 'labels': classes})
        return preds
    
    def post_process(
        self,
        logits,
        ltrb,
        cness,
        input_size
    ):
        probas = logits.sigmoid() # LxC
        high_probas = probas > self.pre_nms_thresh # LxC
        # Indices on L and C axis of high prob pairs anchor/class
        high_prob_anchors_idx, high_prob_cls = high_probas.nonzero(as_tuple=True) # dim l <= L*C
        high_prob_cls += 1 # 0 is for no object
        high_prob_ltrb = ltrb[high_prob_anchors_idx] # lx4
        high_prob_anchors = self.anchors[high_prob_anchors_idx] # lx2
        # Tensor shape l with values from logits*cness such that logits > pre_nms_thresh 
        cness_modulated_probas = probas * cness.sigmoid() # LxC
        high_prob_scores = cness_modulated_probas[high_probas] # l
        # si l est trop longue
        if high_probas.sum().item() > self.pre_nms_top_n:
            # Filter the pre_nms_top_n most probable pairs 
            high_prob_scores, top_k_indices = high_prob_scores.topk(
                self.pre_nms_top_n, sorted=False) 
            high_prob_cls = high_prob_cls[top_k_indices]
            high_prob_ltrb = high_prob_ltrb[top_k_indices]
            high_prob_anchors = high_prob_anchors[top_k_indices]

        # Rewrites bbox (x0,y0,x1,y1) from reg targets (l,t,r,b) following eq (1) in paper
        high_prob_boxes = torch.stack([
            high_prob_anchors[:, 0] - high_prob_ltrb[:, 0],
            high_prob_anchors[:, 1] - high_prob_ltrb[:, 1],
            high_prob_anchors[:, 0] + high_prob_ltrb[:, 2],
            high_prob_anchors[:, 1] + high_prob_ltrb[:, 3],
        ], dim=1)

        high_prob_boxes = torchvision.ops.clip_boxes_to_image(high_prob_boxes, input_size)
        big_enough_box_idxs = torchvision.ops.remove_small_boxes(high_prob_boxes, self.min_size)
        boxes = high_prob_boxes[big_enough_box_idxs]
        # Why not do that on scores and classes too ? 
        classes = high_prob_cls[big_enough_box_idxs]
        scores = high_prob_scores[big_enough_box_idxs]
        #high_prob_scores = torch.sqrt(high_prob_scores) # WHY SQRT ? REmOVED
        # NMS expects boxes to be in xyxy format
        nms_idxs = torchvision.ops.nms(boxes, scores, self.nms_thresh)
        boxes = boxes[nms_idxs]
        scores = scores[nms_idxs]
        classes = classes[nms_idxs]
        if len(nms_idxs) > self.fpn_post_nms_top_n:
            image_thresh, _ = torch.kthvalue(
                scores.cpu(),
                len(nms_idxs) - self.fpn_post_nms_top_n + 1)
            keep = scores >= image_thresh.item()
            #keep = torch.nonzero(keep).squeeze(1)
            boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        # Then back to xywh boxes for preds and metric computation
        boxes[:, 2] -= boxes[:, 0]
        boxes[:, 3] -= boxes[:, 1]
        # Isn't this cond auto valid from the beginning filter ?
        #keep = scores >= pre_nms_thresh
        #boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        return boxes, scores, classes 
    
    def training_step(self, batch, batch_idx):
        
        image = batch["sup"]["image"]               
        cls_logits, bbox_reg, centerness = self.model(image)
        cls_tgts, reg_tgts = associate_targets_to_anchors(
            batch["sup"]['target'],
            self.anchors,
            self.anchor_sizes
        ) # BxNumAnchors, BxNumAnchorsx4    
        losses = self.loss(
            cls_logits,
            bbox_reg,
            centerness,
            cls_tgts,
            reg_tgts
        )
        loss = losses['combined_loss']
        self.log(f"Loss/train", loss.detach().item())
        self.log(f"cls_loss/train", losses['cls_loss'].detach().item())
        self.log(f"reg_loss/train", losses['reg_loss'].detach().item())
        self.log(f"centerness_loss/train", losses['centerness_loss'].detach().item())
        return loss
        
    def validation_step(self, batch, batch_idx):
        
        cls_tgts, reg_tgts = associate_targets_to_anchors(
            batch['target'],
            self.anchors,
            self.anchor_sizes
        )
        image = batch["image"]
        cls_logits, bbox_reg, centerness = self.model(image)
        losses = self.loss(
            cls_logits,
            bbox_reg,
            centerness,
            cls_tgts,
            reg_tgts
        )
        loss = losses['combined_loss']
        self.log(f"Loss/val", loss.detach().item())
        b,c,h,w = image.shape           
        preds = self.post_process_batch(
            cls_logits,
            bbox_reg,
            centerness,
            (h,w),
        )
        self.map_metric.update(preds, batch['target'])
        
    def on_validation_epoch_end(self):
        mapmetric = self.map_metric.compute()['map']
        self.log("map/val", mapmetric)
        self.map_metric.reset()
        
    #def predict_step(self):
    #    image = batch["image"]
    #    cls_logits, bbox_reg, centerness = self.model(image)