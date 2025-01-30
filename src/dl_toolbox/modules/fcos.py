import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision
from dl_toolbox.utils import associate_targets_to_anchors, get_all_anchors_bb_sizes

## LOSS

class LossEvaluator(nn.Module):

    def __init__(self, num_classes):
        super(LossEvaluator, self).__init__()
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.num_classes = num_classes
                
    def __call__(self, cls_logits, reg_preds, cness_preds, cls_tgts, reg_tgts):
        pos_inds_b, pos_inds_loc = torch.nonzero(cls_tgts > 0, as_tuple=True)
        num_pos = len(pos_inds_b)
        reg_preds = reg_preds[pos_inds_b, pos_inds_loc, :]
        reg_tgts = reg_tgts[pos_inds_b, pos_inds_loc, :]
        cness_preds = cness_preds[pos_inds_b, pos_inds_loc, :].squeeze(-1)
        cness_tgts = self._compute_centerness_targets(reg_tgts)
        cls_loss = self._get_cls_loss(cls_logits, cls_tgts, max(num_pos, 1.))
        reg_loss, centerness_loss = 0,0
        if num_pos > 0:
            reg_loss = self._get_reg_loss(
                reg_preds, reg_tgts, cness_tgts)
            centerness_loss = self._get_centerness_loss(
                cness_preds, cness_tgts, num_pos)
        losses = {}
        losses["cls_loss"] = cls_loss
        losses["reg_loss"] = reg_loss
        losses["centerness_loss"] = centerness_loss
        losses["combined_loss"] = cls_loss + reg_loss + centerness_loss
        return losses
    
    def _compute_centerness_targets(self, reg_tgts):
        """
        Args:
            reg_tgts: l, t, r, b values to regress, shape BxNumAx4
        Returns:
            A tensor of shape BxNumA giving how centered each anchor is for the bbox it must regress
        """
        if len(reg_tgts) == 0:
            return reg_tgts.new_zeros(len(reg_tgts))
        left_right = reg_tgts[..., [0, 2]]
        top_bottom = reg_tgts[..., [1, 3]]
        centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                    (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
        return torch.sqrt(centerness)

    def _get_cls_loss(self, cls_preds, cls_targets, num_pos_samples):
        """
        cls_targets takes values in 0...C, 0 only when there is no obj to be detected for the anchor
        """
        onehot = nn.functional.one_hot(cls_targets.long(), self.num_classes+1)[...,1:].float()
        cls_loss = torchvision.ops.sigmoid_focal_loss(cls_preds, onehot)
        return cls_loss.sum() / num_pos_samples

    def _get_reg_loss(self, reg_preds, reg_targets, centerness_targets):
        ltrb_preds = reg_preds.reshape(-1, 4)
        ltrb_tgts = reg_targets.reshape(-1, 4)
        xyxy_preds = torch.cat([-ltrb_preds[:,:2], ltrb_preds[:,2:]], dim=1) 
        xyxy_tgts = torch.cat([-ltrb_tgts[:,:2], ltrb_tgts[:,2:]], dim=1)
        reg_losses = torchvision.ops.distance_box_iou_loss(xyxy_preds, xyxy_tgts, reduction='none')
        sum_centerness_targets = centerness_targets.sum()
        reg_loss = (reg_losses * centerness_targets).sum() / sum_centerness_targets
        return reg_loss

    def _get_centerness_loss(self, centerness_preds, centerness_targets,
                             num_pos_samples):
        centerness_loss = self.centerness_loss_func(centerness_preds,
                                                    centerness_targets)
        return centerness_loss / num_pos_samples
    
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
        optimizer,
        scheduler,
        pre_nms_thresh=0.3,
        pre_nms_top_n=1000,
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
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.map_metric = MeanAveragePrecision(
            box_format='xywh', # make sure your dataset outputs target in xywh format
            backend='faster_coco_eval'
        )
        
        #self.feature_extractor = create_feature_extractor(
        #    resnet50(weights=ResNet50_Weights.IMAGENET1K_V2), 
        #    {
        #        'layer2.3.relu_2': 'layer2', # 1/8th feat map
        #        'layer3.5.relu_2': 'layer3', # 1/16
        #        'layer4.2.relu_2': 'layer4', # 1/32
        #    }
        #)
        #inp = torch.randn(2, 3, 224, 224)
        #with torch.no_grad():
        #    out = self.feature_extractor(inp)
        #in_channels_list = [o.shape[1] for o in out.values()]
        #self.fpn = FeaturePyramidNetwork(
        #    in_channels_list,
        #    out_channels=256,
        #    extra_blocks=LastLevelP6P7(256,256)
        #)
        #
        #features = nn.Sequential(self.feature_extractor, self.fpn)
        #inp = torch.randn(2, 3, 640, 640)
        #with torch.no_grad():
        #    out = features(inp)
        #fm_sizes = [o.shape[2:] for o in out.values()]
        #print(fm_sizes)
        #
        #self.head = Head(256, self.num_classes)
        fm_sizes=[(160,160),(80,80),(40,40),(20,20)]
        fm_strides = [8, 16, 32, 64, 128] 
        bb_sizes = [64, 128, 256, 512] 
        anchors, anchor_sizes = get_all_anchors_bb_sizes(
            fm_sizes, fm_strides, bb_sizes)
        self.register_buffer('anchors', anchors) # Lx2
        self.register_buffer('anchor_sizes', anchor_sizes) # Lx2
        
        self.criterion = LossEvaluator(self.num_classes)
        
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
                "interval": "epoch"
            },
        }

    def forward(self, x):
        return self.model(x)
        #features = self.feature_extractor(x)
        #feature_maps = list(self.fpn(features).values()) # feature maps from FPN
        ##features = list(self.features(x).values()) # feature maps from FPN
        #box_cls, box_regression, centerness = self.head(feature_maps)
        #return box_cls, box_regression, centerness
        #return self.network(x)
    
    #def predict(self, x):
    #    cls_logits, bbox_reg, centerness = self.forward(x)
    #    preds = self.post_process(cls_logits, bbox_reg, centerness, x.shape[-1])
    #    return preds
    

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
        losses = self.criterion(
            cls_logits,
            bbox_reg,
            centerness,
            cls_tgts,
            reg_tgts
        )
        loss = losses['combined_loss']
        self.log(f"Loss/train", loss.detach().item())
        return loss
        
        #b = batch['sup']
        #cls_logits, bbox_reg, centerness = self.forward(b['image']) # BxNumAnchorsxC, BxNumAnchorsx4, BxNumx1
        #cls_tgts, reg_tgts = associate_targets_to_anchors(
        #    b['target'], self.anchors, self.anchor_sizes) # BxNumAnchors, BxNumAnchorsx4
        #losses = self.loss(cls_logits, bbox_reg, centerness, cls_tgts, reg_tgts)
        #train_loss = losses["combined_loss"]
        #self.log(f"Loss/train", train_loss.detach().item())
        #self.train_losses.append(train_loss.detach().item())
        ##preds = self.post_process(cls_logits, bbox_reg, centerness, x.shape[-1])
        ##self.map_metric.update(preds, targets)
        #return train_loss
        
    def validation_step(self, batch, batch_idx):
        
        cls_tgts, reg_tgts = associate_targets_to_anchors(
            batch['target'],
            self.anchors,
            self.anchor_sizes
        )
        image = batch["image"]
        cls_logits, bbox_reg, centerness = self.model(image)
        losses = self.criterion(
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
            
        #cls_logits, bbox_reg, centerness = self.forward(batch['image']) # BxNumAnchorsxC, BxNumAnchorsx4, BxNumx1
        #cls_tgts, reg_tgts = associate_targets_to_anchors(
        #    batch['target'], self.anchors, self.anchor_sizes) # BxNumAnchors, BxNumAnchorsx4
        #losses = self.loss(cls_logits, bbox_reg, centerness, cls_tgts, reg_tgts)
        #val_loss = losses["combined_loss"]
        #self.log(f"Loss/val", val_loss.detach().item())
        #preds = self.post_process(cls_logits, bbox_reg, centerness, batch['image'].shape[-1])
        #self.map_metric.update(preds, batch['target'])
        #self.val_losses.append(val_loss.detach().item())
        
    #def on_train_epoch_end(self):
    #    train_loss = sum(self.train_losses)/len(self.train_losses)
    #    print(f'\n{train_loss=}')
    #    self.train_losses.clear()
        
    def on_validation_epoch_end(self):
        mapmetric = self.map_metric.compute()['map']
        self.log("map/val", mapmetric)
        self.map_metric.reset()
