import torch.nn as nn
import torch
import torchvision
INF = 100000000
import torch.nn.functional as F

import pytorch_lightning as pl
from dl_toolbox.callbacks import ProgressBar
import gc 

import pytorch_lightning as pl
from dl_toolbox.callbacks import ProgressBar
from dl_toolbox import datamodules
import torchvision.transforms.v2 as v2
from functools import partial

import torch
from dl_toolbox.callbacks import ProgressBar, Lora
from functools import partial
import gc


train_tf = v2.Compose(
    [
        v2.Resize(size=480, max_size=640),
        v2.RandomCrop(size=(640,640), pad_if_needed=True, fill=0),
        v2.SanitizeBoundingBoxes(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

test_tf = v2.Compose(
    [
        v2.Resize(size=480, max_size=640),
        v2.RandomCrop(size=(640,640), pad_if_needed=True, fill=0),
        v2.SanitizeBoundingBoxes(),
        v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ]
)

dm = datamodules.Coco(
    data_path='/data',
    train_tf=train_tf,
    test_tf=test_tf,
    batch_tf=None,
    batch_size=4,
    num_workers=7,
    pin_memory=False
)

#dm = datamodules.xView(
#    data_path='/data',
#    merge='building',
#    train_tf=train_tf,
#    test_tf=test_tf,
#    batch_tf=None,
#    batch_size=2,
#    num_workers=1,
#    pin_memory=False
#)

lora = Lora('backbone', 4, True)

num_classes = dm.num_classes

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

# test
anchors = get_fm_anchors(14, 16, 8)

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
    anchors, anchors_bb_sizes = [], []
    for l, ((h,w), s) in enumerate(zip(fm_sizes, fm_strides)):
        fm_anchors = get_fm_anchors(h, w, s)
        sizes = torch.tensor([bb_sizes[l], bb_sizes[l+1]], dtype=torch.float32)
        sizes = sizes.repeat(len(fm_anchors)).view(len(fm_anchors), 2)
        anchors.append(fm_anchors)
        anchors_bb_sizes.append(sizes)
    return torch.cat(anchors, 0), torch.cat(anchors_bb_sizes, 0)
#test
all_anchors = get_all_anchors_bb_sizes([(4,4),(2,2)], [8, 16], [-1, 64, 128])
#all_anchors

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

def _compute_centerness_targets(reg_tgts):
    """
    Args:
        reg_tgts: l, t, r, b values to regress, shape BxNumAx4
    Returns:
        A tensor of shape BxNumA giving how centered each anchor is for the bbox it must regress
    """
    left_right = reg_tgts[..., [0, 2]]
    top_bottom = reg_tgts[..., [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)

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
        cness_tgts = _compute_centerness_targets(reg_tgts)
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

    def _get_cls_loss(self, cls_preds, cls_targets, num_pos_samples):
        """
        cls_targets takes values in 0...C, 0 only when there is no obj to be detected for the anchor
        """
        onehot = F.one_hot(cls_targets.long(), self.num_classes+1)[...,1:].float()
        cls_loss = torchvision.ops.sigmoid_focal_loss(cls_preds, onehot)
        return cls_loss.sum() / num_pos_samples

    def _get_reg_loss(self, reg_preds, reg_targets, centerness_targets):
        reg_preds = reg_preds.reshape(-1, 4)
        reg_targets = reg_targets.reshape(-1, 4)
        reg_losses = torchvision.ops.distance_box_iou_loss(reg_preds, reg_targets, reduction='none')
        sum_centerness_targets = centerness_targets.sum()
        reg_loss = (reg_losses * centerness_targets).sum() / sum_centerness_targets
        return reg_loss

    def _get_centerness_loss(self, centerness_preds, centerness_targets,
                             num_pos_samples):
        centerness_loss = self.centerness_loss_func(centerness_preds,
                                                    centerness_targets)
        return centerness_loss / num_pos_samples
    
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from resnet_fcos import Head

class FCOS(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        out_channels,
        optimizer,
        scheduler,
        tta=None,
        sliding=None,
        pre_nms_thresh=0.3,
        pre_nms_top_n=1000,
        nms_thresh=0.45,
        fpn_post_nms_top_n=50,
        min_size=0,
        *args,
        **kwargs
    ):
        super().__init__()
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.map_metric = MeanAveragePrecision(
            box_format='xywh', # make sure your dataset outputs target in xywh format
            backend='faster_coco_eval'
        )
        self.sliding = sliding
        
        self.backbone = create_feature_extractor(
            resnet50(weights=ResNet50_Weights.IMAGENET1K_V2), 
            {
                'layer2.3.relu_2': 'layer2', # 1/8th feat map
                'layer3.5.relu_2': 'layer3', # 1/16
                'layer4.2.relu_2': 'layer4', # 1/32
            }
        )
        inp = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = self.backbone(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        fpn = FeaturePyramidNetwork(
            in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelP6P7(out_channels,out_channels)
        )
        self.features = nn.Sequential(self.backbone, fpn)
        inp = torch.randn(2, 3, 640, 640)
        with torch.no_grad():
            out = self.features(inp)
        fm_sizes = [o.shape[2:] for o in out.values()]
        self.head = Head(out_channels, num_classes)
        
        fm_strides = [8, 16, 32, 64, 128] 
        bb_sizes = [-1, 64, 128, 256, 512, INF] 
        anchors, anchor_sizes = get_all_anchors_bb_sizes(
            fm_sizes, fm_strides, bb_sizes)
        self.register_buffer('anchors', anchors) # Lx2
        self.register_buffer('anchor_sizes', anchor_sizes) # Lx2
        self.loss = LossEvaluator(num_classes)
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        
        self.train_losses = []
        self.val_losses = []
    
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
        features = list(self.features(x).values()) # feature maps from FPN
        box_cls, box_regression, centerness = self.head(features)
        return box_cls, box_regression, centerness
        #return self.network(x)
    
    def _post_process(self, logits, ltrb, cness, image_size):
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
        boxes = torch.stack([
            high_prob_anchors[:, 0] - high_prob_ltrb[:, 0],
            high_prob_anchors[:, 1] - high_prob_ltrb[:, 1],
            high_prob_anchors[:, 0] + high_prob_ltrb[:, 2],
            high_prob_anchors[:, 1] + high_prob_ltrb[:, 3],
        ], dim=1)
        
        boxes = torchvision.ops.clip_boxes_to_image(boxes, (image_size, image_size))
        big_enough_box_idxs = torchvision.ops.remove_small_boxes(boxes, self.min_size)
        # Why not do that on scores and classes too ? DONE
        boxes = boxes[big_enough_box_idxs]
        scores = high_prob_scores[big_enough_box_idxs]
        classes = high_prob_cls[big_enough_box_idxs]
        #scores = torch.sqrt(high_prob_scores) # WHY SQRT ? REmOVED
        # NMS expects boxes to be in xyxy format
        nms_idxs = torchvision.ops.nms(boxes, scores, self.nms_thresh)
        # Then back to xywh boxes for preds and metric computation
        boxes[:, 2] += boxes[:, 0]
        boxes[:, 3] += boxes[:, 1]
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

        keep = scores >= self.pre_nms_thresh
        boxes, scores, classes = boxes[keep], scores[keep], classes[keep]
        return boxes, scores, classes 
    
    def post_process(
        self,
        cls_preds, # B x L x C 
        reg_preds, # B x L x 4
        cness_preds, # B x L x 1
        image_size
    ): 
        B, L, C = cls_preds.shape
        all_boxes = []
        all_classes = []
        all_scores = []
        for i in range(B):
            logits = cls_preds[i]
            ltrb = reg_preds[i]
            cness = cness_preds[i]
            boxes, scores, classes = self._post_process(logits, ltrb, cness, image_size)
            all_boxes.append(boxes)
            all_classes.append(classes)
            all_scores.append(scores)
        predictions = [{'boxes': bb, 'scores': s, 'labels': l} for bb,s,l in 
                       zip(all_boxes, all_scores, all_classes)]
        return predictions
    
    def training_step(self, batch, batch_idx):
        x, targets, paths = batch["sup"] #targets is a list of dict
        cls_logits, bbox_reg, centerness = self.forward(x) # BxNumAnchorsxC, BxNumAnchorsx4, BxNumx1
        cls_tgts, reg_tgts = associate_targets_to_anchors(
            targets, self.anchors, self.anchor_sizes) # BxNumAnchors, BxNumAnchorsx4
        losses = self.loss(cls_logits, bbox_reg, centerness, cls_tgts, reg_tgts)
        train_loss = losses["combined_loss"]
        self.log(f"loss/train", train_loss.detach().item())
        self.train_losses.append(train_loss.detach().item())
        #preds = self.post_process(cls_logits, bbox_reg, centerness, x.shape[-1])
        #self.map_metric.update(preds, targets)
        return train_loss
        
    def validation_step(self, batch, batch_idx):
        x, targets, paths = batch
        cls_logits, bbox_reg, centerness = self.forward(x) # BxNumAnchorsxC, BxNumAnchorsx4, BxNumx1
        cls_tgts, reg_tgts = associate_targets_to_anchors(
            targets, self.anchors, self.anchor_sizes) # BxNumAnchors, BxNumAnchorsx4
        losses = self.loss(cls_logits, bbox_reg, centerness, cls_tgts, reg_tgts)
        val_loss = losses["combined_loss"]
        self.log(f"Total loss/val", val_loss.detach().item())
        preds = self.post_process(cls_logits, bbox_reg, centerness, x.shape[-1])
        self.map_metric.update(preds, targets)
        self.val_losses.append(val_loss.detach().item())
        
    def on_train_epoch_end(self):
        train_loss = sum(self.train_losses)/len(self.train_losses)
        print(f'\n{train_loss=}\n')
        self.train_losses.clear()
        
    def on_validation_epoch_end(self):
        mapmetric = self.map_metric.compute()['map']
        self.log("map/val", mapmetric)
        #print("\nMAP: ", mapmetric)
        self.map_metric.reset()
        val_loss = sum(self.val_losses)/len(self.val_losses)
        print(f'\n{val_loss=}\n')
        self.val_losses.clear()
        
trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_steps=10000,
    overfit_batches=20,
    #limit_train_batches=5,
    #limit_val_batches=5,
    callbacks=[ProgressBar(), lora]
)

module = FCOS(
    num_classes=num_classes,
    out_channels=256,
    optimizer=partial(torch.optim.SGD, lr=0.01, momentum=0.9, weight_decay=0.0001),
    scheduler=partial(torch.optim.lr_scheduler.ConstantLR, factor=1),
    pre_nms_thresh=0.05,
    pre_nms_top_n=1000,
    nms_thresh=0.6,
    fpn_post_nms_top_n=100,
    min_size=0,
)
 
gc.collect()
torch.cuda.empty_cache()
gc.collect()

trainer.fit(
    module,
    datamodule=dm,
)