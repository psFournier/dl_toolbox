import torch
import torch.nn as nn
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torchvision
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from torchvision.models import resnet50, ResNet50_Weights
from dl_toolbox.utils import associate_targets_to_anchors, get_all_anchors_bb_sizes
import math


class Scale(nn.Module):

    def __init__(self, init_value=1.0):
        super(Scale, self).__init__()
        self.scale = nn.Parameter(torch.FloatTensor([init_value]))

    def forward(self, x):
        return x * self.scale


class Head(nn.Module):

    def __init__(self, in_channels, n_classes, n_share_convs=4, n_feat_levels=5):
        super().__init__()

        #tower = []
        cls_tower = []
        bbox_tower = []
        for _ in range(n_share_convs):
            cls_tower.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=True))
            cls_tower.append(nn.GroupNorm(32, in_channels))
            cls_tower.append(nn.ReLU())
            bbox_tower.append(
                nn.Conv2d(in_channels,
                          in_channels,
                          kernel_size=3,
                          stride=1,
                          padding=1,
                          bias=True))
            bbox_tower.append(nn.GroupNorm(32, in_channels))
            bbox_tower.append(nn.ReLU())
        self.cls_layers = nn.Sequential(*cls_tower)
        self.bbox_layers = nn.Sequential(*bbox_tower)

        self.cls_logits = nn.Conv2d(in_channels,
                                    n_classes,
                                    kernel_size=3,
                                    stride=1,
                                    padding=1)
        self.bbox_pred = nn.Conv2d(in_channels,
                                   4,
                                   kernel_size=3,
                                   stride=1,
                                   padding=1)
        self.ctrness = nn.Conv2d(in_channels,
                                 1,
                                 kernel_size=3,
                                 stride=1,
                                 padding=1)
        self.scales = nn.ModuleList([Scale(init_value=1.0) for _ in range(n_feat_levels)])

        # initialize the bias for focal loss
        prior_prob = 0.01
        bias_value = -math.log((1 - prior_prob) / prior_prob)
        torch.nn.init.constant_(self.cls_logits.bias, bias_value)

    def forward(self, x):
        cls_logits = []
        bbox_preds = []
        cness_preds = []
        for l, features in enumerate(x):
            cls_features = self.cls_layers(features) # BxinChannelsxfeatSize
            bbox_features = self.bbox_layers(features)
            cls_logits.append(self.cls_logits(cls_features).flatten(-2)) # BxNumClsxFeatSize
            cness_preds.append(self.ctrness(bbox_features).flatten(-2)) # Bx1xFeatSize
            reg = self.bbox_pred(bbox_features) # Bx4xFeatSize
            reg = self.scales[l](reg)
            bbox_preds.append(torch.exp(reg).flatten(-2))
        all_logits = torch.cat(cls_logits, dim=-1).permute(0,2,1) # BxNumAnchorsxC
        all_box_regs = torch.cat(bbox_preds, dim=-1).permute(0,2,1) # BxNumAnchorsx4
        all_cness = torch.cat(cness_preds, dim=-1).permute(0,2,1) # BxNumAnchorsx1
        return all_logits, all_box_regs, all_cness


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
    

class FCOS(pl.LightningModule):
    def __init__(
        self,
        class_list,
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
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.map_metric = MeanAveragePrecision(
            box_format='xywh', # make sure your dataset outputs target in xywh format
            backend='faster_coco_eval'
        )
        
        self.feature_extractor = create_feature_extractor(
            resnet50(weights=ResNet50_Weights.IMAGENET1K_V2), 
            {
                'layer2.3.relu_2': 'layer2', # 1/8th feat map
                'layer3.5.relu_2': 'layer3', # 1/16
                'layer4.2.relu_2': 'layer4', # 1/32
            }
        )
        inp = torch.randn(2, 3, 224, 224)
        with torch.no_grad():
            out = self.feature_extractor(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        self.fpn = FeaturePyramidNetwork(
            in_channels_list,
            out_channels=256,
            extra_blocks=LastLevelP6P7(256,256)
        )
        
        features = nn.Sequential(self.feature_extractor, self.fpn)
        inp = torch.randn(2, 3, 640, 640)
        with torch.no_grad():
            out = features(inp)
        fm_sizes = [o.shape[2:] for o in out.values()]
        print(fm_sizes)
        
        self.head = Head(256, self.num_classes)
        
        fm_strides = [8, 16, 32, 64, 128] 
        bb_sizes = [64, 128, 256, 512] 
        anchors, anchor_sizes = get_all_anchors_bb_sizes(
            fm_sizes, fm_strides, bb_sizes)
        self.register_buffer('anchors', anchors) # Lx2
        self.register_buffer('anchor_sizes', anchor_sizes) # Lx2
        self.loss = LossEvaluator(self.num_classes)
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
                "interval": "epoch"
            },
        }

    def forward(self, x):
        features = self.feature_extractor(x)
        feature_maps = list(self.fpn(features).values()) # feature maps from FPN
        #features = list(self.features(x).values()) # feature maps from FPN
        box_cls, box_regression, centerness = self.head(feature_maps)
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
        b = batch['sup']
        cls_logits, bbox_reg, centerness = self.forward(b['image']) # BxNumAnchorsxC, BxNumAnchorsx4, BxNumx1
        cls_tgts, reg_tgts = associate_targets_to_anchors(
            b['target'], self.anchors, self.anchor_sizes) # BxNumAnchors, BxNumAnchorsx4
        losses = self.loss(cls_logits, bbox_reg, centerness, cls_tgts, reg_tgts)
        train_loss = losses["combined_loss"]
        self.log(f"loss/train", train_loss.detach().item())
        self.train_losses.append(train_loss.detach().item())
        #preds = self.post_process(cls_logits, bbox_reg, centerness, x.shape[-1])
        #self.map_metric.update(preds, targets)
        return train_loss
        
    def validation_step(self, batch, batch_idx):
        cls_logits, bbox_reg, centerness = self.forward(batch['image']) # BxNumAnchorsxC, BxNumAnchorsx4, BxNumx1
        cls_tgts, reg_tgts = associate_targets_to_anchors(
            batch['target'], self.anchors, self.anchor_sizes) # BxNumAnchors, BxNumAnchorsx4
        losses = self.loss(cls_logits, bbox_reg, centerness, cls_tgts, reg_tgts)
        val_loss = losses["combined_loss"]
        self.log(f"Total loss/val", val_loss.detach().item())
        preds = self.post_process(cls_logits, bbox_reg, centerness, batch['image'].shape[-1])
        self.map_metric.update(preds, batch['target'])
        self.val_losses.append(val_loss.detach().item())
        
    def on_train_epoch_end(self):
        train_loss = sum(self.train_losses)/len(self.train_losses)
        print(f'\n{train_loss=}')
        self.train_losses.clear()
        
    def on_validation_epoch_end(self):
        mapmetric = self.map_metric.compute()['map']
        self.log("map/val", mapmetric)
        #print("\nMAP: ", mapmetric)
        self.map_metric.reset()
        val_loss = sum(self.val_losses)/len(self.val_losses)
        print(f'\n{val_loss=}')
        self.val_losses.clear()