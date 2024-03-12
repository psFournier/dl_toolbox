import pytorch_lightning as pl
import torch
import torch.nn.functional as F
import torchmetrics as M
import matplotlib.pyplot as plt
from pytorch_lightning.utilities import rank_zero_info
import torch.nn as nn
import math

def nms(bounding_boxes,
        confidence_scores,
        classes,
        threshold,
        class_agnostic=True):
    device = bounding_boxes.device
    if len(bounding_boxes) == 0:
        return torch.tensor([]).to(device), torch.tensor(
            []).to(device), torch.tensor([]).to(device)

    bounding_boxes = bounding_boxes.detach().cpu().numpy()
    confidence_scores = confidence_scores.detach().cpu().numpy()
    classes = classes.detach().cpu().numpy()

    start_x = bounding_boxes[:, 0]
    start_y = bounding_boxes[:, 1]
    end_x = bounding_boxes[:, 2]
    end_y = bounding_boxes[:, 3]

    picked_boxes = []
    picked_scores = []
    picked_classes = []

    areas = (end_x - start_x + 1) * (end_y - start_y + 1)

    order = np.argsort(confidence_scores)

    while order.size > 0:
        index = order[-1]

        picked_boxes.append(bounding_boxes[index])
        picked_scores.append(confidence_scores[index])
        picked_classes.append(classes[index])

        order = order[:-1]
        if len(order) == 0:
            break

        x1 = np.maximum(start_x[index], start_x[order])
        x2 = np.minimum(end_x[index], end_x[order])
        y1 = np.maximum(start_y[index], start_y[order])
        y2 = np.minimum(end_y[index], end_y[order])

        w = np.maximum(0.0, x2 - x1 + 1)
        h = np.maximum(0.0, y2 - y1 + 1)
        intersection = w * h
        ratio = intersection / (areas[index] + areas[order] - intersection)

        if not class_agnostic:
            other_classes = classes[order] != classes[index]
            ratio[other_classes] = 0.0

        left = np.where(ratio < threshold)
        order = order[left]

    outputs = [
        torch.tensor(np.array(picked_boxes)).to(device),
        torch.tensor(np.array(picked_scores)).to(device),
        torch.tensor(np.array(picked_classes)).to(device)
    ]

    return outputs


def _convert_bbox_xywh(bbox):
    xmin, ymin, xmax, ymax = _split_into_xyxy(bbox)
    bbox = torch.cat((xmin, ymin, xmax - xmin + 1, ymax - ymin + 1), dim=-1)
    return bbox


def _split_into_xyxy(bbox):
    xmin, ymin, w, h = bbox.split(1, dim=-1)
    return (
        xmin,
        ymin,
        xmin + (w - 1).clamp(min=0),
        ymin + (h - 1).clamp(min=0),
    )


def remove_small_boxes(boxlist, min_size):
    xywh_boxes = _convert_bbox_xywh(boxlist)
    _, _, ws, hs = xywh_boxes.unbind(dim=1)
    keep = ((ws >= min_size) & (hs >= min_size)).nonzero().squeeze(1)
    return boxlist[keep]


def _clip_to_image(bboxes, image_size):
    h, w = image_size
    bboxes[:, 0].clamp_(min=0, max=h - 1)
    bboxes[:, 1].clamp_(min=0, max=w - 1)
    bboxes[:, 2].clamp_(min=0, max=h - 1)
    bboxes[:, 3].clamp_(min=0, max=w - 1)
    return bboxes


class FCOSPostProcessor(torch.nn.Module):

    def __init__(self, pre_nms_thresh, pre_nms_top_n, nms_thresh,
                 fpn_post_nms_top_n, min_size, num_classes):
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes

    def forward_for_single_feature_map(self, locations, cls_preds, reg_preds,
                                       cness_preds, image_size):
        B, C, _, _ = cls_preds.shape

        cls_preds = cls_preds.permute(0, 2, 3, 1).reshape(B, -1, C).sigmoid() # BxHWxC in [0,1]
        reg_preds = reg_preds.permute(0, 2, 3, 1).reshape(B, -1, 4)
        cness_preds = cness_preds.permute(0, 2, 3, 1).reshape(B, -1).sigmoid()

        candidate_inds = cls_preds > self.pre_nms_thresh # BxHWxC
        pre_nms_top_n = candidate_inds.reshape(B, -1).sum(1) # B
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)

        cls_preds = cls_preds * cness_preds[:, :, None] # BxHWxC
        
        # Conversion en liste de bbox,scores,cls par image du batch
        # POURQUOI le filtre cls_preds > nms_thresh arrive pas après la mul par cness_preds ?
        bboxes = []
        cls_labels = []
        scores = []
        for i in range(B):
            per_cls_preds = cls_preds[i] # HWxC
            per_candidate_inds = candidate_inds[i] # HWxC
            # tenseur de taille L avec les elem de cls_preds*centerness tels que cls_preds > nms_thresh
            per_cls_preds = per_cls_preds[per_candidate_inds] 
            
            # tenseur de taille Lx2 avec les indices des elem de cls_preds où > nms_thresh
            per_candidate_nonzeros = per_candidate_inds.nonzero() 
            # L : positions dans [0,HW] des elem dont cls_preds(c) > nms_thresh 
            per_box_loc = per_candidate_nonzeros[:, 0]
            # L : classe dans [1, C] des elem dont cls_preds(h,w) > nms_thresh
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_reg_preds = reg_preds[i] # HWx4
            # liste des bb des elem dont cls_preds(c) > nms_thresh 
            per_reg_preds = per_reg_preds[per_box_loc] # Lx4
            per_locations = locations[per_box_loc] # Lx2

            per_pre_nms_top_n = pre_nms_top_n[i]
            
            # si y a plus de per_prenms_topn qui passe nms_thresh (si L est trop longue)
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_cls_preds, top_k_indices = per_cls_preds.topk(
                    per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_reg_preds = per_reg_preds[top_k_indices]
                per_locations = per_locations[top_k_indices]

            detections = torch.stack([
                per_locations[:, 0] - per_reg_preds[:, 0],
                per_locations[:, 1] - per_reg_preds[:, 1],
                per_locations[:, 0] + per_reg_preds[:, 2],
                per_locations[:, 1] + per_reg_preds[:, 3],
            ],
                                     dim=1)

            detections = _clip_to_image(detections, (image_size, image_size))
            detections = remove_small_boxes(detections, self.min_size)
            bboxes.append(detections)
            cls_labels.append(per_class)
            scores.append(torch.sqrt(per_cls_preds))

        return bboxes, scores, cls_labels

    def forward(self, locations, cls_preds, reg_preds, cness_preds, image_size):
        sampled_boxes = []
        all_scores = []
        all_classes = []
        for l, o, b, c in list(zip(locations, cls_preds, reg_preds,
                                   cness_preds)):
            boxes, scores, cls_labels = self.forward_for_single_feature_map(
                l, o, b, c, image_size)

            sampled_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(cls_labels)

        all_bboxes = list(zip(*sampled_boxes))
        all_scores = list(zip(*all_scores))
        all_classes = list(zip(*all_classes))

        all_bboxes = [torch.cat(bboxes, dim=0) for bboxes in all_bboxes]
        all_scores = [torch.cat(scores, dim=0) for scores in all_scores]
        all_classes = [torch.cat(classes, dim=0) for classes in all_classes]

        boxes, scores, classes = self.select_over_all_levels(
            all_bboxes, all_scores, all_classes)

        return boxes, scores, classes

    def select_over_all_levels(self, boxlists, scores, classes):
        num_images = len(boxlists)
        all_picked_boxes, all_confidence_scores, all_classes = [], [], []
        for i in range(num_images):
            picked_boxes, confidence_scores, picked_classes = nms(
                boxlists[i], scores[i], classes[i], self.nms_thresh)

            number_of_detections = len(picked_boxes)
            if number_of_detections > self.fpn_post_nms_top_n > 0:
                image_thresh, _ = torch.kthvalue(
                    confidence_scores.cpu(),
                    number_of_detections - self.fpn_post_nms_top_n + 1)
                keep = confidence_scores >= image_thresh.item()

                keep = torch.nonzero(keep).squeeze(1)
                picked_boxes, confidence_scores, picked_classes = picked_boxes[
                    keep], confidence_scores[keep], picked_classes[keep]

            keep = confidence_scores >= self.pre_nms_thresh
            picked_boxes, confidence_scores, picked_classes = picked_boxes[
                keep], confidence_scores[keep], picked_classes[keep]

            all_picked_boxes.append(picked_boxes)
            all_confidence_scores.append(confidence_scores)
            all_classes.append(picked_classes)
        return all_picked_boxes, all_confidence_scores, all_classes

def _locations_per_level(h, w, s):
    locs_x = [i for i in range(w)]
    locs_y = [i for i in range(h)]

    locs_x = [s / 2 + x * s for x in locs_x]
    locs_y = [s / 2 + y * s for y in locs_y]
    locs = [(y, x) for x in locs_x for y in locs_y]
    return torch.tensor(locs)


def _compute_locations(features, fpn_strides):
    locations = []
    for level, feature in enumerate(features):
        h, w = feature.size()[-2:]
        locs = _locations_per_level(h, w, fpn_strides[level]).to(feature.device)
        locations.append(locs)
    return locations

INF = 100000000
MAXIMUM_DISTANCES_PER_LEVEL = [-1, 64, 128, 256, 512, INF]


def _match_reg_distances_shape(MAXIMUM_DISTANCES_PER_LEVEL, num_locs_per_level):
    level_reg_distances = []
    for m in range(1, len(MAXIMUM_DISTANCES_PER_LEVEL)):
        level_distances = torch.tensor([
            MAXIMUM_DISTANCES_PER_LEVEL[m - 1], MAXIMUM_DISTANCES_PER_LEVEL[m]
        ],
                                       dtype=torch.float32)
        locs_per_level = num_locs_per_level[m - 1]
        level_distances = level_distances.repeat(locs_per_level).view(
            locs_per_level, 2)
        level_reg_distances.append(level_distances)
    # return tensor of size sum of locs_per_level x 2
    return torch.cat(level_reg_distances, dim=0)


def _calc_bbox_area(bbox):
    return (bbox[:, 2] - bbox[:, 0] + 1.0) * (bbox[:, 3] - bbox[:, 1] + 1.0)


def _compute_centerness_targets(reg_targets):
    if len(reg_targets) == 0:
        return reg_targets.new_zeros(len(reg_targets))
    left_right = reg_targets[:, [0, 2]]
    top_bottom = reg_targets[:, [1, 3]]
    centerness = (left_right.min(dim=-1)[0] / left_right.max(dim=-1)[0]) * \
                (top_bottom.min(dim=-1)[0] / top_bottom.max(dim=-1)[0])
    return torch.sqrt(centerness)


def _calculate_reg_targets(xs, ys, bbox_targets):
    l = xs[:, None] - bbox_targets[:, 0][None] # Lx1 - 1xT -> LxT
    t = ys[:, None] - bbox_targets[:, 1][None]
    r = bbox_targets[:, 2][None] - xs[:, None]
    b = bbox_targets[:, 3][None] - ys[:, None]
    return torch.stack([l, t, r, b], dim=2) # LxTx4


def _apply_distance_constraints(reg_targets, level_distances):
    max_reg_targets, _ = reg_targets.max(dim=2)
    return torch.logical_and(max_reg_targets >= level_distances[:, None, 0], \
                             max_reg_targets <= level_distances[:, None, 1])


def _prepare_labels(locations, targets_batch):
    device = targets_batch[0].device
    # nb of locs for bbox in original image size
    num_locs_per_level = [len(l) for l in locations]
    # L = sum locs per level x 2 : for each loc in all_locs, the max size of bb authorized
    level_distances = _match_reg_distances_shape(MAXIMUM_DISTANCES_PER_LEVEL,
                                                 num_locs_per_level).to(device)
    all_locations = torch.cat(locations, dim=0).to(device) # Lx2
    xs, ys = all_locations[:, 0], all_locations[:, 1] # L & L

    all_reg_targets = []
    all_cls_targets = []
    for targets in targets_batch:
        bbox_targets = targets[:, :4] # Tx4
        cls_targets = targets[:, 4] # T
        
        # for each loc in L and each target in T, the reg target
        reg_targets = _calculate_reg_targets(xs, ys, bbox_targets) # LxTx4

        is_in_boxes = reg_targets.min(dim=2)[0] > 0 # min returns values and indices -> LxT

        fits_to_feature_level = _apply_distance_constraints(
            reg_targets, level_distances).to(device) # LxT

        bbox_areas = _calc_bbox_area(bbox_targets) # T
        
        # area of each target bbox repeated for each loc with inf where the the loc is not 
        # in the target bbox or if the loc is not at the right level for this bbox size
        locations_to_gt_area = bbox_areas[None].repeat(len(all_locations), 1) # LxT
        locations_to_gt_area[is_in_boxes == 0] = INF
        locations_to_gt_area[fits_to_feature_level == 0] = INF
        
        # for each loc, area and target idx of the target of min area at that loc
        loc_min_area, loc_mind_idxs = locations_to_gt_area.min(dim=1) # val&idx, size L, idx in [0,T-1]

        reg_targets = reg_targets[range(len(all_locations)), loc_mind_idxs] # Lx4

        cls_targets = cls_targets[loc_mind_idxs] # L
        cls_targets[loc_min_area == INF] = 0
        
        all_cls_targets.append(
            torch.split(cls_targets, num_locs_per_level, dim=0))
        all_reg_targets.append(
            torch.split(reg_targets, num_locs_per_level, dim=0))
    # all_cls_targets contains B lists of num levels elem of loc_per_levelsx1
    return _match_pred_format(all_cls_targets, all_reg_targets, locations)


def _match_pred_format(cls_targets, reg_targets, locations):
    cls_per_level = []
    reg_per_level = []
    for level in range(len(locations)):
        cls_per_level.append(torch.cat([ct[level] for ct in cls_targets],
                                       dim=0))

        reg_per_level.append(torch.cat([rt[level] for rt in reg_targets],
                                       dim=0))
    # reg_per_level is a list of num_levels tensors of size Bxnum_loc_per_levelx4
    return cls_per_level, reg_per_level


def _get_positive_samples(cls_labels, reg_labels, box_cls_preds, box_reg_preds,
                          centerness_preds, num_classes):
    box_cls_flatten = []
    box_regression_flatten = []
    centerness_flatten = []
    labels_flatten = []
    reg_targets_flatten = []
    for l in range(len(cls_labels)):
        box_cls_flatten.append(box_cls_preds[l].permute(0, 2, 3, 1).reshape(
            -1, num_classes))
        box_regression_flatten.append(box_reg_preds[l].permute(0, 2, 3,
                                                               1).reshape(
                                                                   -1, 4))
        labels_flatten.append(cls_labels[l].reshape(-1))
        reg_targets_flatten.append(reg_labels[l].reshape(-1, 4))
        centerness_flatten.append(centerness_preds[l].reshape(-1))

    cls_preds = torch.cat(box_cls_flatten, dim=0)
    cls_targets = torch.cat(labels_flatten, dim=0)
    reg_preds = torch.cat(box_regression_flatten, dim=0)
    reg_targets = torch.cat(reg_targets_flatten, dim=0)
    centerness_preds = torch.cat(centerness_flatten, dim=0)

    pos_inds = torch.nonzero(cls_targets > 0).squeeze(1)

    reg_preds = reg_preds[pos_inds]
    reg_targets = reg_targets[pos_inds]
    centerness_preds = centerness_preds[pos_inds]

    return reg_preds, reg_targets, cls_preds, cls_targets, centerness_preds, pos_inds

def sigmoid_focal_loss(logits, targets, alpha=0.25, gamma=2):
    num_classes = logits.shape[1]
    dtype = targets.dtype
    device = targets.device
    class_range = torch.arange(1, num_classes + 1, dtype=dtype,
                               device=device).unsqueeze(0)

    t = targets.unsqueeze(1)
    p = torch.sigmoid(logits)
    term1 = (1 - p)**gamma * torch.log(p)
    term2 = p**gamma * torch.log(1 - p)
    return -(t == class_range).float() * term1 * alpha - (
        (t != class_range) * (t >= 0)).float() * term2 * (1 - alpha)


class IOULoss(nn.Module):

    def __init__(self, loss_type="iou"):
        super(IOULoss, self).__init__()
        self.loss_type = loss_type

    def forward(self, pred, target, weight=None):
        pred_left = pred[:, 0]
        pred_top = pred[:, 1]
        pred_right = pred[:, 2]
        pred_bottom = pred[:, 3]

        target_left = target[:, 0]
        target_top = target[:, 1]
        target_right = target[:, 2]
        target_bottom = target[:, 3]

        target_area = (target_left + target_right) * \
                      (target_top + target_bottom)
        pred_area = (pred_left + pred_right) * \
                    (pred_top + pred_bottom)

        w_intersect = torch.min(pred_left, target_left) + torch.min(
            pred_right, target_right)
        g_w_intersect = torch.max(pred_left, target_left) + torch.max(
            pred_right, target_right)
        h_intersect = torch.min(pred_bottom, target_bottom) + torch.min(
            pred_top, target_top)
        g_h_intersect = torch.max(pred_bottom, target_bottom) + torch.max(
            pred_top, target_top)
        ac_uion = g_w_intersect * g_h_intersect + 1e-7
        area_intersect = w_intersect * h_intersect
        area_union = target_area + pred_area - area_intersect
        ious = (area_intersect + 1.0) / (area_union + 1.0)
        gious = ious - (ac_uion - area_union) / ac_uion
        if self.loss_type == 'iou':
            losses = -torch.log(ious)
        elif self.loss_type == 'linear_iou':
            losses = 1 - ious
        elif self.loss_type == 'giou':
            losses = 1 - gious
        else:
            raise NotImplementedError

        if weight is not None and weight.sum() > 0:
            return (losses * weight).sum()
        else:
            assert losses.numel() != 0
            return losses.sum()


class LossEvaluator:

    def __init__(self):
        self.reg_loss_func = IOULoss("giou")
        self.centerness_loss_func = nn.BCEWithLogitsLoss(reduction="sum")
        self.cls_loss_func = sigmoid_focal_loss

    def _get_cls_loss(self, cls_preds, cls_targets, total_num_pos):
        cls_loss = self.cls_loss_func(cls_preds, cls_targets.int())
        return cls_loss.sum() / total_num_pos

    def _get_reg_loss(self, reg_preds, reg_targets, centerness_targets):
        sum_centerness_targets = centerness_targets.sum()
        reg_preds = reg_preds.reshape(-1, 4)
        reg_targets = reg_targets.reshape(-1, 4)
        reg_loss = self.reg_loss_func(reg_preds, reg_targets,
                                      centerness_targets)
        return reg_loss / sum_centerness_targets # Why dividing by that here ?

    def _get_centerness_loss(self, centerness_preds, centerness_targets,
                             total_num_pos):
        centerness_loss = self.centerness_loss_func(centerness_preds,
                                                    centerness_targets)
        return centerness_loss / total_num_pos

    def _evaluate_losses(self, reg_preds, cls_preds, centerness_preds,
                         reg_targets, cls_targets, centerness_targets,
                         pos_inds):
        total_num_pos = max(pos_inds.new_tensor([pos_inds.numel()]), 1.0)

        cls_loss = self._get_cls_loss(cls_preds, cls_targets, total_num_pos)

        if pos_inds.numel() > 0:
            reg_loss = self._get_reg_loss(reg_preds, reg_targets,
                                          centerness_targets)
            centerness_loss = self._get_centerness_loss(centerness_preds,
                                                        centerness_targets,
                                                        total_num_pos)
        else:
            reg_loss = reg_preds.sum() # 0 ??
            centerness_loss = centerness_preds.sum() # 0 ??

        return reg_loss, cls_loss, centerness_loss

    def __call__(self, locations, preds, targets_batch, num_classes):
        # reg_targets is a list of num_levels tensors of size Bxnum_loc_per_levelx4
        cls_targets, reg_targets = _prepare_labels(locations, targets_batch)

        cls_preds, reg_preds, centerness_preds = preds

        reg_preds, reg_targets, cls_preds, cls_targets, centerness_preds, pos_inds = _get_positive_samples(
            cls_targets, reg_targets, cls_preds, reg_preds, centerness_preds,
            num_classes)

        centerness_targets = _compute_centerness_targets(reg_targets)

        reg_loss, cls_loss, centerness_loss = self._evaluate_losses(
            reg_preds, cls_preds, centerness_preds, reg_targets, cls_targets,
            centerness_targets, pos_inds)

        return cls_loss, reg_loss, centerness_loss

class FCOS(pl.LightningModule):
    def __init__(
        self,
        network,
        num_classes,
        optimizer,
        scheduler,
        loss,
        batch_tf,
        metric_ignore_index,
        norm,
        one_hot,
        tta=None,
        sliding=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.le = LossEvaluator()
        self.post_processor = FCOSPostProcessor(
            pre_nms_thresh=pre_nms_thresh,
            pre_nms_top_n=pre_nms_top_n,
            nms_thresh=nms_thresh,
            fpn_post_nms_top_n=fpn_post_nms_top_n,
            min_size=0,
            num_classes=num_classes)
        self.fpn_strides = fpn_strides
        self.num_classes = num_classes
        self.network = network(in_channels=in_channels, num_classes=num_classes)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.map_metric = MeanAveragePrecision()
    
    def configure_optimizers(self):
        parameters = list(self.parameters())
        trainable_parameters = list(filter(lambda p: p.requires_grad, parameters))
        rank_zero_info(
            f"The model will start training with only {sum([int(torch.numel(p)) for p in trainable_parameters])} "
            f"trainable parameters out of {sum([int(torch.numel(p)) for p in parameters])}."
        )
        optimizer = self.optimizer(params=trainable_parameters)
        scheduler = self.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            },
        }
    
    def forward(self, images, targets_batch=None):
        features, box_cls, box_regression, centerness = self.network(images)
        # locations is a list of num_feat_level elem, where each elem indicates the tensor of locations in the original image corresponding to each location in the feature map at this level
        locations = _compute_locations(features, self.fpn_strides)
        outputs = {}
        if targets_batch != None:
            cls_loss, reg_loss, centerness_loss = self.le(
                locations, (box_cls, box_regression, centerness),
                targets_batch,
                num_classes=self.num_classes)
            outputs["cls_loss"] = cls_loss
            outputs["reg_loss"] = reg_loss
            outputs["centerness_loss"] = centerness_loss
            outputs["combined_loss"] = cls_loss + reg_loss + centerness_loss

        image_size = images.shape[-1]
        predicted_boxes, scores, all_classes = self.post_processor(
            locations, box_cls, box_regression, centerness, image_size)

        outputs["predicted_boxes"] = predicted_boxes
        outputs["scores"] = scores
        outputs["pred_classes"] = all_classes
        return outputs
    
    def training_step(self, batch, batch_idx):
        batch = batch["sup"]
        x = batch["image"]
        y = batch["label"]
        results = self.forward(x, y)
        loss = results["combined_loss"]
        self.log(f"loss/train", loss.detach().item())
        self.log(f"cls_loss/train", results["cls_loss"].detach().item())
        self.log(f"reg_loss/train", results["reg_loss"].detach().item())
        self.log(f"centerness_loss/train", results["centerness_loss"].detach().item())
        return loss
        
    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["label"]
        results = self.forward(x, y)
        loss = results["combined_loss"]
        self.map_metric.update(preds, y)
        self.log(f"loss/val", loss.detach().item())
        self.log(f"cls_loss/val", results["cls_loss"].detach().item())
        self.log(f"reg_loss/val", results["reg_loss"].detach().item())
        self.log(f"centerness_loss/val", results["centerness_loss"].detach().item())
        
    def on_validation_epoch_end(self):
        self.log("map/val", self.map_metric.compute())
        self.map_metric.reset()