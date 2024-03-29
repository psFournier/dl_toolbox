import torch.nn as nn
import torchvision
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_info
from dl_toolbox.losses import giou
INF = 100000000
import torch

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


def _prepare_labels(locations, targets_batch, reg_dists):
    device = targets_batch[0].device
    # L = sum locs per level x 2 : for each loc in all_locs, the max size of bb authorized
    all_locations = locations.to(device) # Lx2
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
            reg_targets, reg_dists).to(device) # LxT

        bbox_areas = torchvision.ops.box_area(bbox_targets) # compared to above, does not deal with 0dim bb
        
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
        
        all_cls_targets.append(cls_targets)
        all_reg_targets.append(reg_targets)
    
    return torch.stack(all_cls_targets), torch.stack(all_reg_targets)




class FCOS(pl.LightningModule):
    def __init__(
        self,
        network,
        num_classes,
        optimizer,
        scheduler,
        in_channels=3,
        pre_nms_thresh=0.3,
        pre_nms_top_n=1000,
        nms_thresh=0.45,
        fpn_post_nms_top_n=50,
        min_size=0,
        *args,
        **kwargs
    ):
        super().__init__()
        self.network = network(in_channels=in_channels, num_classes=num_classes)
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size        
        self.fpn_strides = [8, 16, 32, 64, 128]
        self.feat_sizes = self.network.feat_sizes
        self.max_dist_per_level = [-1, 64, 128, 256, 512, INF]
        self.num_classes = num_classes
        self.map_metric_train = MeanAveragePrecision()
        self.map_metric_val = MeanAveragePrecision()
        # locations is a list of num_feat_level elem, where each elem indicates the tensor of 
        # locations in the original image corresponding to each location in the feature map at this level
        self.locations, self.reg_dists = self._compute_locations()
    
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
    
    def _compute_locations(self):
        locations = []
        reg_dists = []
        
        def _locations_per_level(h, w, s):
            locs_x = [i for i in range(w)]
            locs_y = [i for i in range(h)]
            locs_x = [s / 2 + x * s for x in locs_x]
            locs_y = [s / 2 + y * s for y in locs_y]
            locs = [(x, y) for y in locs_y for x in locs_x]
            return torch.tensor(locs)
        
        for level, (h,w) in enumerate(self.feat_sizes):
            locs = _locations_per_level(h, w, self.fpn_strides[level])
            locations.append(locs)
            
            level_distances = torch.tensor([
                self.max_dist_per_level[level], self.max_dist_per_level[level+1]
            ], dtype=torch.float32)
            level_distances = level_distances.repeat(len(locs)).view(
                len(locs), 2)
            reg_dists.append(level_distances)
            
        all_locs = torch.cat(locations)
        all_reg_dists = torch.cat(reg_dists)
        return all_locs, all_reg_dists
    
    def _post_process(
        self,
        locations, # sum num loc all levels (=L) x 2
        cls_preds, # B x L x C 
        reg_preds, # B x L x 4
        cness_preds, # B x L x 1
        image_size):
        
        B, num_locs, C = cls_preds.shape
        cls_preds = cls_preds.sigmoid() # BxLxC in [0,1]
        cness_preds = cness_preds.sigmoid()
        
        candidate_inds = cls_preds > self.pre_nms_thresh # BxLxC
        cls_preds = cls_preds * cness_preds # BxLxC
        
        pre_nms_top_n = candidate_inds.reshape(B, -1).sum(1) # B
        pre_nms_top_n = pre_nms_top_n.clamp(max=self.pre_nms_top_n)
        
        #bboxes is a list of B tensors of size lx4 (filtered with pre_nms_threshold)
        bboxes = []
        cls_labels = []
        scores = []
        for i in range(B):
            # Tensor with true where score for loc l and class c > pre_nms_thresh
            per_candidate_inds = candidate_inds[i] # LxC
            # tenseur de taille lx2 (l!=L) avec les indices des elem de cls_preds où > nms_thresh
            per_candidate_nonzeros = per_candidate_inds.nonzero() 
            # dim l : positions dans [0,L] des elem dont cls_preds(c) > nms_thresh 
            per_box_loc = per_candidate_nonzeros[:, 0]
            # dim l : classe dans [1, C] des elem dont cls_preds(h,w) > nms_thresh
            per_class = per_candidate_nonzeros[:, 1] + 1

            per_reg_preds = reg_preds[i] # Lx4
            # liste des bb des elem dont cls_preds(c) > nms_thresh 
            per_reg_preds = per_reg_preds[per_box_loc] # lx4
            per_locations = locations[per_box_loc] # lx2

            per_pre_nms_top_n = pre_nms_top_n[i]
            
            per_cls_preds = cls_preds[i] # LxC
            # tenseur de taille L avec les elem de cls_preds*centerness tels que cls_preds > nms_thresh
            per_cls_preds = per_cls_preds[per_candidate_inds] 
            # si y a plus de per_pre_nms_topn qui passe nms_thresh (si l est trop longue)
            if per_candidate_inds.sum().item() > per_pre_nms_top_n.item():
                per_cls_preds, top_k_indices = per_cls_preds.topk(
                    per_pre_nms_top_n, sorted=False)
                per_class = per_class[top_k_indices]
                per_reg_preds = per_reg_preds[top_k_indices]
                per_locations = per_locations[top_k_indices]
            
            # Rewrites bbox (x0,y0,x1,y1) from reg targets (l,t,r,b) following eq (1) in paper
            per_bboxes = torch.stack([
                per_locations[:, 0] - per_reg_preds[:, 0],
                per_locations[:, 1] - per_reg_preds[:, 1],
                per_locations[:, 0] + per_reg_preds[:, 2],
                per_locations[:, 1] + per_reg_preds[:, 3],
            ], dim=1)
            per_bboxes = torchvision.ops.clip_boxes_to_image(per_bboxes, (image_size, image_size))
            per_bboxes = per_bboxes[torchvision.ops.remove_small_boxes(per_bboxes, self.min_size)]
            per_scores = torch.sqrt(per_cls_preds)
            
            picked_indices = torchvision.ops.nms(per_bboxes, per_scores, self.nms_thresh)
            picked_boxes = per_bboxes[picked_indices]
            confidence_scores = per_scores[picked_indices]
            picked_classes = per_class[picked_indices]
            
            number_of_detections = len(picked_indices)
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
            
            bboxes.append(picked_boxes)
            cls_labels.append(picked_classes)
            scores.append(confidence_scores)
        
        return bboxes, scores, cls_labels

    def _get_cls_loss(self, cls_preds, cls_targets, total_num_pos):
        nc = cls_preds.shape[-1]
        onehot = F.one_hot(cls_targets.long(), nc+1)[...,1:].float()
        cls_loss = torchvision.ops.sigmoid_focal_loss(cls_preds, onehot)
        return cls_loss.sum() / total_num_pos

    def _get_reg_loss(self, reg_preds, reg_targets, centerness_targets):
        reg_preds = reg_preds.reshape(-1, 4)
        reg_targets = reg_targets.reshape(-1, 4)
        reg_loss = giou(reg_preds, reg_targets, weight=centerness_targets)
        return reg_loss / centerness_targets.sum()

    def _get_centerness_loss(self, centerness_preds, centerness_targets,
                             total_num_pos):
        centerness_loss = F.binary_cross_entropy_with_logits(
            centerness_preds.squeeze(), centerness_targets, reduction='sum')
        return centerness_loss / total_num_pos
    
    def forward(self, images, targets_batch=None):
        features, cls_preds, reg_preds, cness_preds = self.network(images)
        locations = self.locations.to(features.device)
        predicted_boxes, scores, all_classes = self._post_process(
            locations, cls_preds, reg_preds, cness_preds, images.shape[-1])
        
        outputs = {}
        if targets_batch != None:
            reg_dists = self.reg_dists.to(features.device) # remove this by reorg code
            cls_targets, reg_targets = _prepare_labels(locations, targets_batch, reg_dists)
            pos_inds_b, pos_inds_loc = torch.nonzero(cls_targets > 0, as_tuple=True)
            reg_preds = reg_preds[pos_inds_b, pos_inds_loc, :]
            reg_targets = reg_targets[pos_inds_b, pos_inds_loc, :]
            cness_preds = cness_preds[pos_inds_b, pos_inds_loc, :]
            cness_targets = _compute_centerness_targets(reg_targets)
            total_num_pos = max(pos_inds_b.new_tensor([pos_inds_b.numel()]), 1.0)
            cls_loss = self._get_cls_loss(cls_preds, cls_targets, total_num_pos)
            if pos_inds_b.numel() > 0:
                reg_loss = self._get_reg_loss(reg_preds, reg_targets,
                                              cness_targets)
                centerness_loss = self._get_centerness_loss(cness_preds,
                                                            cness_targets,
                                                            total_num_pos)
            else:
                reg_loss = reg_preds.sum() # 0 ??
                centerness_loss = cness_preds.sum() # 0 ??
            
            outputs["cls_loss"] = cls_loss
            outputs["reg_loss"] = reg_loss
            outputs["centerness_loss"] = centerness_loss
            outputs["combined_loss"] = cls_loss + reg_loss + centerness_loss

        outputs["predicted_boxes"] = predicted_boxes
        outputs["scores"] = scores
        outputs["pred_classes"] = all_classes
        return outputs
    
    def training_step(self, batch, batch_idx):
        x, bboxes, labels, image_paths = batch
        y = [torch.cat([bb, l], dim=1) for bb, l in zip(bboxes, labels)]
        results = self.forward(x, y)
        loss = results["combined_loss"]
        self.log(f"loss/train", loss.detach().item())
        self.log(f"cls_loss/train", results["cls_loss"].detach().item())
        self.log(f"reg_loss/train", results["reg_loss"].detach().item())
        self.log(f"centerness_loss/train", results["centerness_loss"].detach().item())
        preds = [{'boxes': bb, 'scores': s, 'labels': l} for bb,s,l in zip(
            results["predicted_boxes"], results["scores"], results["pred_classes"]
        )]
        targets = [{'boxes': bb, 'labels': l.squeeze(dim=1)} for bb,l in zip(bboxes, labels)]
        self.map_metric_train.update(preds, targets)
        return loss
    
    def on_train_epoch_end(self):
        map_train = self.map_metric_train.compute()
        for k,v in map_train.items():
            self.log(f"{k}/train", v)
        print("\nMAP train: ", map_train['map'])
        self.map_metric_train.reset()        
        
    def validation_step(self, batch, batch_idx):
        x, bboxes, labels, image_paths = batch
        y = [torch.cat([bb, l], dim=1) for bb, l in zip(bboxes, labels)]
        results = self.forward(x, y)
        loss = results["combined_loss"]
        preds = [{'boxes': bb, 'scores': s, 'labels': l} for bb,s,l in zip(
            results["predicted_boxes"], results["scores"], results["pred_classes"]
        )]
        targets = [{'boxes': bb, 'labels': l.squeeze(dim=1)} for bb,l in zip(bboxes, labels)]
        self.map_metric_val.update(preds, targets)
        self.log(f"loss/val", loss.detach().item())
        self.log(f"cls_loss/val", results["cls_loss"].detach().item())
        self.log(f"reg_loss/val", results["reg_loss"].detach().item())
        self.log(f"centerness_loss/val", results["centerness_loss"].detach().item())
        if batch_idx==0:
            self.trainer.logger.experiment.add_image_with_boxes(
                f"preds/val",
                x[0].detach().cpu(),
                preds[0]['boxes'].detach().cpu(),#,
                global_step=self.trainer.global_step,
                dataformats='CHW', 
                labels=[f"{l}: {s}" for l,s in zip(preds[0]['labels'], preds[0]['scores'])]
            )
        
    def on_validation_epoch_end(self):
        map_val = self.map_metric_val.compute()
        for k,v in map_val.items():
            self.log(f"{k}/val", v)
        print("\nMAP val: ", map_val['map'])
        self.map_metric_val.reset()