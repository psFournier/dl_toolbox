import timm
import torch
import torchvision
from timm.layers import resample_abs_pos_embed     
import torch.nn as nn
from torchvision.ops import box_convert, generalized_box_iou
from scipy.optimize import linear_sum_assignment
import torch.nn.functional as F
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import tv_tensors
import torchvision.transforms.v2.functional as v2F
from torchvision.tv_tensors import BoundingBoxFormat


class SetCriterion(nn.Module):
    def __init__(self, num_classes, eos_coef):
        """ Create the criterion.
        Parameters:
            num_classes: number of object categories, omitting the special no-object category
            matcher: module able to compute a matching between targets and proposals
            weight_dict: dict containing as key the names of the losses and as values their relative weight.
            eos_coef: relative classification weight applied to the no-object category
            losses: list of all the losses to be applied. See get_loss for list of available losses.
        """
        super().__init__()
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        src_logits = outputs['pred_logits']
        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o
        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        idx = self._get_src_permutation_idx(indices)
        src_boxes = outputs['pred_boxes'][idx]
        target_boxes = torch.cat([t['boxes'][i] for t, (_, i) in zip(targets, indices)], dim=0)
        loss_bbox = F.l1_loss(src_boxes, target_boxes, reduction='none')
        losses = {}
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_convert(src_boxes, 'xywh', 'xyxy'),
            box_convert(target_boxes, 'xywh', 'xyxy')))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def forward(self, outputs, targets, matches):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        dev = next(iter(outputs.values())).device
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=dev)
        # Compute all the requested losses
        losses = {}
        losses.update(self.loss_labels(outputs, targets, matches, num_boxes))
        losses.update(self.loss_boxes(outputs, targets, matches, num_boxes))
        loss = sum(losses.values())
        return loss    

class Yolos(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        det_token_num,
        backbone,
        optimizer,
        scheduler,
        tta=None,
        sliding=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.backbone = timm.create_model(backbone, pretrained=True, dynamic_img_size=True)
        hdim = self.backbone.embed_dim 
        self.det_token_num = det_token_num
        self.add_det_tokens_to_backbone()
        self.class_embed = torchvision.ops.MLP(hdim, [hdim,hdim,num_classes+1])
        self.bbox_embed = torchvision.ops.MLP(hdim, [hdim,hdim,4])
        self.loss = SetCriterion(num_classes, 0.5)
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.map_metric = MeanAveragePrecision(box_format='xywh')
        
    def add_det_tokens_to_backbone(self):
        det_token = nn.Parameter(torch.zeros(1, self.det_token_num, self.backbone.embed_dim))
        self.det_token = torch.nn.init.trunc_normal_(det_token, std=.02)
        det_pos_embed = torch.zeros(1, self.det_token_num, self.backbone.embed_dim)
        det_pos_embed = torch.nn.init.trunc_normal_(det_pos_embed, std=.02)
        cls_pos_embed = self.backbone.pos_embed[:, 0, :][:,None] # size 1x1xembed_dim
        patch_pos_embed = self.backbone.pos_embed[:, 1:, :] # 1xnum_patchxembed_dim
        all_pos_embed = torch.cat((cls_pos_embed, det_pos_embed, patch_pos_embed), dim=1)
        self.pos_embed = torch.nn.Parameter(all_pos_embed)
        self.backbone.num_prefix_tokens += self.det_token_num
    
    def configure_optimizers(self):
        train_params = list(filter(lambda p: p.requires_grad, self.parameters()))
        nb_train = sum([int(torch.numel(p)) for p in train_params])
        nb_tot = sum([int(torch.numel(p)) for p in self.parameters()])
        print(f"Training {nb_train} params out of {nb_tot}.")
        
        #if hasattr(self, 'no_weight_decay'):
        #    wd_val = 0. #self.optimizer.weight_decay
        #    nwd_params = self.no_weight_decay()
        #    train_params = param_groups_weight_decay(train_params, wd_val, nwd_params)
        #    print(f"{len(train_params[0]['params'])} are not affected by weight decay.")
        
        optimizer = self.optimizer(params=train_params)
        scheduler = self.scheduler(optimizer)
        return [optimizer], [scheduler]
    
    def _pos_embed_with_det(self, x):
        if self.backbone.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=self.backbone.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        else:
            pos_embed = self.pos_embed

        to_cat = []
        if self.backbone.cls_token is not None:
            to_cat.append(self.backbone.cls_token.expand(x.shape[0], -1, -1))
        if self.backbone.reg_token is not None:
            to_cat.append(self.backbone.reg_token.expand(x.shape[0], -1, -1))
        to_cat.append(self.det_token.expand(x.shape[0], -1, -1)) # HERE det tokens
        
        # check that no embed class is indeed false for ViT ?
        if to_cat:
            x = torch.cat(to_cat + [x], dim=1)
        x = x + pos_embed

        return self.backbone.pos_drop(x)
    
    def backbone_forward_features(self, x):
        x = self.backbone.patch_embed(x)
        x = self._pos_embed_with_det(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        return x
    
    def forward(self, x):
        x = self.backbone_forward_features(x)
        x = x[:,1:1+self.det_token_num,...]
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x).sigmoid()
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return out
    
    @torch.no_grad()
    def hungarian_matching(self, outputs, targets):
        """ 
        Params:
            outputs=dict:
                 "pred_logits": Tensor of dim [B, num_queries, num_classes] with the class logits
                 "pred_boxes": Tensor of dim [B, num_queries, 4] with the pred box coord
            targets=list (len(targets) = batch_size) of dicts, each dict:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coord

        Returns:
            A list of size batch_size, containing tuples of (index_i, index_j) where:
                - index_i is the indices of the selected predictions (in order)
                - index_j is the indices of the corresponding selected targets (in order)
            For each batch element, it holds:
                len(index_i) = len(index_j) = min(num_queries, num_target_boxes)
        """
        bs, num_queries = outputs["pred_logits"].shape[:2]

        # We flatten to compute the cost matrices in a batch
        out_prob = outputs["pred_logits"].flatten(0, 1).softmax(-1)  # [batch_size * num_queries, num_classes]
        out_bbox = outputs["pred_boxes"].flatten(0, 1)  # [batch_size * num_queries, 4]

        # Also concat the target labels and boxes
        tgt_ids = torch.cat([v["labels"] for v in targets])
        tgt_bbox = torch.cat([v["boxes"] for v in targets])
            
        # Compute the classification cost. Contrary to the loss, we don't use the NLL,
        # but approximate it in 1 - proba[target class].
        # The 1 is a constant that doesn't change the matching, it can be ommitted.
        cost_class = -out_prob[:, tgt_ids]

        # Compute the L1 cost between boxes
        cost_bbox = torch.cdist(out_bbox, tgt_bbox, p=1)
        
        # Compute the giou cost betwen boxes
        out_bbox = box_convert(out_bbox, 'xywh', 'xyxy')
        tgt_bbox = box_convert(tgt_bbox, 'xywh', 'xyxy')
        cost_giou = -generalized_box_iou(out_bbox, tgt_bbox)

        # Final cost matrix
        C = cost_bbox + cost_class + cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
    @torch.no_grad()
    def post_process(self, outputs, images):
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']
        b,c,h,w = images.shape
        target_sizes = torch.Tensor((h,w)).repeat((b,1))
        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2
        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = out_bbox * scale_fct[:, None, :].to(out_bbox.device)
        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]
        return results
    
    def norm_bb(self, inpt_bb):
        tensor_bb = inpt_bb.as_subclass(torch.Tensor)
        bb = tensor_bb.clone() if tensor_bb.is_floating_point() else tensor_bb.float()
        xyxy_bb = v2F.convert_bounding_box_format(
            bb, old_format=inpt_bb.format, new_format=tv_tensors.BoundingBoxFormat.XYXY, inplace=True
        )
        xyxy_bb[..., 0::2].div_(inpt_bb.canvas_size[1])
        xyxy_bb[..., 1::2].div_(inpt_bb.canvas_size[0])
        out_bb = v2F.convert_bounding_box_format(
            xyxy_bb, old_format=BoundingBoxFormat.XYXY, new_format=inpt_bb.format, inplace=True
        )
        return tv_tensors.wrap(out_bb, like=inpt_bb)
    
    def norm_targets(self, targets):
        return [{'labels':t['labels'], 'boxes':self.norm_bb(t['boxes'])} for t in targets]
    
    def training_step(self, batch, batch_idx):
        x, targets, paths = batch["sup"]
        outputs = self.forward(x)
        # are we sure we need to norm targets ?
        norm_tgts = self.norm_targets(targets)
        matches = self.hungarian_matching(outputs, norm_tgts)
        loss = self.loss(outputs, norm_tgts, matches)
        self.log(f"loss/train", loss.detach().item())
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, targets, paths = batch
        outputs = self.forward(x)
        norm_tgts = self.norm_targets(targets)
        matches = self.hungarian_matching(outputs, norm_tgts)
        loss = self.loss(outputs, norm_tgts, matches)
        self.log(f"loss/val", loss.detach().item())
        preds = self.post_process(outputs, x)
        # map_metric does not work yet with tv_tensors
        targets = [{'labels':t['labels'], 'boxes':t['boxes'].as_subclass(torch.Tensor)} for t in targets]
        self.map_metric.update(preds, targets)
        
    def on_validation_epoch_end(self):
        mapmetric = self.map_metric.compute()['map']
        self.log("map/val", mapmetric)
        print("\nMAP: ", mapmetric)
        self.map_metric.reset()