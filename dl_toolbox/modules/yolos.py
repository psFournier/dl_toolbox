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
            eos_coef: relative classification weight applied to the no-object category
        """
        super().__init__()
        self.num_classes = num_classes
        self.eos_coef = eos_coef
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, matches, num_boxes, log=True):
        """
        Params:
            matches: list of batch_size pairs (I, J) of arrays such that output bbox I[n] must be matched with target bbox J[n]
        """
                
        # all N tgt labels are reordered following the matches and concatenated  
        reordered_labels = [t["labels"][J] for t, (_, J) in zip(targets, matches)]
        reordered_labels = torch.cat(reordered_labels) # Nx1
        
        # batch_idxs[i] is the idx in the batch of the img to which the i-th elem in the new order corresponds
        batch_idxs = [torch.full_like(pred, i) for i, (pred, _) in enumerate(matches)]
        batch_idxs = torch.cat(batch_idxs) # Nx1
        
        # src_idxs[i] is the idx of the preds for img batch_idxs[i] to which the i-th elem in the new order corresponds
        pred_idxs = torch.cat([pred for (pred, _) in matches]) # Nx1
        
        # target_classes is of shape batch_size x num det tokens, and is num_classes everywhere, except for each token that is matched to a tgt bb, where it is the label of the matched tgt
        pred_logits = outputs['pred_logits'] #BxNdetTokxNcls
        target_classes = torch.full(
            pred_logits.shape[:2], #Shape BxNdetTok
            self.num_classes, #Filled with num_cls
            dtype=torch.int64, 
            device=pred_logits.device
        )
        target_classes[(batch_idxs, pred_idxs)] = reordered_labels
        
        loss_ce = F.cross_entropy(pred_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, matches, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        reordered_target_boxes = [t['boxes'][i] for t, (_, i) in zip(targets, matches)]
        reordered_target_boxes = torch.cat(reordered_target_boxes) # Nx4
        
        # batch_idxs[i] is the idx in the batch of the img to which the i-th elem in the new order corresponds
        batch_idxs = [torch.full_like(pred, i) for i, (pred, _) in enumerate(matches)]
        batch_idxs = torch.cat(batch_idxs) # Nx1
        
        # src_idxs[i] is the idx of the preds for img batch_idxs[i] to which the i-th elem in the new order corresponds
        pred_idxs = torch.cat([pred for (pred, _) in matches]) # Nx1
        
        pred_boxes = outputs['pred_boxes'] # BxNdetTokx4
        reordered_pred_boxes = pred_boxes[(batch_idxs, pred_idxs)] # Nx4

        losses = {}
        loss_bbox = F.l1_loss(
            reordered_pred_boxes,
            reordered_target_boxes,
            reduction='none'
        )
        losses['loss_bbox'] = loss_bbox.sum() / num_boxes
        loss_giou = 1 - torch.diag(generalized_box_iou(
            box_convert(reordered_pred_boxes, 'xywh', 'xyxy'),
            box_convert(reordered_target_boxes, 'xywh', 'xyxy')))
        losses['loss_giou'] = loss_giou.sum() / num_boxes
        return losses

    def forward(self, outputs, targets, matches):
        """ This performs the loss computation.
        Parameters:
            outputs: dict of tensors, see the output specification of the model for the format
            targets: list of dicts, such that len(targets) == batch_size. The expected keys in each dict depends on the losses applied, see each loss' doc
            matches: list of batch_size pairs (I, J) of arrays such that for pair (I,J) output bbox I[n] must be matched with target bbox J[n]
        """
        # Compute the average (?) number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        device = next(iter(outputs.values())).device
        num_boxes = torch.as_tensor(
            [num_boxes],
            dtype=torch.float,
            device=device
        )
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
        pred_thresh,
        tta=None,
        sliding=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.backbone = timm.create_model(
            backbone,
            pretrained=True,
            dynamic_img_size=True #Deals with inputs of other size than pretraining
        )
        self.embed_dim = self.backbone.embed_dim 
        self.det_token_num = det_token_num
        self.add_det_tokens_to_backbone()
        self.class_embed = torchvision.ops.MLP(
            self.embed_dim,
            [self.embed_dim, self.embed_dim, num_classes+1]
        ) #Num_classes + 1 to deal with no_obj category
        self.bbox_embed = torchvision.ops.MLP(
            self.embed_dim,
            [self.embed_dim, self.embed_dim, 4]
        )
        self.loss = SetCriterion(
            num_classes,
            0.5 #Coeff of no-obj category
        )
        self.num_classes = num_classes
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.map_metric = MeanAveragePrecision(
            box_format='xywh',
            backend='faster_coco_eval'
        )
        self.sliding = sliding
        self.pred_thresh = pred_thresh
        
    def add_det_tokens_to_backbone(self):
        det_token = nn.Parameter(
            torch.zeros(
                1,
                self.det_token_num,
                self.backbone.embed_dim
            )
        )
        self.det_token = torch.nn.init.trunc_normal_(
            det_token,
            std=.02
        )
        det_pos_embed = nn.Parameter(
            torch.zeros(
                1,
                self.det_token_num,
                self.backbone.embed_dim
            )
        )
        self.det_pos_embed = torch.nn.init.trunc_normal_(
            det_pos_embed,
            std=.02
        )
        #The ViT needs to know how many input tokens are not for patch embeddings
        self.backbone.num_prefix_tokens += self.det_token_num
    
    def configure_optimizers(self):
        train_params = list(filter(lambda p: p[1].requires_grad, self.named_parameters()))
        nb_train = sum([int(torch.numel(p[1])) for p in train_params])
        nb_tot = sum([int(torch.numel(p)) for p in self.parameters()])
        #print(f"Training {nb_train} params out of {nb_tot}: {[p[0] for p in train_params]}")
        print(f"Training {nb_train} params out of {nb_tot}")
        
        #if hasattr(self, 'no_weight_decay'):
        #    wd_val = 0. #self.optimizer.weight_decay
        #    nwd_params = self.no_weight_decay()
        #    train_params = param_groups_weight_decay(train_params, wd_val, nwd_params)
        #    print(f"{len(train_params[0]['params'])} are not affected by weight decay.")
        
        optimizer = self.optimizer(params=[p[1] for p in train_params])
        scheduler = self.scheduler(optimizer)
        return [optimizer], [scheduler]
    
    def raw_logits_and_bboxs(self, x):
        x = self.backbone.patch_embed(x)
        # If cls token in backbone
        cls_pos_embed = self.backbone.pos_embed[:, 0, :][:,None] # size 1x1xembed_dim
        patch_pos_embed = self.backbone.pos_embed[:, 1:, :] # 1xnum_patchxembed_dim
        pos_embed = torch.cat((cls_pos_embed, self.det_pos_embed, patch_pos_embed), dim=1)
        if self.backbone.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                pos_embed,
                (H, W),
                num_prefix_tokens=self.backbone.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
        # If cls token in backbone
        cls_token = self.backbone.cls_token.expand(x.shape[0], -1, -1) 
        det_token = self.det_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, det_token, x], dim=1)
        x += pos_embed
        x = self.backbone.pos_drop(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        x = self.backbone.blocks(x)
        x = self.backbone.norm(x)
        x = x[:,1:1+self.det_token_num,...]
        outputs_class = self.class_embed(x)
        outputs_coord = self.bbox_embed(x).sigmoid()
        out = {'pred_logits': outputs_class, 'pred_boxes': outputs_coord}
        return out
    
    
    def forward(self, x, sliding=None):
        if sliding is not None:
            auxs = [self.forward(aux) for aux in sliding(x)]
            return sliding.merge(auxs)
        else:
            return self.raw_logits_and_bboxs(x)

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
        outputs = self.raw_logits_and_bboxs(x)
        # are we sure we need to norm targets ?
        norm_tgts = self.norm_targets(targets)
        matches = self.hungarian_matching(outputs, norm_tgts)
        loss = self.loss(outputs, norm_tgts, matches)
        self.log(f"loss/train", loss.detach().item())
        return loss
        
    def validation_step(self, batch, batch_idx):
        x, targets, paths = batch
        outputs = self.forward(x, sliding=self.sliding)
        norm_tgts = self.norm_targets(targets)
        matches = self.hungarian_matching(outputs, norm_tgts)
        loss = self.loss(outputs, norm_tgts, matches)
        self.log(f"Total loss/val", loss.detach().item())
        preds = self.post_process(outputs, x)
        # map_metric does not work yet with tv_tensors
        targets = [{'labels':t['labels'], 'boxes':t['boxes'].as_subclass(torch.Tensor)} for t in targets]
        self.map_metric.update(preds, targets)
        
    def on_validation_epoch_end(self):
        mapmetric = self.map_metric.compute()['map']
        self.log("map/val", mapmetric)
        print("\nMAP: ", mapmetric)
        self.map_metric.reset()
        
    def predict_step(self, batch, batch_idx):
        x, targets, paths = batch
        outputs = self.forward(x, sliding=self.sliding)
        preds = self.post_process(outputs, x)
        return preds