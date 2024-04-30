import timm
import torch
import torchvision
from timm.layers import resample_abs_pos_embed     
import torch.nn as nn

class Detector(nn.Module):
    def __init__(self, num_classes, det_token_num=100):
        super().__init__()
        self.backbone = timm.create_model('vit_small_patch14_dinov2', pretrained=True, dynamic_img_size=True)
        hidden_dim = 384 
        self.det_token_num = det_token_num
        self.add_det_tokens()
        self.class_embed = torchvision.ops.MLP(384, [384,384,num_classes+1])
        self.bbox_embed = torchvision.ops.MLP(384, [384,384,4])
        
    def add_det_tokens(self):
        
        det_token = nn.Parameter(torch.zeros(1, self.det_token_num, self.backbone.embed_dim))
        self.det_token = torch.nn.init.trunc_normal_(det_token, std=.02)
        
        det_pos_embed = torch.zeros(1, self.det_token_num, self.backbone.embed_dim)
        det_pos_embed = torch.nn.init.trunc_normal_(det_pos_embed, std=.02)
        cls_pos_embed = self.backbone.pos_embed[:, 0, :][:,None] # size 1x1xembed_dim
        patch_pos_embed = self.backbone.pos_embed[:, 1:, :] # 1xnum_patchxembed_dim
        self.pos_embed = torch.nn.Parameter(torch.cat((cls_pos_embed, det_pos_embed, patch_pos_embed), dim=1))
        
        self.backbone.num_prefix_tokens += self.det_token_num
        
    def _pos_embed_with_det(self, x: torch.Tensor) -> torch.Tensor:
        if self.backbone.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                self.pos_embed,
                (H, W),
                num_prefix_tokens=0 if self.backbone.no_embed_class else self.backbone.num_prefix_tokens,
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

        if self.backbone.no_embed_class:
            # deit-3, updated JAX (big vision)
            # position embedding does not overlap with class token, add then concat
            x = x + pos_embed
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
        else:
            # original timm, JAX, and deit vit impl
            # pos_embed has entry for class token, concat then add
            if to_cat:
                x = torch.cat(to_cat + [x], dim=1)
            x = x + pos_embed

        return self.backbone.pos_drop(x)
        
    def backbone_forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = self.backbone.patch_embed(x)
        x = self._pos_embed_with_det(x)
        x = self.backbone.patch_drop(x)
        x = self.backbone.norm_pre(x)
        if self.backbone.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.backbone.blocks, x)
        else:
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
    
import torch
import torch.nn as nn
from torchvision.ops import box_convert, generalized_box_iou
from scipy.optimize import linear_sum_assignment

class HungarianMatcher(nn.Module):
    """This class computes an assignment between the targets and the predictions of the network

    For efficiency reasons, the targets don't include the no_object. Because of this, in general,
    there are more predictions than targets. In this case, we do a 1-to-1 matching of the best predictions,
    while the others are un-matched (and thus treated as non-objects).
    """

    def __init__(self, cost_class: float = 1, cost_bbox: float = 1, cost_giou: float = 1):
        """Creates the matcher

        Params:
            cost_class: This is the relative weight of the classification error in the matching cost
            cost_bbox: This is the relative weight of the L1 error of the bounding box coordinates in the matching cost
            cost_giou: This is the relative weight of the giou loss of the bounding box in the matching cost
        """
        super().__init__()
        self.cost_class = cost_class
        self.cost_bbox = cost_bbox
        self.cost_giou = cost_giou
        assert cost_class != 0 or cost_bbox != 0 or cost_giou != 0, "all costs cant be 0"

    @torch.no_grad()
    def forward(self, outputs, targets):
        """ Performs the matching

        Params:
            outputs: This is a dict that contains at least these entries:
                 "pred_logits": Tensor of dim [batch_size, num_queries, num_classes] with the classification logits
                 "pred_boxes": Tensor of dim [batch_size, num_queries, 4] with the predicted box coordinates

            targets: This is a list of targets (len(targets) = batch_size), where each target is a dict containing:
                 "labels": Tensor of dim [num_target_boxes] (where num_target_boxes is the number of ground-truth
                           objects in the target) containing the class labels
                 "boxes": Tensor of dim [num_target_boxes, 4] containing the target box coordinates

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
        C = self.cost_bbox * cost_bbox + self.cost_class * cost_class + self.cost_giou * cost_giou
        C = C.view(bs, num_queries, -1).cpu()

        sizes = [len(v["boxes"]) for v in targets]

        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        return [(torch.as_tensor(i, dtype=torch.int64), torch.as_tensor(j, dtype=torch.int64)) for i, j in indices]
    
import torch.nn.functional as F

class SetCriterion(nn.Module):
    """ This class computes the loss for DETR.
    The process happens in two steps:
        1) we compute hungarian assignment between ground truth boxes and the outputs of the model
        2) we supervise each pair of matched ground-truth / prediction (supervise class and box)
    """
    def __init__(self, num_classes, matcher, weight_dict, eos_coef, losses):
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
        self.matcher = matcher
        self.weight_dict = weight_dict
        self.eos_coef = eos_coef
        self.losses = losses
        empty_weight = torch.ones(self.num_classes + 1)
        empty_weight[-1] = self.eos_coef
        self.register_buffer('empty_weight', empty_weight)

    def loss_labels(self, outputs, targets, indices, num_boxes, log=True):
        """Classification loss (NLL)
        targets dicts must contain the key "labels" containing a tensor of dim [nb_target_boxes]
        """
        assert 'pred_logits' in outputs
        src_logits = outputs['pred_logits']

        idx = self._get_src_permutation_idx(indices)
        target_classes_o = torch.cat([t["labels"][J] for t, (_, J) in zip(targets, indices)])
        target_classes = torch.full(src_logits.shape[:2], self.num_classes,
                                    dtype=torch.int64, device=src_logits.device)
        target_classes[idx] = target_classes_o

        loss_ce = F.cross_entropy(src_logits.transpose(1, 2), target_classes, self.empty_weight)
        losses = {'loss_ce': loss_ce}

        #if log:
        #    # TODO this should probably be a separate loss, not hacked in this one here
        #    losses['class_error'] = 100 - accuracy(src_logits[idx], target_classes_o)[0]
        return losses

    @torch.no_grad()
    def loss_cardinality(self, outputs, targets, indices, num_boxes):
        """ Compute the cardinality error, ie the absolute error in the number of predicted non-empty boxes
        This is not really a loss, it is intended for logging purposes only. It doesn't propagate gradients
        """
        pred_logits = outputs['pred_logits']
        device = pred_logits.device
        tgt_lengths = torch.as_tensor([len(v["labels"]) for v in targets], device=device)
        # Count the number of predictions that are NOT "no-object" (which is the last class)
        card_pred = (pred_logits.argmax(-1) != pred_logits.shape[-1] - 1).sum(1)
        card_err = F.l1_loss(card_pred.float(), tgt_lengths.float())
        losses = {'cardinality_error': card_err}
        return losses

    def loss_boxes(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
           The target boxes are expected in format (center_x, center_y, w, h), normalized by the image size.
        """
        assert 'pred_boxes' in outputs
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

    def loss_masks(self, outputs, targets, indices, num_boxes):
        """Compute the losses related to the masks: the focal loss and the dice loss.
           targets dicts must contain the key "masks" containing a tensor of dim [nb_target_boxes, h, w]
        """
        assert "pred_masks" in outputs

        src_idx = self._get_src_permutation_idx(indices)
        tgt_idx = self._get_tgt_permutation_idx(indices)
        src_masks = outputs["pred_masks"]
        src_masks = src_masks[src_idx]
        masks = [t["masks"] for t in targets]
        # TODO use valid to mask invalid areas due to padding in loss
        target_masks, valid = nested_tensor_from_tensor_list(masks).decompose()
        target_masks = target_masks.to(src_masks)
        target_masks = target_masks[tgt_idx]

        # upsample predictions to the target size
        src_masks = interpolate(src_masks[:, None], size=target_masks.shape[-2:],
                                mode="bilinear", align_corners=False)
        src_masks = src_masks[:, 0].flatten(1)

        target_masks = target_masks.flatten(1)
        target_masks = target_masks.view(src_masks.shape)
        losses = {
            "loss_mask": sigmoid_focal_loss(src_masks, target_masks, num_boxes),
            "loss_dice": dice_loss(src_masks, target_masks, num_boxes),
        }
        return losses

    def _get_src_permutation_idx(self, indices):
        # permute predictions following indices
        batch_idx = torch.cat([torch.full_like(src, i) for i, (src, _) in enumerate(indices)])
        src_idx = torch.cat([src for (src, _) in indices])
        return batch_idx, src_idx

    def _get_tgt_permutation_idx(self, indices):
        # permute targets following indices
        batch_idx = torch.cat([torch.full_like(tgt, i) for i, (_, tgt) in enumerate(indices)])
        tgt_idx = torch.cat([tgt for (_, tgt) in indices])
        return batch_idx, tgt_idx

    def get_loss(self, loss, outputs, targets, indices, num_boxes, **kwargs):
        loss_map = {
            'labels': self.loss_labels,
            'cardinality': self.loss_cardinality,
            'boxes': self.loss_boxes,
            'masks': self.loss_masks
        }
        assert loss in loss_map, f'do you really want to compute {loss} loss?'
        return loss_map[loss](outputs, targets, indices, num_boxes, **kwargs)

    def forward(self, outputs, targets):
        """ This performs the loss computation.
        Parameters:
             outputs: dict of tensors, see the output specification of the model for the format
             targets: list of dicts, such that len(targets) == batch_size.
                      The expected keys in each dict depends on the losses applied, see each loss' doc
        """
        outputs_without_aux = {k: v for k, v in outputs.items() if k != 'aux_outputs'}

        # Retrieve the matching between the outputs of the last layer and the targets
        indices = self.matcher(outputs_without_aux, targets)

        # Compute the average number of target boxes accross all nodes, for normalization purposes
        num_boxes = sum(len(t["labels"]) for t in targets)
        num_boxes = torch.as_tensor([num_boxes], dtype=torch.float, device=next(iter(outputs.values())).device)
        #if is_dist_avail_and_initialized():
        #    torch.distributed.all_reduce(num_boxes)
        #num_boxes = torch.clamp(num_boxes / get_world_size(), min=1).item()

        # Compute all the requested losses
        losses = {}
        for loss in self.losses:
            losses.update(self.get_loss(loss, outputs, targets, indices, num_boxes))

        # In case of auxiliary losses, we repeat this process with the output of each intermediate layer.
        if 'aux_outputs' in outputs:
            for i, aux_outputs in enumerate(outputs['aux_outputs']):
                indices = self.matcher(aux_outputs, targets)
                for loss in self.losses:
                    if loss == 'masks':
                        # Intermediate masks losses are too costly to compute, we ignore them.
                        continue
                    kwargs = {}
                    if loss == 'labels':
                        # Logging is enabled only for the last layer
                        kwargs = {'log': False}
                    l_dict = self.get_loss(loss, aux_outputs, targets , indices, num_boxes, **kwargs)
                    l_dict = {k + f'_{i}': v for k, v in l_dict.items()}
                    losses.update(l_dict)

        return losses
    
class PostProcess(nn.Module):
    """ This module converts the model's output into the format expected by the coco api"""
    @torch.no_grad()
    def forward(self, outputs, target_sizes):
        """ Perform the computation
        Parameters:
            outputs: raw outputs of the model
            target_sizes: tensor of dimension [batch_size x 2] containing the size of each images of the batch
                          For evaluation, this must be the original image size (before any data augmentation)
                          For visualization, this should be the image size after data augment, but before padding
        """
        out_logits, out_bbox = outputs['pred_logits'], outputs['pred_boxes']

        assert len(out_logits) == len(target_sizes)
        assert target_sizes.shape[1] == 2

        prob = F.softmax(out_logits, -1)
        scores, labels = prob[..., :-1].max(-1)

        #boxes = box_convert(out_bbox, 'xywh', 'xyxy')        
        img_h, img_w = target_sizes.unbind(1)
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)
        boxes = out_bbox * scale_fct[:, None, :]
        #boxes = box_convert(boxes, 'xyxy', 'xywh') 

        results = [{'scores': s, 'labels': l, 'boxes': b} for s, l, b in zip(scores, labels, boxes)]

        return results
    
import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
from torchvision import tv_tensors
import torchvision.transforms.v2 as v2
from torchvision.tv_tensors import BoundingBoxFormat
        
def _norm_bounding_boxes(bounding_boxes, format, canvas_size):
    bounding_boxes = bounding_boxes.clone() if bounding_boxes.is_floating_point() else bounding_boxes.float()
    xyxy_boxes = v2.functional.convert_bounding_box_format(
        bounding_boxes, old_format=format, new_format=tv_tensors.BoundingBoxFormat.XYXY, inplace=True
    )
    xyxy_boxes[..., 0::2].div_(canvas_size[1])
    xyxy_boxes[..., 1::2].div_(canvas_size[0])
    out_boxes = v2.functional.convert_bounding_box_format(
        xyxy_boxes, old_format=BoundingBoxFormat.XYXY, new_format=format, inplace=True
    )
    return out_boxes

def norm_bounding_boxes(inpt):
    output = _norm_bounding_boxes(inpt.as_subclass(torch.Tensor), format=inpt.format, canvas_size=inpt.canvas_size)
    return tv_tensors.wrap(output, like=inpt)

class Yolos(pl.LightningModule):
    def __init__(
        self,
        network,
        criterion,
        weight_dict,
        post_process,
        num_classes,
        *args,
        **kwargs
    ):
        super().__init__()
        self.criterion = criterion
        self.weight_dict = weight_dict
        self.post_process = post_process
        self.num_classes = num_classes
        self.network = network
        self.map_metric = MeanAveragePrecision(box_format='xywh')
    
    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=5e-2,
            eps=1e-8,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            steps_per_epoch=10,
            epochs=20
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
            },
        }
    
    def forward(self, images, targets_batch=None):
        return self.network(images)
    
    def norm_targets(self, targets):
        return [{'labels':t['labels'], 'boxes':norm_bounding_boxes(t['boxes'])} for t in targets]
    
    def training_step(self, batch, batch_idx):
        images, targets = batch['sup']
        outputs = self.network(images)            
        loss_dict = self.criterion(outputs, self.norm_targets(targets))
        loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        self.log(f"loss/train", loss.detach().item())
        return loss
        
    def validation_step(self, batch, batch_idx):
        images, targets = batch
        outputs = self.network(images)
        loss_dict = self.criterion(outputs, self.norm_targets(targets))
        loss = sum(loss_dict[k] * self.weight_dict[k] for k in loss_dict.keys() if k in self.weight_dict)
        self.log(f"loss/val", loss.detach().item())
        target_sizes = torch.Tensor(tuple(images.size()[-2:])).repeat((images.size(0),1)).to(images.device)
        preds = self.post_process(outputs, target_sizes)
        # map_metric does not work yet with tv_tensors
        targets = [{'labels':t['labels'], 'boxes':t['boxes'].as_subclass(torch.Tensor)} for t in targets]
        self.map_metric.update(preds, targets)
        
    def on_validation_epoch_end(self):
        mapmetric = self.map_metric.compute()['map']
        self.log("map/val", mapmetric)
        print("\nMAP: ", mapmetric)
        self.map_metric.reset()
        
import pytorch_lightning as pl
from dl_toolbox.callbacks import ProgressBar
from dl_toolbox import datamodules
import torchvision.transforms.v2 as v2
import gc

network = Detector(60, 100)
matcher = HungarianMatcher()
weight_dict = {'loss_ce': 1, 'loss_bbox': 1, 'loss_giou':2}
losses = ['labels', 'boxes', 'cardinality']
criterion = SetCriterion(60, matcher=matcher, weight_dict=weight_dict,
                             eos_coef=0.5, losses=losses)

tf = v2.Compose(
    [
        v2.RandomCrop(224),
        v2.SanitizeBoundingBoxes(),
        #NormBoundingBoxes(),
        v2.ToDtype(
            dtype={tv_tensors.Image: torch.float32, "others":None}, 
            scale=True
        )
    ]
)
 
dm = datamodules.xView1(
    data_path='/data',
    merge='all60',
    train_tf=tf,
    test_tf=tf,
    batch_size=8,
    num_workers=4,
    pin_memory=False
)

gc.collect()
torch.cuda.empty_cache()
gc.collect()

trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=20,
    limit_train_batches=1.,
    limit_val_batches=1.,
    #callbacks=[ProgressBar()]
)

module = Yolos(
    network=network,
    criterion=criterion,
    weight_dict=weight_dict,
    post_process=PostProcess(),
    num_classes=60
)


trainer.fit(
    module,
    datamodule=dm,
)