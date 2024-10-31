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
from torchvision.tv_tensors import BoundingBoxFormat
from dl_toolbox.utils import *
from torchvision.models.feature_extraction import create_feature_extractor
from dl_toolbox.modules import FeatureExtractor


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
        
        # target_classes is of shape batch_size x num det tokens, and is num_classes (=no_obj) everywhere, except for each token that is matched to a tgt, where it is the label of the matched tgt
        pred_logits = outputs['pred_logits'] #BxNdetTokxNcls
        target_classes = torch.full(
            pred_logits.shape[:2], #BxNdetTok
            self.num_classes, #Filled with num_cls
            dtype=torch.int64, 
            device=pred_logits.device
        )
        target_classes[(batch_idxs, pred_idxs)] = reordered_labels
        loss_ce = F.cross_entropy(
            pred_logits.transpose(1, 2), #BxNclsxd1xd2...
            target_classes, #Bxd1xd2...
            self.empty_weight
        )
        
        ## If we did as follows, then there would be no incentive for the network to output small logits for non-matched tokens
        #reordered_pred_logits = pred_logits[(batch_idxs, pred_idxs)] # NxNcls
        #other_loss_ce = F.cross_entropy(
        #    reordered_pred_logits,
        #    reordered_labels
        #)
        
        losses = {'loss_ce': loss_ce}
        return losses

    def loss_boxes(self, outputs, targets, matches, num_boxes):
        """Compute the losses related to the bounding boxes, the L1 regression loss and the GIoU loss
           targets dicts must contain the key "boxes" containing a tensor of dim [nb_target_boxes, 4]
        """
        reordered_target_boxes = [t['boxes'][i] for t, (_, i) in zip(targets, matches)]
        reordered_target_boxes = torch.cat(reordered_target_boxes) # Nx4
        #print(f'{reordered_target_boxes.shape =}')
        
        # batch_idxs[i] is the idx in the batch of the img to which the i-th elem in the new order corresponds
        batch_idxs = [torch.full_like(pred, i) for i, (pred, _) in enumerate(matches)]
        batch_idxs = torch.cat(batch_idxs) # Nx1
        
        # src_idxs[i] is the idx of the preds for img batch_idxs[i] to which the i-th elem in the new order corresponds
        pred_idxs = torch.cat([pred for (pred, _) in matches]) # Nx1
        
        pred_boxes = outputs['pred_boxes'] # BxNdetTokx4
        #print(f'{pred_boxes.shape =}')
        reordered_pred_boxes = pred_boxes[(batch_idxs, pred_idxs)] # Nx4
        #print(f'{reordered_pred_boxes.shape =}')

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
        class_list,
        det_token_num,
        encoder,
        optimizer,
        scheduler,
        pred_thresh,
        #tta=None,
        #sliding=None,
        *args,
        **kwargs
    ):
        super().__init__()
        self.class_list = class_list
        self.num_classes = len(class_list)
        self.det_token_num = det_token_num
        
        self.feature_extractor = FeatureExtractor(encoder)
        self.num_prefix_tokens = self.feature_extractor.encoder.num_prefix_tokens
        self.embed_dim = self.feature_extractor.encoder.embed_dim
        self.patch_size = self.feature_extractor.encoder.patch_embed.patch_size[0]
        
        #self.backbone = timm.create_model(
        #    encoder,
        #    pretrained=True,
        #    dynamic_img_size=False #Deals with inputs of other size than pretraining
        #)
        #self.feature_extractor = create_feature_extractor(
        #    self.backbone,
        #    {'norm': 'features'}
        #)
        #self.embed_dim = self.backbone.embed_dim 
        #self.num_prefix_tokens = self.backbone.num_prefix_tokens
        
        self.add_detection_tokens()
        self.class_embed = torchvision.ops.MLP(
            self.embed_dim,
            [self.embed_dim, self.embed_dim, self.num_classes+1]
        ) #Num_classes + 1 to deal with no_obj category
        self.bbox_embed = torchvision.ops.MLP(
            self.embed_dim,
            [self.embed_dim, self.embed_dim, 4]
        )
        self.loss = SetCriterion(
            self.num_classes,
            0.5 #Coeff of no-obj category
        )
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.map_metric = MeanAveragePrecision(
            box_format='xywh',
            backend='faster_coco_eval'
        )
        #self.sliding = sliding
        self.pred_thresh = pred_thresh
        
    def add_detection_tokens(self):
        det_token = nn.Parameter(
            torch.zeros(
                1,
                self.det_token_num,
                self.embed_dim
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
                self.embed_dim
            )
        )
        self.det_pos_embed = torch.nn.init.trunc_normal_(
            det_pos_embed,
            std=.02
        )
        #The ViT needs to know how many input tokens are not for patch embeddings
        self.num_prefix_tokens += self.det_token_num
    
    def configure_optimizers(self):
        trainable = lambda p: p[1].requires_grad
        train_params = list(filter(trainable, self.named_parameters()))
        nb_train = sum([int(torch.numel(p[1])) for p in train_params])
        nb_tot = sum([int(torch.numel(p)) for p in self.parameters()])
        print(f"Training {nb_train} trainable params out of {nb_tot}")
        #if hasattr(self, 'no_weight_decay'):
        #    wd_val = 0. #self.optimizer.weight_decay
        #    nwd_params = self.no_weight_decay()
        #    train_params = param_groups_weight_decay(train_params, wd_val, nwd_params)
        #    print(f"{len(train_params[0]['params'])} are not affected by weight decay.")
        optimizer = self.optimizer(params=[p[1] for p in train_params])
        scheduler = self.scheduler(optimizer)
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "epoch"
            },
        }
    
    def raw_logits_and_bboxs(self, x):
        """ This code relies on class_token=True in ViT
        """
        x = self.feature_extractor.encoder.patch_embed(x)
        
        # Inserting position embedding for detection tokens and resampling if dynamic
        cls_pos_embed = self.feature_extractor.encoder.pos_embed[:, 0, :][:,None] # size 1x1xembed_dim
        patch_pos_embed = self.feature_extractor.encoder.pos_embed[:, 1:, :] # 1xnum_patchxembed_dim
        pos_embed = torch.cat((cls_pos_embed, self.det_pos_embed, patch_pos_embed), dim=1)
        if self.feature_extractor.encoder.dynamic_img_size:
            B, H, W, C = x.shape
            pos_embed = resample_abs_pos_embed(
                pos_embed,
                (H, W),
                num_prefix_tokens=self.num_prefix_tokens,
            )
            x = x.view(B, -1, C)
            
        # Inserting detection tokens    
        cls_token = self.feature_extractor.encoder.cls_token.expand(x.shape[0], -1, -1) 
        det_token = self.det_token.expand(x.shape[0], -1, -1)
        x = torch.cat([cls_token, det_token, x], dim=1)
        
        # Forward ViT
        x += pos_embed
        x = self.feature_extractor.encoder.pos_drop(x)
        x = self.feature_extractor.encoder.patch_drop(x)
        x = self.feature_extractor.encoder.norm_pre(x)
        x = self.feature_extractor.encoder.blocks(x)
        x = self.feature_extractor.encoder.norm(x)
        
        # Extracting processed detection tokens + forward heads
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
        
        # Finds the minimum cost detection token/target assignment per img
        indices = [linear_sum_assignment(c[i]) for i, c in enumerate(C.split(sizes, -1))]
        int64 = lambda x: torch.as_tensor(x, dtype=torch.int64)
        return [(int64(i), int64(j)) for i, j in indices]
    
    @torch.no_grad()
    def post_process(self, outputs, images):
        b,c,h,w = images.shape
        target_sizes = torch.Tensor((h,w)).repeat((b,1)) # shape 2 -> bx2
        prob = F.softmax(outputs['pred_logits'], -1) # bxNdetTokxNcls
        # Most prob cls (except no-obj) and its score per img per token
        scores, labels = prob[..., :-1].max(-1) # bxNdetTok
        img_h, img_w = target_sizes.unbind(dim=1) # separates heights and widths, shapes b
        scale_fct = torch.stack([img_w, img_h, img_w, img_h], dim=1)[:, None, :] # bx1x4
        out_bbox = outputs['pred_boxes'] # bxNdetTokx4
        # This scaling should work whether out_bb have meaning in xyxy or xywh format 
        boxes = out_bbox * scale_fct.to(out_bbox.device)
        results = [{'scores': s, 'labels': l, 'boxes': b}
                   for s, l, b in zip(scores, labels, boxes)]
        return results
    
    def predict(self, x):
        outputs = self.forward(x)
        preds = self.post_process(outputs, x)
        return preds
    
    def norm_bb(self, inpt_bb):
        """
        Convert bb to xyxy format for normalization and back to their initial format (xywh?)
        """
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
        batch = batch["sup"]
        x = batch["image"]
        y = batch["target"]
        #x, targets, paths = batch["sup"]
        outputs = self.raw_logits_and_bboxs(x)
        # Out bb corners come from a sigmoid layer, so we need tgt bb to be in [0,1] too
        norm_tgts = self.norm_targets(y) 
        matches = self.hungarian_matching(outputs, norm_tgts)
        loss = self.loss(outputs, norm_tgts, matches)
        self.log(f"Loss/train", loss.detach().item())
        return loss
        
    def validation_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["target"]
        
        #x, targets, paths = batch
        outputs = self.forward(x)
        norm_tgts = self.norm_targets(y)
        matches = self.hungarian_matching(outputs, norm_tgts)
        loss = self.loss(outputs, norm_tgts, matches)
        self.log(f"Loss/val", loss.detach().item())
        # map_metric reuires a certain form for predictions
        preds = self.post_process(outputs, x)
        # map_metric does not work yet with tv_tensors
        targets = [{'labels':t['labels'], 'boxes':t['boxes'].as_subclass(torch.Tensor)} for t in y]
        # self.map_metric works with bb in xywh format
        # Make sure your dataset outputs targets in XYWH format
        self.map_metric.update(preds, targets)
        
    def on_validation_epoch_end(self):
        mapmetric = self.map_metric.compute()['map']
        self.log("map/val", mapmetric)
        print("\nMAP: ", mapmetric)
        self.map_metric.reset()
        
    def predict_step(self, batch, batch_idx):
        x = batch["image"]
        y = batch["target"]
        #x, targets, paths = batch
        outputs = self.forward(x)
        preds = self.post_process(outputs, x)
        return preds