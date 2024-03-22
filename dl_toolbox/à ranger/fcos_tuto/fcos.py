from torch.utils.data import Dataset
from PIL import Image
import torch
from torchvision.transforms import v2
from torchvision import tv_tensors

class ObjDetDataset(Dataset):

    def __init__(self, data, transforms=None):
        image_paths = []
        targets = []
        for instance in data:
            image_paths.append(instance['image_path'])
            targets.append(instance["target"])
        self.image_paths = image_paths
        self.targets = targets
        self.transforms = transforms

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        image_path = self.image_paths[idx]
        image = Image.open(image_path).convert("RGB")
        w, h = image.size
        image = v2.functional.pil_to_tensor(image)
        targets = self.targets[idx]
        targets = torch.Tensor(targets)
        bboxes = tv_tensors.BoundingBoxes(targets[:,:4], format="XYXY", canvas_size=(h,w))
        labels = targets[:, 4:]
        if self.transforms:
            image, bboxes = self.transforms(image, bboxes)
        return image, bboxes, labels, image_path
    
from torch.utils.data import DataLoader, RandomSampler
from pathlib import Path
import pandas as pd
import numpy as np
from pytorch_lightning import LightningDataModule
from pytorch_lightning.utilities import CombinedLoader
from functools import partial
from torchvision.models import resnet50, ResNet50_Weights

IMAGE_SIZE=480
STEPS = 1000


class PascalVOC(LightningDataModule):
    
    def __init__(
        self,
        data_path,
        train_tf,
        test_tf,
        batch_size_s,
        steps_per_epoch,
        num_workers,
        pin_memory,
        *args,
        **kwargs
    ):
        super().__init__()
        self.data_path = Path(data_path)
        self.train_tf = train_tf
        self.test_tf = test_tf
        self.batch_size_s = batch_size_s
        self.steps_per_epoch = steps_per_epoch
        self.num_workers = num_workers
        self.pin_memory = pin_memory
    
    def prepare_data(self):
        img_dir = self.data_path/"PASCALVOC/VOCdevkit/VOC2012/JPEGImages"
        self.instances = []
        labels = ['__background__', 'aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog', 'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'tvmonitor']
        for _, row in pd.read_pickle("voc_combined.csv").iterrows():
            img_path = row["filename"]
            labels_ = row["labels"]
            image_path = f"{img_dir}/{img_path}"
            labels_ = [[labels.index(l)] for l in labels_]
            targets_ = np.concatenate([row["bboxes"], labels_],
                                      axis=-1).tolist()
            self.instances.append({"image_path": image_path, "target": targets_})
    
    def setup(self, stage=None):
        split = int(0.95*len(self.instances))
        train_data = self.instances[:split]
        val_data = self.instances[split:]
        self.train_s_set = ObjDetDataset(train_data, transforms=self.train_tf)
        self.val_set = ObjDetDataset(val_data, transforms=self.test_tf)
    
    @staticmethod
    def _collate(batch):
        images_b, bboxes_b, labels_b, image_paths_b = list(zip(*batch))
        # don't stack bb because each batch elem may not have the same nb of bb
        return torch.stack(images_b), bboxes_b, labels_b, image_paths_b 
                
    def _dataloader(self, dataset):
        return partial(
            DataLoader,
            dataset=dataset,
            collate_fn=self._collate,
            num_workers=self.num_workers,
            pin_memory=self.pin_memory
        )
                       
    def train_dataloader(self):
        return self._dataloader(self.train_s_set)(
            sampler=RandomSampler(
                self.train_s_set,
                replacement=True,
                num_samples=self.steps_per_epoch*self.batch_size_s
            ),
            drop_last=True,
            batch_size=self.batch_size_s
        )
    
    def val_dataloader(self):
        return self._dataloader(self.val_set)(
            shuffle=False,
            drop_last=False,
            batch_size=self.batch_size_s
        )
    
train_tf = v2.Compose([
    v2.Resize(size=(IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    v2.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.2),
    #v2.CenterCrop(size=(224,224)),
    v2.RandomHorizontalFlip(p=0.5),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

test_tf = v2.Compose([
    v2.Resize(size=(IMAGE_SIZE, IMAGE_SIZE), antialias=True),
    #v2.CenterCrop(size=(224,224)),
    v2.ToDtype(torch.float32, scale=True),
    v2.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])

dm = PascalVOC(
    data_path='/data',
    train_tf=train_tf,
    test_tf=test_tf, # taken from https://pytorch.org/vision/0.17/models/generated/torchvision.models.resnet50.html#torchvision.models.ResNet50_Weights
    batch_size_s=4,
    steps_per_epoch=STEPS,
    num_workers=6,
    pin_memory=True,
)

from torchvision.models import resnet50, ResNet50_Weights
from torchvision.models import convnext_tiny, ConvNeXt_Tiny_Weights
from torchvision.models.feature_extraction import get_graph_node_names
from torchvision.models.feature_extraction import create_feature_extractor
from torchvision.ops.feature_pyramid_network import FeaturePyramidNetwork, LastLevelP6P7
from dl_toolbox.networks.fcos import Head
import torch.nn as nn

class FCOS(torch.nn.Module):
    
    def __init__(self, num_classes=19, out_channels=256, image_size=IMAGE_SIZE):
        super(FCOS, self).__init__()
        #backbone = resnet50(weights=None)#ResNet50_Weights.DEFAULT)
        backbone = convnext_tiny(weights=ConvNeXt_Tiny_Weights.IMAGENET1K_V1)
        return_nodes = {
            'features.3.2.add': 'layer2',
            'features.5.8.add': 'layer3',
            'features.7.2.add': 'layer4',
        }
        # Extract 3 main layers
        self.feature_extractor = create_feature_extractor(backbone, return_nodes)
        # Dry run to get number of channels for FPN
        inp = torch.randn(2, 3, image_size, image_size)
        with torch.no_grad():
            out = self.feature_extractor(inp)
        in_channels_list = [o.shape[1] for o in out.values()]
        # Build FPN
        fpn = FeaturePyramidNetwork(
            in_channels_list,
            out_channels=out_channels,
            extra_blocks=LastLevelP6P7(out_channels,out_channels)
        )
        self.fpn_features = nn.Sequential(self.feature_extractor, fpn)
        inp = torch.randn(2, 3, image_size, image_size)
        with torch.no_grad():
            out = self.fpn_features(inp)
        self.feat_sizes = [o.shape[2:] for o in out.values()]
        self.head = Head(out_channels, num_classes)

    def forward(self, images):
        features = list(self.fpn_features(images).values())
        box_cls, box_regression, centerness = self.head(features)
        all_level_preds = (torch.cat([t.flatten(-2) for t in o], dim=-1) for o in [features, box_cls, box_regression, centerness])
        return (torch.permute(t, (0,2,1)) for t in all_level_preds)
    
network = FCOS(num_classes=19)

import torch.nn as nn
import torchvision

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

def giou(pred, target, weight=None):

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
    #ious = (area_intersect + 1.0) / (area_union + 1.0)
    ious = (area_intersect) / (area_union + 1e-7)
    gious = ious - (ac_uion - area_union) / ac_uion
    losses = 1 - gious

    if weight is not None and weight.sum() > 0:
        return (losses * weight).sum()
    else:
        assert losses.numel() != 0
        return losses.sum()

import pytorch_lightning as pl
from torchmetrics.detection.mean_ap import MeanAveragePrecision
import torch.nn.functional as F
from pytorch_lightning.utilities import rank_zero_info

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
        
    def forward(self,
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


INF = 100000000

class FCOS(pl.LightningModule):
    def __init__(
        self,
        network,
        num_classes,
        *args,
        **kwargs
    ):
        super().__init__()
        
        self.post_processor = FCOSPostProcessor(
            pre_nms_thresh=0.3,
            pre_nms_top_n=1000,
            nms_thresh=0.45,
            fpn_post_nms_top_n=50,
            min_size=0,
            num_classes=num_classes)
        self.fpn_strides = [8, 16, 32, 64, 128]
        self.feat_sizes = network.feat_sizes
        self.max_dist_per_level = [-1, 64, 128, 256, 512, INF]
        self.num_classes = num_classes
        self.network = network
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
        optimizer = torch.optim.AdamW(
            trainable_parameters,
            lr=1e-3,
            betas=(0.9, 0.999),
            weight_decay=5e-2,
            eps=1e-8,
        )
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=1e-3,
            steps_per_epoch=STEPS,
            epochs=100
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {
                "scheduler": scheduler,
                "interval": "step"
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
        predicted_boxes, scores, all_classes = self.post_processor(
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
        
import pytorch_lightning as pl
from dl_toolbox.callbacks import ProgressBar, FeatureFt
import os

tensorboard = pl.loggers.TensorBoardLogger(
    "/data/outputs/test_fcos/2103_1344", "", "", default_hp_metric=False
)



trainer = pl.Trainer(
    accelerator='gpu',
    devices=1,
    max_epochs=200,
    limit_train_batches=1.,
    limit_val_batches=1.,
    callbacks=[FeatureFt(do_finetune=False, unfreeze_at_epoch=50)],
    logger=tensorboard
)

module = FCOS(
    network,
    num_classes=19
)


trainer.fit(
    module,
    datamodule=dm,
)