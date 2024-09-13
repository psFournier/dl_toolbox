import torch.nn as nn
import torch
import torchvision

class FCOSPostProcessor(nn.Module):

    def __init__(self, locs_info, pre_nms_thresh, pre_nms_top_n, nms_thresh,
                 fpn_post_nms_top_n, min_size, num_classes):
        super(FCOSPostProcessor, self).__init__()
        self.pre_nms_thresh = pre_nms_thresh
        self.pre_nms_top_n = pre_nms_top_n
        self.nms_thresh = nms_thresh
        self.fpn_post_nms_top_n = fpn_post_nms_top_n
        self.min_size = min_size
        self.num_classes = num_classes
        locs_per_level, bb_sizes_per_level, num_locs_per_level = locs_info
        self.register_buffer('locations', torch.cat(locs_per_level, dim=0)) # Lx2
        self.num_locs_per_level = num_locs_per_level

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
            # Tensor with true where score for loc l and class c > pre_nms_thresh
            per_candidate_inds = candidate_inds[i] # HWxC
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
            
            per_cls_preds = cls_preds[i] # HWxC
            # tenseur de taille L avec les elem de cls_preds*centerness tels que cls_preds > nms_thresh
            per_cls_preds = per_cls_preds[per_candidate_inds] 
            
            # si y a plus de per_prenms_topn qui passe nms_thresh (si L est trop longue)
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
            ],
                                     dim=1)
            per_bboxes = torchvision.ops.clip_boxes_to_image(per_bboxes, (image_size, image_size))
            #detections = _clip_to_image(detections, (image_size, image_size))
            per_bboxes = per_bboxes[torchvision.ops.remove_small_boxes(per_bboxes, self.min_size)]
            #detections = remove_small_boxes(detections, self.min_size)
            bboxes.append(per_bboxes)
            cls_labels.append(per_class)
            scores.append(torch.sqrt(per_cls_preds))
            
        #bboxes is a list of B tensors of size Lx4 (potentially filtered with pre_nms_threshold)
        return bboxes, scores, cls_labels

    def forward(self, cls_preds, reg_preds, cness_preds, image_size):
        # loc: list of n_feat_level tensors of size HW(level)
        # reg_preds: list of n_feat_level tensors BxHW(level)x4
        
        # list of n_feat_level lists of B tensors of size Lx4
        sampled_boxes = []
        all_scores = []
        all_classes = []
        locations = torch.split(self.locations, self.num_locs_per_level, dim=0)
        for l, o, b, c in list(zip(locations, cls_preds, reg_preds,
                                   cness_preds)):
            boxes, scores, cls_labels = self.forward_for_single_feature_map(
                l, o, b, c, image_size)
            # boxes : list of B tensors Lx4
            sampled_boxes.append(boxes)
            all_scores.append(scores)
            all_classes.append(cls_labels)
        
        # list of B lists of n_feat_level bbox preds
        all_bboxes = list(zip(*sampled_boxes))
        all_scores = list(zip(*all_scores))
        all_classes = list(zip(*all_classes))
    
        # list of B tensors with all feature level bbox preds grouped
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
            picked_indices = torchvision.ops.nms(boxlists[i], scores[i], self.nms_thresh)
            picked_boxes = boxlists[i][picked_indices]
            confidence_scores = scores[i][picked_indices]
            picked_classes = classes[i][picked_indices]

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

            all_picked_boxes.append(picked_boxes)
            all_confidence_scores.append(confidence_scores)
            all_classes.append(picked_classes)
        
        # all_picked_boxes : list of B tensors with all feature level bbox preds filtered by nms
        return all_picked_boxes, all_confidence_scores, all_classes