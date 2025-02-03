import torchvision
import torch
import torch.nn as nn

class FCOSLoss(nn.Module):

    def __init__(self, num_classes):
        super(FCOSLoss, self).__init__()
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
        #reg_loss, centerness_loss = 0,0
        #if num_pos > 0:
        reg_loss = self._get_reg_loss(
            reg_preds, reg_tgts, cness_tgts)
        centerness_loss = self._get_centerness_loss(
            cness_preds, cness_tgts, max(num_pos, 1.))
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
        if len(reg_tgts) == 0:
            return reg_tgts.new_zeros(len(reg_tgts))
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
        ltrb_preds = reg_preds.reshape(-1, 4)
        ltrb_tgts = reg_targets.reshape(-1, 4)
        xyxy_preds = torch.cat([-ltrb_preds[:,:2], ltrb_preds[:,2:]], dim=1) 
        xyxy_tgts = torch.cat([-ltrb_tgts[:,:2], ltrb_tgts[:,2:]], dim=1)
        reg_losses = torchvision.ops.distance_box_iou_loss(xyxy_preds, xyxy_tgts, reduction='none')
        sum_centerness_targets = centerness_targets.sum()
        reg_loss = (reg_losses * centerness_targets).sum() / (sum_centerness_targets+1e-8)
        return reg_loss

    def _get_centerness_loss(self, centerness_preds, centerness_targets,
                             num_pos_samples):
        centerness_loss = self.centerness_loss_func(centerness_preds,
                                                    centerness_targets)
        return centerness_loss / num_pos_samples
    