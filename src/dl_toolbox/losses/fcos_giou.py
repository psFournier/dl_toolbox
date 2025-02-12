import torch 

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