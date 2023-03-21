import torch
import numpy as np
import torch.nn.functional as F

def dist_calculator(gt_bboxes, anchor_bboxes):
    """compute center distance between all bbox and gt

    Args:
        gt_bboxes (Tensor): shape(bs*n_max_boxes, 4)
        anchor_bboxes (Tensor): shape(num_total_anchors, 4)
    Return:
        distances (Tensor): shape(bs*n_max_boxes, num_total_anchors)
        ac_points (Tensor): shape(num_total_anchors, 2)
    """
    gt_cx = (gt_bboxes[:, 0] + gt_bboxes[:, 2]) / 2.0
    gt_cy = (gt_bboxes[:, 1] + gt_bboxes[:, 3]) / 2.0
    gt_points = torch.stack([gt_cx, gt_cy], dim=1)
    ac_cx = (anchor_bboxes[:, 0] + anchor_bboxes[:, 2]) / 2.0
    ac_cy = (anchor_bboxes[:, 1] + anchor_bboxes[:, 3]) / 2.0
    ac_points = torch.stack([ac_cx, ac_cy], dim=1)

    distances = (gt_points[:, None, :] - ac_points[None, :, :]).pow(2).sum(-1).sqrt()

    return distances, ac_points
def rbox2poly(obboxes):
    """
    Trans rbox format to poly format.
    Args:
        rboxes (array/tensor): (num_gts, [cx cy l s theta]) theta∈[0, 180)
    Returns:
        polys (array/tensor): (num_gts, [x1 y1 x2 y2 x3 y3 x4 y4])
    """
    if isinstance(obboxes, torch.Tensor):
        center, longSide, shortSide, theta = obboxes[:, :2], obboxes[:, 2:3], obboxes[:, 3:4], obboxes[:, 4:5]
        Cos, Sin = torch.cos(theta * torch.pi / 180.0), torch.sin(theta * torch.pi / 180.0)

        vector1 = torch.cat((longSide / 2.0 * Cos, longSide / 2.0 * Sin), dim=-1)
        vector2 = torch.cat((shortSide / 2.0 * Sin, -shortSide / 2.0 * Cos), dim=-1)
        point1 = center - vector1 - vector2
        point2 = center - vector1 + vector2
        point3 = center + vector1 + vector2
        point4 = center + vector1 - vector2
        order = obboxes.shape[:-1]
        return torch.cat((point1, point2, point3, point4), dim=-1).reshape(*order, 8)
    else:
        center, longSide, shortSide, theta = np.split(obboxes, (2, 3, 4), axis=-1)
        Cos, Sin = np.cos(theta), np.sin(theta)

        vector1 = np.concatenate([longSide / 2.0 * Cos, longSide / 2.0 * Sin], axis=-1)
        vector2 = np.concatenate([shortSide / 2.0 * Sin, -shortSide / 2.0 * Cos], axis=-1)

        point1 = center - vector1 - vector2
        point2 = center - vector1 + vector2
        point3 = center + vector1 + vector2
        point4 = center + vector1 - vector2
        order = obboxes.shape[:-1]
        return np.concatenate([point1, point2, point3, point4], axis=-1).reshape(*order, 8)
def select_candidates_in_gts_R(xy_center, gt_bboxes, gt_angle, eps=1e-9):
    """select the gt point in anchor's center in obb

    Args:
        anc_points (Tensor): shape(num_total_anchors, 2)
        gt_bboxes (tensor): shape(bs, n_max_boxes, 4)
        gt_angle (tensor): shape(bs, n_max_boxes, 1)

    Returns:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    gt_obbs = torch.concat([gt_bboxes, gt_angle], dim=-1)
    bs, n_max_boxes, _ = gt_bboxes.size()
    _gt_obbs = gt_obbs.reshape([bs,-1,5])
    _gt_poly = rbox2poly(_gt_obbs).reshape([bs,-1,4,2])
    points = xy_center.unsqueeze(0).unsqueeze(0)
    a,b,c,d = torch.split(_gt_poly,[1,1,1,1], dim=2)
    ab = b - a
    ad = d - a
    # [B, N, L, 2]
    ap = points - a
    # [B, N, L]
    norm_ab = torch.sum(ab * ab, axis=-1)
    # [B, N, L]
    norm_ad = torch.sum(ad * ad, axis=-1)
    # [B, N, L] dot product
    ap_dot_ab = torch.sum(ap * ab, axis=-1)
    # [B, N, L] dot product
    ap_dot_ad = torch.sum(ap * ad, axis=-1)
    # [B, N, L] <A, B> = |A|*|B|*cos(theta)
    is_in_box = (
        (ap_dot_ab >= 0)
        & (ap_dot_ab <= norm_ab)
        & (ap_dot_ad >= 0)
        & (ap_dot_ad <= norm_ad)
    )
    return is_in_box


def select_candidates_in_gts(xy_centers, gt_bboxes, eps=1e-9):
    """select the positive anchors's center in gt

    Args:
        xy_centers (Tensor): shape(bs*n_max_boxes, num_total_anchors, 4)
        gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    n_anchors = xy_centers.size(0)
    bs, n_max_boxes, _ = gt_bboxes.size()
    _gt_bboxes = gt_bboxes.reshape([-1, 4])
    xy_centers = xy_centers.unsqueeze(0).repeat(bs * n_max_boxes, 1, 1)
    gt_bboxes_lt = _gt_bboxes[:, 0:2].unsqueeze(1).repeat(1, n_anchors, 1)
    gt_bboxes_rb = _gt_bboxes[:, 2:4].unsqueeze(1).repeat(1, n_anchors, 1)
    b_lt = xy_centers - gt_bboxes_lt
    b_rb = gt_bboxes_rb - xy_centers
    bbox_deltas = torch.cat([b_lt, b_rb], dim=-1)
    bbox_deltas = bbox_deltas.reshape([bs, n_max_boxes, n_anchors, -1])
    return (bbox_deltas.min(axis=-1)[0] > eps).to(gt_bboxes.dtype)

def select_highest_overlaps(mask_pos, overlaps, n_max_boxes):
    """if an anchor box is assigned to multiple gts,
        the one with the highest iou will be selected.

    Args:
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
        overlaps (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    Return:
        target_gt_idx (Tensor): shape(bs, num_total_anchors)
        fg_mask (Tensor): shape(bs, num_total_anchors)
        mask_pos (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    fg_mask = mask_pos.sum(axis=-2)
    if fg_mask.max() > 1:
        mask_multi_gts = (fg_mask.unsqueeze(1) > 1).repeat([1, n_max_boxes, 1])
        max_overlaps_idx = overlaps.argmax(axis=1)
        is_max_overlaps = F.one_hot(max_overlaps_idx, n_max_boxes)
        is_max_overlaps = is_max_overlaps.permute(0, 2, 1).to(overlaps.dtype)
        mask_pos = torch.where(mask_multi_gts, is_max_overlaps, mask_pos)
        fg_mask = mask_pos.sum(axis=-2)
    target_gt_idx = mask_pos.argmax(axis=-2)
    return target_gt_idx, fg_mask , mask_pos

def iou_calculator(box1, box2, eps=1e-9):
    """Calculate iou for batch

    Args:
        box1 (Tensor): shape(bs, n_max_boxes, 1, 4)
        box2 (Tensor): shape(bs, 1, num_total_anchors, 4)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    box1 = box1.unsqueeze(2)  # [N, M1, 4] -> [N, M1, 1, 4]
    box2 = box2.unsqueeze(1)  # [N, M2, 4] -> [N, 1, M2, 4]
    px1y1, px2y2 = box1[:, :, :, 0:2], box1[:, :, :, 2:4]
    gx1y1, gx2y2 = box2[:, :, :, 0:2], box2[:, :, :, 2:4]
    x1y1 = torch.maximum(px1y1, gx1y1)
    x2y2 = torch.minimum(px2y2, gx2y2)
    overlap = (x2y2 - x1y1).clip(0).prod(-1)
    area1 = (px2y2 - px1y1).clip(0).prod(-1)
    area2 = (gx2y2 - gx1y1).clip(0).prod(-1)
    union = area1 + area2 - overlap + eps

    return overlap / union
