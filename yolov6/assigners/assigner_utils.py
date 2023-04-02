import torch
import numpy as np
import torch.nn.functional as F
from yolov6.utils.nms_R import xywh2xyxy
from yolov6.utils.general import Rbbox2dist
from mmcv.ops import box_iou_rotated

from yolov6.utils.nms_R import rbox2poly, rbox2poly_radius


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


def select_candidates_in_gts_R(xy_centers, gt_bboxes, gt_angles, eps=1e-9):
    """select the gt point in anchor's center in obb

    Args:
        anc_points (Tensor): shape(num_total_anchors, 2)
        gt_bboxes (tensor): shape(bs, n_max_boxes, 5) angles radius
        gt_angles (tensor): shape(bs, n_max_boxes, 1)

    Returns:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """

    gt_obbs = torch.concat([gt_bboxes, gt_angles], dim=-1)
    bs, n_max_boxes, _ = gt_bboxes.size()
    # NOTE [bs, N, 5] -> [bs, N, 4, 2]
    _gt_obbs = gt_obbs.reshape([bs, -1, 5])
    _gt_poly = rbox2poly(_gt_obbs).reshape([bs, -1, 4, 2])
    # _gt_poly = rbox2poly_radius(_gt_obbs).reshape([bs, -1, 4, 2])
    # NOTE [1, L, 2] -> [1, 1, L, 2]
    points = xy_centers.unsqueeze(0).unsqueeze(0)
    a, b, c, d = torch.split(_gt_poly, [1, 1, 1, 1], dim=2)
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
    is_in_box = (ap_dot_ab >= eps) & (ap_dot_ab <= norm_ab) & (ap_dot_ad >= eps) & (ap_dot_ad <= norm_ad)
    return is_in_box.to(gt_bboxes.dtype)


def select_candidates_in_gts(xy_centers, gt_bboxes, gt_angles, eps=1e-9):
    """select the positive anchors's center in gt

    Args:
        xy_centers (Tensor): shape(bs*n_max_boxes, num_total_anchors, 4)
        gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    gt_bboxes = xywh2xyxy(gt_bboxes)
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
    return target_gt_idx, fg_mask, mask_pos


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


def iou_calculator_xywh(box1, box2, eps=1e-9):
    """Calculate iou for batch

    Args:
        box1 (Tensor): shape(bs, n_max_boxes, 1, 4)
        box2 (Tensor): shape(bs, 1, num_total_anchors, 4)
    Return:
        (Tensor): shape(bs, n_max_boxes, num_total_anchors)
    """
    box1 = xywh2xyxy(box1)
    box2 = xywh2xyxy(box2)
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


def rbbox_overlaps(bboxes1, bboxes2, mode: str = "iou", is_aligned: bool = False):
    """Calculate overlap between two set of rotated bboxes.

    Args:
        bboxes1 (Tensor): shape (B, m, 5) in <cx, cy, w, h, t> format
            or empty.
        bboxes2 (Tensor): shape (B, n, 5) in <cx, cy, w, h, t> format
            or empty.
        mode (str): 'iou' (intersection over union), 'iof' (intersection over
            foreground). Defaults to 'iou'.
        is_aligned (bool): If True, then m and n must be equal.
            Defaults to False.

    Returns:
        Tensor: shape (m, n) if ``is_aligned`` is False else shape (m,)
    """
    assert mode in ["iou", "iof"]
    # Either the boxes are empty or the length of boxes's last dimension is 5
    assert bboxes1.size(-1) == 5 or bboxes1.size(0) == 0
    assert bboxes2.size(-1) == 5 or bboxes2.size(0) == 0

    rows = bboxes1.size(0)
    cols = bboxes2.size(0)
    if is_aligned:
        assert rows == cols

    if rows * cols == 0:
        return bboxes1.new(rows, 1) if is_aligned else bboxes1.new(rows, cols)

    # resolve `rbbox_overlaps` abnormal when input rbbox is too small.
    clamped_bboxes1 = bboxes1.detach().clone()
    clamped_bboxes2 = bboxes2.detach().clone()
    clamped_bboxes1[:, 2:4].clamp_(min=1e-3)
    clamped_bboxes2[:, 2:4].clamp_(min=1e-3)

    # resolve `rbbox_overlaps` abnormal when coordinate value is too large.
    # TODO: fix in mmcv
    clamped_bboxes1[:, :2].clamp_(min=-1e7, max=1e7)
    clamped_bboxes2[:, :2].clamp_(min=-1e7, max=1e7)

    return box_iou_rotated(clamped_bboxes1, clamped_bboxes2, mode, is_aligned)
