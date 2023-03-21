# Copyright (c) OpenMMLab. All rights reserved.
import math

import cv2
import numpy as np
import torch
import mmcv.ops.box_iou_rotated as rotate_iou

def bbox_flip(bboxes, img_shape, direction='horizontal'):
    """Flip bboxes horizontally or vertically.
    Args:
        bboxes (Tensor): Shape (..., 5*k)
        img_shape (tuple): Image shape.
        direction (str): Flip direction, options are "horizontal", "vertical",
            "diagonal". Default: "horizontal"
    Returns:
        Tensor: Flipped bboxes.
    """
    version = 'oc'
    assert bboxes.shape[-1] % 5 == 0
    assert direction in ['horizontal', 'vertical', 'diagonal']
    flipped = bboxes.clone()
    if direction == 'horizontal':
        flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
    elif direction == 'vertical':
        flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
    else:
        flipped[:, 0] = img_shape[1] - bboxes[:, 0] - 1
        flipped[:, 1] = img_shape[0] - bboxes[:, 1] - 1
    if version == 'oc':
        rotated_flag = (bboxes[:, 4] != np.pi / 2)
        flipped[rotated_flag, 4] = np.pi / 2 - bboxes[rotated_flag, 4]
        flipped[rotated_flag, 2] = bboxes[rotated_flag, 3]
        flipped[rotated_flag, 3] = bboxes[rotated_flag, 2]
    else:
        flipped[:, 4] = norm_angle(np.pi - bboxes[:, 4], version)
    return flipped


def bbox_mapping_back(bboxes,
                      img_shape,
                      scale_factor,
                      flip,
                      flip_direction='horizontal'):
    """Map bboxes from testing scale to original image scale."""
    new_bboxes = bbox_flip(bboxes, img_shape,
                           flip_direction) if flip else bboxes
    new_bboxes[:, :4] = new_bboxes[:, :4] / new_bboxes.new_tensor(scale_factor)
    return new_bboxes.view(bboxes.shape)


def rbbox2result(bboxes, labels, num_classes):
    """Convert detection results to a list of numpy arrays.
    Args:
        bboxes (torch.Tensor): shape (n, 6)
        labels (torch.Tensor): shape (n, )
        num_classes (int): class number, including background class
    Returns:
        list(ndarray): bbox results of each class
    """
    if bboxes.shape[0] == 0:
        return [np.zeros((0, 6), dtype=np.float32) for _ in range(num_classes)]
    else:
        bboxes = bboxes.cpu().numpy()
        labels = labels.cpu().numpy()
        return [bboxes[labels == i, :] for i in range(num_classes)]


def rbbox2roi(bbox_list):
    """Convert a list of bboxes to roi format.
    Args:
        bbox_list (list[Tensor]): a list of bboxes corresponding to a batch
            of images.
    Returns:
        Tensor: shape (n, 6), [batch_ind, cx, cy, w, h, a]
    """
    rois_list = []
    for img_id, bboxes in enumerate(bbox_list):
        if bboxes.size(0) > 0:
            img_inds = bboxes.new_full((bboxes.size(0), 1), img_id)
            rois = torch.cat([img_inds, bboxes[:, :5]], dim=-1)
        else:
            rois = bboxes.new_zeros((0, 6))
        rois_list.append(rois)
    rois = torch.cat(rois_list, 0)
    return rois


def poly2obb(polys, version='oc'):
    """Convert polygons to oriented bounding boxes.
    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
        version (Str): angle representations.
    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = poly2obb_oc(polys)
    elif version == 'le135':
        results = poly2obb_le135(polys)
    elif version == 'le90':
        results = poly2obb_le90(polys)
    else:
        raise NotImplementedError
    return results


def poly2obb_np(polys, version='oc'):
    """Convert polygons to oriented bounding boxes.
    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
        version (Str): angle representations.
    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = poly2obb_np_oc(polys)
    elif version == 'le135':
        results = poly2obb_np_le135(polys)
    elif version == 'le90':
        results = poly2obb_np_le90(polys)
    else:
        raise NotImplementedError
    return results


def obb2hbb(rbboxes, version='oc'):
    """Convert oriented bounding boxes to horizontal bounding boxes.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.
    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
    """
    if version == 'oc':
        results = obb2hbb_oc(rbboxes)
    elif version == 'le135':
        results = obb2hbb_le135(rbboxes)
    elif version == 'le90':
        results = obb2hbb_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def obb2poly(rbboxes, version='oc'):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if version == 'oc':
        results = obb2poly_oc(rbboxes)
    elif version == 'le135':
        results = obb2poly_le135(rbboxes)
    elif version == 'le90':
        results = obb2poly_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def obb2poly_np(rbboxes, version='oc'):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.
    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    if version == 'oc':
        results = obb2poly_np_oc(rbboxes)
    elif version == 'le135':
        results = obb2poly_np_le135(rbboxes)
    elif version == 'le90':
        results = obb2poly_np_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def obb2xyxy(rbboxes, version='oc'):
    """Convert oriented bounding boxes to horizontal bounding boxes.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
        version (Str): angle representations.
    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    if version == 'oc':
        results = obb2xyxy_oc(rbboxes)
    elif version == 'le135':
        results = obb2xyxy_le135(rbboxes)
    elif version == 'le90':
        results = obb2xyxy_le90(rbboxes)
    else:
        raise NotImplementedError
    return results


def hbb2obb(hbboxes, version='oc'):
    """Convert horizontal bounding boxes to oriented bounding boxes.
    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
        version (Str): angle representations.
    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    if version == 'oc':
        results = hbb2obb_oc(hbboxes)
    elif version == 'le135':
        results = hbb2obb_le135(hbboxes)
    elif version == 'le90':
        results = hbb2obb_le90(hbboxes)
    else:
        raise NotImplementedError
    return results


def poly2obb_oc(polys):
    """Convert polygons to oriented bounding boxes.
    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    points = torch.reshape(polys, [-1, 4, 2])
    cxs = torch.unsqueeze(torch.sum(points[:, :, 0], axis=1), axis=1) / 4.
    cys = torch.unsqueeze(torch.sum(points[:, :, 1], axis=1), axis=1) / 4.
    _ws = torch.unsqueeze(dist_torch(points[:, 0], points[:, 1]), axis=1)
    _hs = torch.unsqueeze(dist_torch(points[:, 1], points[:, 2]), axis=1)
    _thetas = torch.unsqueeze(
        torch.atan2(-(points[:, 1, 0] - points[:, 0, 0]),
                    points[:, 1, 1] - points[:, 0, 1]),
        axis=1)
    odd = torch.eq(torch.remainder((_thetas / (np.pi * 0.5)).floor_(), 2), 0)
    ws = torch.where(odd, _hs, _ws)
    hs = torch.where(odd, _ws, _hs)
    thetas = torch.remainder(_thetas, np.pi * 0.5)
    rbboxes = torch.cat([cxs, cys, ws, hs, thetas], axis=1)
    return rbboxes


def poly2obb_le135(polys):
    """Convert polygons to oriented bounding boxes.
    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]
    angles = norm_angle(angles, 'le135')
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


def poly2obb_le90(polys):
    """Convert polygons to oriented bounding boxes.
    Args:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    polys = torch.reshape(polys, [-1, 8])
    pt1, pt2, pt3, pt4 = polys[..., :8].chunk(4, 1)
    edge1 = torch.sqrt(
        torch.pow(pt1[..., 0] - pt2[..., 0], 2) +
        torch.pow(pt1[..., 1] - pt2[..., 1], 2))
    edge2 = torch.sqrt(
        torch.pow(pt2[..., 0] - pt3[..., 0], 2) +
        torch.pow(pt2[..., 1] - pt3[..., 1], 2))
    angles1 = torch.atan2((pt2[..., 1] - pt1[..., 1]),
                          (pt2[..., 0] - pt1[..., 0]))
    angles2 = torch.atan2((pt4[..., 1] - pt1[..., 1]),
                          (pt4[..., 0] - pt1[..., 0]))
    angles = polys.new_zeros(polys.shape[0])
    angles[edge1 > edge2] = angles1[edge1 > edge2]
    angles[edge1 <= edge2] = angles2[edge1 <= edge2]
    angles = norm_angle(angles, 'le90')
    x_ctr = (pt1[..., 0] + pt3[..., 0]) / 2.0
    y_ctr = (pt1[..., 1] + pt3[..., 1]) / 2.0
    edges = torch.stack([edge1, edge2], dim=1)
    width, _ = torch.max(edges, 1)
    height, _ = torch.min(edges, 1)
    return torch.stack([x_ctr, y_ctr, width, height, angles], 1)


def poly2obb_np_oc(poly):
    """Convert polygons to oriented bounding boxes.
    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    while not 0 < a <= 90:
        if a == -90:
            a += 180
        else:
            a += 90
            w, h = h, w
    a = a / 180 * np.pi
    assert 0 < a <= np.pi / 2
    return x, y, w, h, a


def poly2obb_np_le135(poly):
    """Convert polygons to oriented bounding boxes.
    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)
    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])
    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) + (pt1[1] - pt2[1]) *
                    (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) + (pt2[1] - pt3[1]) *
                    (pt2[1] - pt3[1]))
    if edge1 < 2 or edge2 < 2:
        return
    width = max(edge1, edge2)
    height = min(edge1, edge2)
    angle = 0
    if edge1 > edge2:
        angle = np.arctan2(float(pt2[1] - pt1[1]), float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        angle = np.arctan2(float(pt4[1] - pt1[1]), float(pt4[0] - pt1[0]))
    angle = norm_angle(angle, 'le135')
    x_ctr = float(pt1[0] + pt3[0]) / 2
    y_ctr = float(pt1[1] + pt3[1]) / 2
    return x_ctr, y_ctr, width, height, angle


def poly2obb_np_le90(poly):
    """Convert polygons to oriented bounding boxes.
    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]
    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(bboxps)
    x, y, w, h, a = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[
        2]
    if w < 2 or h < 2:
        return
    a = a / 180 * np.pi
    if w < h:
        w, h = h, w
        a += np.pi / 2
    while not np.pi / 2 > a >= -np.pi / 2:
        if a >= np.pi / 2:
            a -= np.pi
        else:
            a += np.pi
    assert np.pi / 2 > a >= -np.pi / 2
    return x, y, w, h, a


def obb2poly_oc(rboxes):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x = rboxes[:, 0]
    y = rboxes[:, 1]
    w = rboxes[:, 2]
    h = rboxes[:, 3]
    a = rboxes[:, 4]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    return torch.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y], dim=-1)


def obb2poly_le135(rboxes):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


def obb2poly_le90(rboxes):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        polys (torch.Tensor): [x0,y0,x1,y1,x2,y2,x3,y3]
    """
    N = rboxes.shape[0]
    if N == 0:
        return rboxes.new_zeros((rboxes.size(0), 8))
    x_ctr, y_ctr, width, height, angle = rboxes.select(1, 0), rboxes.select(
        1, 1), rboxes.select(1, 2), rboxes.select(1, 3), rboxes.select(1, 4)
    tl_x, tl_y, br_x, br_y = \
        -width * 0.5, -height * 0.5, \
        width * 0.5, height * 0.5
    rects = torch.stack([tl_x, br_x, br_x, tl_x, tl_y, tl_y, br_y, br_y],
                        dim=0).reshape(2, 4, N).permute(2, 0, 1)
    sin, cos = torch.sin(angle), torch.cos(angle)
    M = torch.stack([cos, -sin, sin, cos], dim=0).reshape(2, 2,
                                                          N).permute(2, 0, 1)
    polys = M.matmul(rects).permute(2, 1, 0).reshape(-1, N).transpose(1, 0)
    polys[:, ::2] += x_ctr.unsqueeze(1)
    polys[:, 1::2] += y_ctr.unsqueeze(1)
    return polys.contiguous()


def obb2hbb_oc(rbboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,pi/2]
    """
    w = rbboxes[:, 2::5]
    h = rbboxes[:, 3::5]
    a = rbboxes[:, 4::5]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    hbboxes = rbboxes.clone().detach()
    hbboxes[:, 2::5] = hbbox_h
    hbboxes[:, 3::5] = hbbox_w
    hbboxes[:, 4::5] = np.pi / 2
    return hbboxes


def obb2hbb_le135(rotatex_boxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
    """
    polys = obb2poly_le135(rotatex_boxes)
    xmin, _ = polys[:, ::2].min(1)
    ymin, _ = polys[:, 1::2].min(1)
    xmax, _ = polys[:, ::2].max(1)
    ymax, _ = polys[:, 1::2].max(1)
    bboxes = torch.stack([xmin, ymin, xmax, ymax], dim=1)
    x_ctr = (bboxes[:, 2] + bboxes[:, 0]) / 2.0
    y_ctr = (bboxes[:, 3] + bboxes[:, 1]) / 2.0
    edges1 = torch.abs(bboxes[:, 2] - bboxes[:, 0])
    edges2 = torch.abs(bboxes[:, 3] - bboxes[:, 1])
    angles = bboxes.new_zeros(bboxes.size(0))
    inds = edges1 < edges2
    rotated_boxes = torch.stack((x_ctr, y_ctr, edges1, edges2, angles), dim=1)
    rotated_boxes[inds, 2] = edges2[inds]
    rotated_boxes[inds, 3] = edges1[inds]
    rotated_boxes[inds, 4] = np.pi / 2.0
    return rotated_boxes


def obb2hbb_le90(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        hbbs (torch.Tensor): [x_ctr,y_ctr,w,h,-pi/2]
    """
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    hbboxes = torch.cat([center - bias, center + bias], dim=-1)
    _x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    _y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    _w = hbboxes[..., 2] - hbboxes[..., 0]
    _h = hbboxes[..., 3] - hbboxes[..., 1]
    _theta = theta.new_zeros(theta.size(0))
    obboxes1 = torch.stack([_x, _y, _w, _h, _theta], dim=-1)
    obboxes2 = torch.stack([_x, _y, _h, _w, _theta - np.pi / 2], dim=-1)
    obboxes = torch.where((_w >= _h)[..., None], obboxes1, obboxes2)
    return obboxes


def hbb2obb_oc(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.
    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    rbboxes = torch.stack([x, y, h, w, theta + np.pi / 2], dim=-1)
    return rbboxes


def hbb2obb_le135(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.
    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    obboxes1 = torch.stack([x, y, w, h, theta], dim=-1)
    obboxes2 = torch.stack([x, y, h, w, theta + np.pi / 2], dim=-1)
    obboxes = torch.where((w >= h)[..., None], obboxes1, obboxes2)
    return obboxes


def hbb2obb_le90(hbboxes):
    """Convert horizontal bounding boxes to oriented bounding boxes.
    Args:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    Returns:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    """
    x = (hbboxes[..., 0] + hbboxes[..., 2]) * 0.5
    y = (hbboxes[..., 1] + hbboxes[..., 3]) * 0.5
    w = hbboxes[..., 2] - hbboxes[..., 0]
    h = hbboxes[..., 3] - hbboxes[..., 1]
    theta = x.new_zeros(*x.shape)
    obboxes1 = torch.stack([x, y, w, h, theta], dim=-1)
    obboxes2 = torch.stack([x, y, h, w, theta - np.pi / 2], dim=-1)
    obboxes = torch.where((w >= h)[..., None], obboxes1, obboxes2)
    return obboxes


def obb2xyxy_oc(rbboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    w = rbboxes[:, 2::5]
    h = rbboxes[:, 3::5]
    a = rbboxes[:, 4::5]
    cosa = torch.cos(a)
    sina = torch.sin(a)
    hbbox_w = cosa * w + sina * h
    hbbox_h = sina * w + cosa * h
    # pi/2 >= a > 0, so cos(a)>0, sin(a)>0
    dx = rbboxes[..., 0]
    dy = rbboxes[..., 1]
    dw = hbbox_w.reshape(-1)
    dh = hbbox_h.reshape(-1)
    x1 = dx - dw / 2
    y1 = dy - dh / 2
    x2 = dx + dw / 2
    y2 = dy + dh / 2
    return torch.stack((x1, y1, x2, y2), -1)


def obb2xyxy_le135(rotatex_boxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    N = rotatex_boxes.shape[0]
    if N == 0:
        return rotatex_boxes.new_zeros((rotatex_boxes.size(0), 4))
    polys = obb2poly_le135(rotatex_boxes)
    xmin, _ = polys[:, ::2].min(1)
    ymin, _ = polys[:, 1::2].min(1)
    xmax, _ = polys[:, ::2].max(1)
    ymax, _ = polys[:, 1::2].max(1)
    return torch.stack([xmin, ymin, xmax, ymax], dim=1)


def obb2xyxy_le90(obboxes):
    """Convert oriented bounding boxes to horizontal bounding boxes.
    Args:
        obbs (torch.Tensor): [x_ctr,y_ctr,w,h,angle]
    Returns:
        hbbs (torch.Tensor): [x_lt,y_lt,x_rb,y_rb]
    """
    # N = obboxes.shape[0]
    # if N == 0:
    #     return obboxes.new_zeros((obboxes.size(0), 4))
    center, w, h, theta = torch.split(obboxes, [2, 1, 1, 1], dim=-1)
    Cos, Sin = torch.cos(theta), torch.sin(theta)
    x_bias = torch.abs(w / 2 * Cos) + torch.abs(h / 2 * Sin)
    y_bias = torch.abs(w / 2 * Sin) + torch.abs(h / 2 * Cos)
    bias = torch.cat([x_bias, y_bias], dim=-1)
    return torch.cat([center - bias, center + bias], dim=-1)


def obb2poly_np_oc(rbboxes):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]
    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    x = rbboxes[:, 0]
    y = rbboxes[:, 1]
    w = rbboxes[:, 2]
    h = rbboxes[:, 3]
    a = rbboxes[:, 4]
    score = rbboxes[:, 5]
    cosa = np.cos(a)
    sina = np.sin(a)
    wx, wy = w / 2 * cosa, w / 2 * sina
    hx, hy = -h / 2 * sina, h / 2 * cosa
    p1x, p1y = x - wx - hx, y - wy - hy
    p2x, p2y = x + wx - hx, y + wy - hy
    p3x, p3y = x + wx + hx, y + wy + hy
    p4x, p4y = x - wx + hx, y - wy + hy
    polys = np.stack([p1x, p1y, p2x, p2y, p3x, p3y, p4x, p4y, score], axis=-1)
    polys = get_best_begin_point(polys)
    return polys


def obb2poly_np_le135(rrects):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]
    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    polys = []
    for rrect in rrects:
        x_ctr, y_ctr, width, height, angle, score = rrect[:6]
        tl_x, tl_y, br_x, br_y = -width / 2, -height / 2, width / 2, height / 2
        rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
        R = np.array([[np.cos(angle), -np.sin(angle)],
                      [np.sin(angle), np.cos(angle)]])
        poly = R.dot(rect)
        x0, x1, x2, x3 = poly[0, :4] + x_ctr
        y0, y1, y2, y3 = poly[1, :4] + y_ctr
        poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3, score],
                        dtype=np.float32)
        polys.append(poly)
    polys = np.array(polys)
    polys = get_best_begin_point(polys)
    return polys


def obb2poly_np_le90(obboxes):
    """Convert oriented bounding boxes to polygons.
    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]
    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    try:
        center, w, h, theta, score = np.split(obboxes, (2, 3, 4, 5), axis=-1)
    except:  # noqa: E722
        results = np.stack([0., 0., 0., 0., 0., 0., 0., 0., 0.], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta), np.sin(theta)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([-h / 2 * Sin, h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center + vector1 - vector2
    point3 = center + vector1 + vector2
    point4 = center - vector1 + vector2
    polys = np.concatenate([point1, point2, point3, point4, score], axis=-1)
    polys = get_best_begin_point(polys)
    return polys


def cal_line_length(point1, point2):
    """Calculate the length of line.
    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]
    Returns:
        length (float)
    """
    return math.sqrt(
        math.pow(point1[0] - point2[0], 2) +
        math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.
    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4, score = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
               [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
               [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
               [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combine[i][0], dst_coordinate[0]) \
                     + cal_line_length(combine[i][1], dst_coordinate[1]) \
                     + cal_line_length(combine[i][2], dst_coordinate[2]) \
                     + cal_line_length(combine[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.hstack(
        (np.array(combine[force_flag]).reshape(8), np.array(score)))


def get_best_begin_point(coordinates):
    """Get the best begin points of polygons.
    Args:
        coordinate (ndarray): shape(n, 9).
    Returns:
        reorder coordinate (ndarray): shape(n, 9).
    """
    coordinates = list(map(get_best_begin_point_single, coordinates.tolist()))
    coordinates = np.array(coordinates)
    return coordinates


def norm_angle(angle, angle_range):
    """Limit the range of angles.
    Args:
        angle (ndarray): shape(n, ).
        angle_range (Str): angle representations.
    Returns:
        angle (ndarray): shape(n, ).
    """
    if angle_range == 'oc':
        return angle
    elif angle_range == 'le135':
        return (angle + np.pi / 4) % np.pi - np.pi / 4
    elif angle_range == 'le90':
        return (angle + np.pi / 2) % np.pi - np.pi / 2
    else:
        print('Not yet implemented.')


def dist_torch(point1, point2):
    """Calculate the distance between two points.
    Args:
        point1 (torch.Tensor): shape(n, 2).
        point2 (torch.Tensor): shape(n, 2).
    Returns:
        distance (torch.Tensor): shape(n, 1).
    """
    return torch.norm(point1 - point2, dim=-1)



def check_points_in_polys(points, polys):
    """Check whether point is in rotated boxes
    Args:
        points (tensor): (1, L, 2) anchor points
        polys (tensor): [B, N, 4, 2] gt_polys
        eps (float): default 1e-9
    Returns:
        is_in_polys (tensor): (B, N, L)
    """
    # [1, L, 2] -> [1, 1, L, 2]
    points = points.unsqueeze(0)
    # [B, N, 4, 2] -> [B, N, 1, 2]
    a, b, c, d = polys.split(4, axis=2)
    ab = b - a
    ad = d - a
    # [B, N, L, 2]
    ap = points - a
    # [B, N, 1]
    norm_ab = torch.sum(ab * ab, axis=-1)
    # [B, N, 1]
    norm_ad = torch.sum(ad * ad, axis=-1)
    # [B, N, L] dot product
    ap_dot_ab = torch.sum(ap * ab, axis=-1)
    # [B, N, L] dot product
    ap_dot_ad = torch.sum(ap * ad, axis=-1)
    # [B, N, L] <A, B> = |A|*|B|*cos(theta)
    is_in_polys = (
        (ap_dot_ab >= 0)
        & (ap_dot_ab <= norm_ab)
        & (ap_dot_ad >= 0)
        & (ap_dot_ad <= norm_ad)
    )
    return is_in_polys


def check_points_in_polys(points, polys):
    """Check whether point is in rotated boxes
    Args:
        points (tensor): (1, L, 2) anchor points
        polys (tensor): [B, N, 4, 2] gt_polys
        eps (float): default 1e-9
    Returns:
        is_in_polys (tensor): (B, N, L)
    """
    # [1, L, 2] -> [1, 1, L, 2]
    points = points.unsqueeze(0)
    # [B, N, 4, 2] -> [B, N, 1, 2]
    a, b, c, d = torch.split(points,[1,1,1,1],axis=2)
    ab = b - a
    ad = d - a
    # [B, N, L, 2]
    ap = points - a
    # [B, N, 1]
    norm_ab = torch.sum(ab * ab, axis=-1)
    # [B, N, 1]
    norm_ad = torch.sum(ad * ad, axis=-1)
    # [B, N, L] dot product
    ap_dot_ab = torch.sum(ap * ab, axis=-1)
    # [B, N, L] dot product
    ap_dot_ad = torch.sum(ap * ad, axis=-1)
    # [B, N, L] <A, B> = |A|*|B|*cos(theta)
    is_in_polys = (
        (ap_dot_ab >= 0)
        & (ap_dot_ab <= norm_ab)
        & (ap_dot_ad >= 0)
        & (ap_dot_ad <= norm_ad)
    )
    return is_in_polys


def check_points_in_rotated_boxes(points, boxes):
    """Check whether point is in rotated boxes
    Args:
        points (tensor): (1, L, 2) anchor points
        boxes (tensor): [B, N, 5] gt_bboxes
        eps (float): default 1e-9

    Returns:
        is_in_box (tensor): (B, N, L)
    """
    # [B, N, 5] -> [B, N, 4, 2]
    polys = obb2poly(boxes,version='oc')
    return check_points_in_polys(points, polys)

def rotated_iou_similarity(box1,box2):
    """Calculate rotate iou of box1 and box2
    Args:
        box1 (Tensor): box with the shape [N,M1,5] 
        box2 (Tensor): box with the shape [N,M2,5]
    Return:
        iou (Tensor): box between box1 and box2 with shape [N,M1,M2]
    """
    rotated_ious = []
    for b1, b2 in zip(box1, box2):
        rotated_ious.append(rotate_iou(b1, b2))
    return torch.stack(rotated_ious,axis=0)