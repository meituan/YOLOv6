#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import glob
import torch
import requests
from pathlib import Path
from yolov6.utils.events import LOGGER


def increment_name(path):
    """increase save directory's id"""
    path = Path(path)
    sep = ""
    if path.exists():
        path, suffix = (path.with_suffix(""), path.suffix) if path.is_file() else (path, "")
        for n in range(1, 9999):
            p = f"{path}{sep}{n}{suffix}"
            if not os.path.exists(p):
                break
        path = Path(p)
    return path


def find_latest_checkpoint(search_dir="."):
    """Find the most recent saved checkpoint in search_dir."""
    checkpoint_list = glob.glob(f"{search_dir}/**/last*.pt", recursive=True)
    return max(checkpoint_list, key=os.path.getctime) if checkpoint_list else ""


def dist2bbox(distance, anchor_points, box_format="xyxy"):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == "xyxy":
        bbox = torch.cat([x1y1, x2y2], -1)
    elif box_format == "xywh":
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = torch.cat([c_xy, wh], -1)
    return bbox


def bbox2dist(anchor_points, bbox, reg_max):
    """Transform bbox(xyxy) to dist(ltrb)."""
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = torch.cat([lt, rb], -1).clip(0, reg_max - 0.01)
    return dist


def dist2Rbbox(distance, angle, anchor_points, box_format="xywh"):
    """Transform distance(ltrb) to box(xywh or xyxy)."""
    cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
    rot_matrix = torch.cat([cos_angle, -sin_angle, sin_angle, cos_angle],
                           dim=-1)
    rot_matrix = rot_matrix.reshape(*rot_matrix.shape[:-1], 2, 2)
    wh = distance[..., :2] + distance[..., 2:]
    offset_t = (distance[..., 2:] - distance[..., :2]) / 2
    offset = torch.matmul(rot_matrix, offset_t[..., None]).squeeze(-1)
    ctr = anchor_points[..., :2] + offset
    if box_format == "xywh":
        bbox = torch.cat([ctr, wh], -1)
    # NOTE 可能有bug
    elif box_format == "xyxy":
        x1y1 = ctr - distance[..., :2]
        x2y2 = ctr + distance[..., 2:]
        bbox = torch.cat([x1y1, x2y2], -1)
    return bbox




def Rbbox2dist(anchor_points, bbox, angle, reg_max, eps=0.01):
    """Transform bbox(xywh) to dist(ltrb)."""
    ctr, wh = torch.split(bbox, [2, 2], dim=-1)

    cos_angle, sin_angle = torch.cos(angle), torch.sin(angle)
    rot_matrix = torch.cat([cos_angle, sin_angle, -sin_angle, cos_angle],
                           dim=-1)
    rot_matrix = rot_matrix.reshape(*rot_matrix.shape[:-1], 2, 2)

    offset = anchor_points - ctr
    offset = torch.matmul(rot_matrix, offset[..., None])
    offset = offset.squeeze(-1)

    w, h = wh[..., 0], wh[..., 1]
    offset_x, offset_y = offset[..., 0], offset[..., 1]
    left = w / 2 + offset_x
    right = w / 2 - offset_x
    top = h / 2 + offset_y
    bottom = h / 2 - offset_y
    if reg_max is not None:
        left = left.clamp(min=0, max=reg_max - eps)
        top = top.clamp(min=0, max=reg_max - eps)
        right = right.clamp(min=0, max=reg_max - eps)
        bottom = bottom.clamp(min=0, max=reg_max - eps)
    return torch.stack((left, top, right, bottom), -1)


def xywh2xyxy(bboxes):
    """Transform bbox(xywh) to box(xyxy)."""
    bboxes = bboxes.clone()
    bboxes[..., 0] = bboxes[..., 0] - bboxes[..., 2] * 0.5
    bboxes[..., 1] = bboxes[..., 1] - bboxes[..., 3] * 0.5
    bboxes[..., 2] = bboxes[..., 0] + bboxes[..., 2]
    bboxes[..., 3] = bboxes[..., 1] + bboxes[..., 3]
    return bboxes


def box_iou(box1, box2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Arguments:
        box1 (Tensor[N, 4])
        box2 (Tensor[M, 4])
    Returns:
        iou (Tensor[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(box1.T)
    area2 = box_area(box2.T)

    # inter(N,M) = (rb(N,M,2) - lt(N,M,2)).clamp(0).prod(2)
    inter = (torch.min(box1[:, None, 2:], box2[:, 2:]) - torch.max(box1[:, None, :2], box2[:, :2])).clamp(0).prod(2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def download_ckpt(path):
    """Download checkpoints of the pretrained models"""
    basename = os.path.basename(path)
    dir = os.path.abspath(os.path.dirname(path))
    os.makedirs(dir, exist_ok=True)
    LOGGER.info(f"checkpoint {basename} not exist, try to downloaded it from github.")
    # need to update the link with every release
    url = f"https://github.com/meituan/YOLOv6/releases/download/0.3.0/{basename}"
    r = requests.get(url, allow_redirects=True)
    assert r.status_code == 200, "Unable to download checkpoints, manually download it"
    open(path, "wb").write(r.content)
    LOGGER.info(f"checkpoint {basename} downloaded and saved")
