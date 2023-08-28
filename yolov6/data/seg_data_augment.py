#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py

import math
import random

import cv2
import numpy as np


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    '''HSV color-space augmentation.'''
    if hgain or sgain or vgain:
        r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
        hue, sat, val = cv2.split(cv2.cvtColor(im, cv2.COLOR_BGR2HSV))
        dtype = im.dtype  # uint8

        x = np.arange(0, 256, dtype=r.dtype)
        lut_hue = ((x * r[0]) % 180).astype(dtype)
        lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
        lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

        im_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val)))
        cv2.cvtColor(im_hsv, cv2.COLOR_HSV2BGR, dst=im)  # no return needed


def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=True, stride=32):
    '''Resize and pad image while meeting stride-multiple constraints.'''
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    elif isinstance(new_shape, list) and len(new_shape) == 1:
       new_shape = (new_shape[0], new_shape[0])

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border

    return im, r, (left, top)


def mixup(im, labels, segments, im2, labels2, segments2):
    # Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf
    r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    segments = np.concatenate((segments, segments2), 0)
    return im, labels, segments


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    '''Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio.'''
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (ar < ar_thr)  # candidates


def random_affine(img, labels=(), segments=(), degrees=10, translate=.1, scale=.1, shear=10,
                  new_shape=(640, 640), task=""):
    '''Applies Random affine transformation.'''
    n = len(labels)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    height, width = new_shape
    # print(height, width, (height, width))

    M, s = get_transform_matrix(img.shape[:2], (height, width), degrees, scale, shear, translate)
    if (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    new_segments = []
    # Transform label coordinates
    if n:
        new = np.zeros((n, 4))
        segments = resample_segments(segments)
        for i, segment in enumerate(segments):
            xy = np.ones((len(segment), 3))
            xy[:, :2] = segment
            xy = xy @ M.T  # transform
            xy = (xy[:, :2])

            # clip
            new[i] = segment2box(xy, width, height)
            new_segments.append(xy)
        i = box_candidates(box1=labels[:, 1:5].T * s, box2=new.T, area_thr=0.01)
        if task!="val":
            labels = labels[i]
            labels[:, 1:5] = new[i]
            new_segments = np.array(new_segments)[i]
        else:
            labels[:, 1:5] = new
            new_segments = np.array(new_segments)
    return img, labels, new_segments

def copy_paste(im, labels, segments, p=0.5):
    # Implement Copy-Paste augmentation https://arxiv.org/abs/2012.07177, labels as nx5 np.array(cls, xyxy)
    n = len(segments)
    if p and n:
        h, w, c = im.shape  # height, width, channels
        im_new = np.zeros(im.shape, np.uint8)
        for j in random.sample(range(n), k=round(p * n)):
            l, s = labels[j], segments[j]
            box = w - l[3], l[2], w - l[1], l[4]
            ioa = bbox_ioa(box, labels[:, 1:5])  # intersection over area
            if (ioa < 0.30).all():  # allow 30% obscuration of existing labels
                labels = np.concatenate((labels, [[l[0], *box]]), 0)
                segments.append(np.concatenate((w - s[:, 0:1], s[:, 1:2]), 1))
                cv2.drawContours(im_new, [segments[j].astype(np.int32)], -1, (1, 1, 1), cv2.FILLED)
        result = cv2.flip(im, 1)  # augment segments (flip left-right)
        i = cv2.flip(im_new, 1).astype(bool)
        im[i] = result[i]  # cv2.imwrite('debug.jpg', im)  # debug

    return im, labels, segments

def bbox_ioa(box1, box2, eps=1e-7):
    """ Returns the intersection over box2 area given box1, box2. Boxes are x1y1x2y2
    box1:       np.array of shape(4)
    box2:       np.array of shape(nx4)
    returns:    np.array of shape(n)
    """

    # Get the coordinates of bounding boxes
    b1_x1, b1_y1, b1_x2, b1_y2 = box1
    b2_x1, b2_y1, b2_x2, b2_y2 = box2.T

    # Intersection area
    inter_area = (np.minimum(b1_x2, b2_x2) - np.maximum(b1_x1, b2_x1)).clip(0) * \
                 (np.minimum(b1_y2, b2_y2) - np.maximum(b1_y1, b2_y1)).clip(0)

    # box2 area
    box2_area = (b2_x2 - b2_x1) * (b2_y2 - b2_y1) + eps

    # Intersection over box2 area
    return inter_area / box2_area


def regen_labels(labels=None, segments=None, new_shape=(640, 640)):
    '''Applies Random affine transformation.'''
    n = len(segments)
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)
    height, width = new_shape

    new_segments = []
    # Transform label coordinates
    if n:
        new = np.zeros((n, 4))
        segments = resample_segments(segments)
        for i, segment in enumerate(segments):
            new[i] = segment2box(segment, width, height)
            new_segments.append(segment)
        labels[:, 1:5] = new[i]
        new_segments = np.array(new_segments)[i]

    return labels, new_segments

def resample_segments(segments, n=1000):
    # Up-sample an (n,2) segment
    for i, s in enumerate(segments):
        s = np.concatenate((s, s[0:1, :]), axis=0)
        x = np.linspace(0, len(s) - 1, n)
        xp = np.arange(len(s))
        segments[i] = np.concatenate([np.interp(x, xp, s[:, i]) for i in range(2)]).reshape(2, -1).T  # segment xy
    return segments


def get_transform_matrix(img_shape, new_shape, degrees, scale, shear, translate):
    new_height, new_width = new_shape
    # print(new_height, new_width)
    # Center
    C = np.eye(3)
    C[0, 2] = -img_shape[1] / 2  # x translation (pixels)
    C[1, 2] = -img_shape[0] / 2  # y translation (pixels)

    # Rotation and Scale
    R = np.eye(3)
    a = random.uniform(-degrees, degrees)
    # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
    s = random.uniform(1 - scale, 1 + scale)
    # s = 2 ** random.uniform(-scale, scale)
    R[:2] = cv2.getRotationMatrix2D(angle=a, center=(0, 0), scale=s)

    # Shear
    S = np.eye(3)
    S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
    S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

    # Translation
    T = np.eye(3)
    T[0, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_width  # x translation (pixels)
    T[1, 2] = random.uniform(0.5 - translate, 0.5 + translate) * new_height  # y transla ion (pixels)

    # Combined rotation matrix
    M = T @ S @ R @ C  # order of operations (right to left) is IMPORTANT
    return M, s


def mosaic_augmentation(shape, imgs, hs, ws, labels, segments, hyp, specific_shape = False, target_height=640, target_width=640):
    '''Applies Mosaic augmentation.'''
    assert len(imgs) == 4, "Mosaic augmentation of current version only supports 4 images."
    labels4 = []
    segments4 = []
    if not specific_shape:
        if isinstance(shape, list) or isinstance(shape, np.ndarray):
            target_height, target_width = shape
        else:
            target_height = target_width = shape

    yc, xc = (int(random.uniform(x//2, 3*x//2)) for x in (target_height, target_width) )  # mosaic center x, y

    for i in range(len(imgs)):
        # Load image
        img, h, w = imgs[i], hs[i], ws[i]
        # place img in img4
        if i == 0:  # top left
            img4 = np.full((target_height * 2, target_width * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles

            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, target_width * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(target_height * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, target_width * 2), min(target_height * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels_per_img = labels[i].copy()
        segments_per_img = segments[i].copy()
        if labels_per_img.size:
            boxes = np.copy(labels_per_img[:, 1:])
            boxes[:, 0] = w * (labels_per_img[:, 1] - labels_per_img[:, 3] / 2) + padw  # top left x
            boxes[:, 1] = h * (labels_per_img[:, 2] - labels_per_img[:, 4] / 2) + padh  # top left y
            boxes[:, 2] = w * (labels_per_img[:, 1] + labels_per_img[:, 3] / 2) + padw  # bottom right x
            boxes[:, 3] = h * (labels_per_img[:, 2] + labels_per_img[:, 4] / 2) + padh  # bottom right y
            for __ in range(len(segments_per_img)):
                segments_per_img[__][:, 0] = w * segments_per_img[__][:, 0] + padw
                segments_per_img[__][:, 1] = h * segments_per_img[__][:, 1] + padh
            labels_per_img[:, 1:] = boxes

        labels4.append(labels_per_img)
        segments4.extend(segments_per_img)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    # for x in (labels4[:, 1:]):
    #     np.clip(x, 0, 2 * s, out=x)
    labels4[:, 1::2] = np.clip(labels4[:, 1::2], 0, 2 * target_width)
    labels4[:, 2::2] = np.clip(labels4[:, 2::2], 0, 2 * target_height)
    for __ in range(len(segments4)):
        segments4[__][:, 0] = np.clip(segments4[__][:, 0], 0, 2 * target_width)
        segments4[__][:, 1] = np.clip(segments4[__][:, 1], 0, 2 * target_height)

    # Augment
    return img4, labels4, segments4
    img4, labels4, segments4 = random_affine(img4, labels4, segments4,
                                  degrees=hyp['degrees'],
                                  translate=hyp['translate'],
                                  scale=hyp['scale'],
                                  shear=hyp['shear'],
                                  new_shape=(target_height, target_width))

    return img4, labels4, segments4

def segment2box(segment, width=640, height=640):
    # Convert 1 segment label to 1 box label, applying inside-image constraint, i.e. (xy1, xy2, ...) to (xyxy)
    x, y = segment.T  # segment xy
    inside = (x >= 0) & (y >= 0) & (x <= width) & (y <= height)
    x, y, = x[inside], y[inside]
    return np.array([x.min(), y.min(), x.max(), y.max()]) if any(x) else np.zeros((1, 4))  # xyxy

