#!/usr/bin/env python3
# -*- coding:utf-8 -*-
# This code is based on
# https://github.com/ultralytics/yolov5/blob/master/utils/dataloaders.py

import math
import random

import cv2
import numpy as np
import torch
from mmcv.ops import box_iou_rotated


def minAreaRect2longSideFormat(rectangle_inf):
    width = rectangle_inf[1][0]
    height = rectangle_inf[1][1]
    theta = rectangle_inf[-1]
    longSide = max(width, height)
    shortSide = min(width, height)
    if theta == 90:
        if longSide == width:
            pass
        else:
            theta = 0
        # * 正方形
        if np.around(longSide, 2) == np.around(shortSide, 2):
            theta = 0
    else:
        # * 正四边形 minAreaRect 会直接判定为左侧补角，符合要求 做长度截断判断即可
        if np.around(longSide, 2) == np.around(shortSide, 2):
            pass
        else:
            if longSide == width:
                pass
            else:
                theta += 90

    if 179 < theta < 180:
        # theta = 179
        theta = 179
    if theta < 0 or theta >= 180:
        raise ValueError("theta < 0 or theta >= 180")

    return (rectangle_inf[0], (longSide, shortSide), theta)


def longSideFormat2minAreaRect(longSide_inf):
    longSide = longSide_inf[1][0]
    shortSide = longSide_inf[1][1]
    theta = longSide_inf[-1]
    width = longSide
    height = shortSide
    if theta == 0:
        width = shortSide
        height = longSide
        theta = 90
    # ! 有概率非全部 不影响绘图
    # elif theta == 90:
    #     width = longSide
    #     height = shortSide
    #     theta = 0
    else:
        # * 正四边形
        if np.around(longSide, 2) == np.around(shortSide, 2):
            width = longSide
            height = shortSide
            pass
        if theta > 90:
            width = shortSide
            height = longSide
            theta -= 90
        else:
            pass

    if theta >= 180:
        raise ValueError("theta >= 180")

    return (longSide_inf[0], (width, height), theta)


def plot_single_obb_img_test(img, labels):
    # plot validation predictions
    # TODO test

    Color = [tuple(np.random.choice(range(256), size=3)) for _ in range(20)]

    for vis_bbox in labels.tolist():
        cls_id = int(vis_bbox[0])
        cx = int(vis_bbox[1])
        cy = int(vis_bbox[2])
        w = int(vis_bbox[3])
        h = int(vis_bbox[4])
        angle = int(vis_bbox[5])
        rect = ((cx, cy), (w, h), angle)
        poly = cv2.boxPoints(longSideFormat2minAreaRect(rect))
        poly = np.int0(poly)
        cv2.drawContours(
            img,
            contours=[poly],
            contourIdx=-1,
            color=tuple([int(x) for x in Color[cls_id]]),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        # cv2.putText(
        #     img,
        #     f"{self.data_dict['names'][cls_id]}: {box_score:.2f}",
        #     (cx - 5, cy - 5),
        #     cv2.FONT_HERSHEY_COMPLEX,
        #     0.5,
        #     tuple([int(x) for x in Color[cls_id]]),
        #     thickness=2,
        # )
    return img


def plot_single_poly_img_test(img, labels):
    # plot validation predictions
    # TODO test

    Color = [tuple(np.random.choice(range(256), size=3)) for _ in range(20)]

    for poly in labels.tolist():
        poly = np.int0(poly)
        poly = poly.reshape(-1, 2)
        cv2.drawContours(
            img,
            contours=[poly],
            contourIdx=-1,
            color=tuple([int(x) for x in Color[0]]),
            thickness=2,
            lineType=cv2.LINE_AA,
        )
        # cv2.putText(
        #     img,
        #     f"{self.data_dict['names'][cls_id]}: {box_score:.2f}",
        #     (cx - 5, cy - 5),
        #     cv2.FONT_HERSHEY_COMPLEX,
        #     0.5,
        #     tuple([int(x) for x in Color[cls_id]]),
        #     thickness=2,
        # )
    return img


def cal_line_length(point1, point2):
    """Calculate the length of line.

    Args:
        point1 (List): [x,y]
        point2 (List): [x,y]

    Returns:
        length (float)
    """
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))


def get_best_begin_point_single(coordinate):
    """Get the best begin point of the single polygon.

    Args:
        coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]

    Returns:
        reorder coordinate (List): [x1, y1, x2, y2, x3, y3, x4, y4, score]
    """
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combine = [
        [[x1, y1], [x2, y2], [x3, y3], [x4, y4]],
        [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
        [[x3, y3], [x4, y4], [x1, y1], [x2, y2]],
        [[x4, y4], [x1, y1], [x2, y2], [x3, y3]],
    ]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = (
            cal_line_length(combine[i][0], dst_coordinate[0])
            + cal_line_length(combine[i][1], dst_coordinate[1])
            + cal_line_length(combine[i][2], dst_coordinate[2])
            + cal_line_length(combine[i][3], dst_coordinate[3])
        )
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
    return np.array(combine[force_flag]).reshape(8)


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


def obb2poly_bp_le180(obboxes):
    """Convert oriented bounding boxes to polygons.

    Args:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle,score]

    Returns:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3,score]
    """
    try:
        center, w, h, theta = np.split(obboxes, (2, 3, 4), axis=-1)
    except:  # noqa: E722
        results = np.stack([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0], axis=-1)
        return results.reshape(1, -1)
    Cos, Sin = np.cos(theta * np.pi / 180.0), np.sin(theta * np.pi / 180.0)
    vector1 = np.concatenate([w / 2 * Cos, w / 2 * Sin], axis=-1)
    vector2 = np.concatenate([h / 2 * Sin, -h / 2 * Cos], axis=-1)
    point1 = center - vector1 - vector2
    point2 = center - vector1 + vector2
    point3 = center + vector1 + vector2
    point4 = center + vector1 - vector2
    polys = np.concatenate([point1, point2, point3, point4], axis=-1)
    polys = get_best_begin_point(polys)
    return polys


def poly2obb_np_le180(poly):
    """Convert polygons to oriented bounding boxes.

    Args:
        polys (ndarray): [x0,y0,x1,y1,x2,y2,x3,y3]

    Returns:
        obbs (ndarray): [x_ctr,y_ctr,w,h,angle]
    """
    bboxps = np.array(poly).reshape((4, 2))
    rbbox = cv2.minAreaRect(np.int0(bboxps))
    x, y, width, height, theta = rbbox[0][0], rbbox[0][1], rbbox[1][0], rbbox[1][1], rbbox[2]
    longSide = max(width, height)
    shortSide = min(width, height)
    if width < 2 or height < 2:
        return

    if theta == 90:
        if longSide == width:
            pass
        else:
            theta = 0
        # * 正方形
        if np.around(longSide, 2) == np.around(shortSide, 2):
            theta = 0
    else:
        # * 正四边形 minAreaRect 会直接判定为左侧补角，符合要求 做长度截断判断即可
        if np.around(longSide, 2) == np.around(shortSide, 2):
            pass
        else:
            if longSide == width:
                pass
            else:
                theta += 90

    if 179 < theta <= 180:
        # theta = 179
        ...
    if theta < 0 or theta >= 180:
        raise ValueError("theta < 0 or theta >= 180")

    return x, y, longSide, shortSide, theta


class PolyRandomRotate(object):
    """Rotate img & bbox.
    Reference: https://github.com/hukaixuan19970627/OrientedRepPoints_DOTA

    Args:
        rotate_ratio (float, optional): The rotating probability.
            Default: 0.5.
        mode (str, optional) : Indicates whether the angle is chosen in a
            random range (mode='range') or in a preset list of angles
            (mode='value'). Defaults to 'range'.
        angles_range(int|list[int], optional): The range of angles.
            If mode='range', angle_ranges is an int and the angle is chosen
            in (-angles_range, +angles_ranges).
            If mode='value', angles_range is a non-empty list of int and the
            angle is chosen in angles_range.
            Defaults to 180 as default mode is 'range'.
        auto_bound(bool, optional): whether to find the new width and height
            bounds.
        rect_classes (None|list, optional): Specifies classes that needs to
            be rotated by a multiple of 90 degrees.
        allow_negative (bool, optional): Whether to allow an image that does
            not contain any bbox area. Default False.
        version  (str, optional): Angle representations. Defaults to 'le90'.
    """

    def __init__(
        self,
        rotate_ratio=0.5,
        mode="range",
        angles_range=180,
        auto_bound=False,
        rect_classes=None,
        allow_negative=False,
        version="le90",
    ):
        self.rotate_ratio = rotate_ratio
        self.auto_bound = auto_bound
        assert mode in ["range", "value"], f"mode is supposed to be 'range' or 'value', but got {mode}."
        if mode == "range":
            assert isinstance(angles_range, int), "mode 'range' expects angle_range to be an int."
        else:
            assert mmcv.is_seq_of(angles_range, int) and len(
                angles_range
            ), "mode 'value' expects angle_range as a non-empty list of int."
        self.mode = mode
        self.angles_range = angles_range
        self.discrete_range = [90, 180, -90, -180]
        self.rect_classes = rect_classes
        self.allow_negative = allow_negative
        self.version = version

    @property
    def is_rotate(self):
        """Randomly decide whether to rotate."""
        return np.random.rand() < self.rotate_ratio

    def apply_image(self, img, bound_h, bound_w, interp=cv2.INTER_LINEAR):
        """
        img should be a numpy array, formatted as Height * Width * Nchannels
        """
        if len(img) == 0:
            return img
        return cv2.warpAffine(img, self.rm_image, (bound_w, bound_h), flags=interp)

    def apply_coords(self, coords):
        """
        coords should be a N * 2 array-like, containing N couples of (x, y)
        points
        """
        if len(coords) == 0:
            return coords
        coords = np.asarray(coords, dtype=float)
        return cv2.transform(coords[:, np.newaxis, :], self.rm_coords)[:, 0, :]

    def create_rotation_matrix(self, center, angle, bound_h, bound_w, offset=0):
        """Create rotation matrix."""
        center += offset
        rm = cv2.getRotationMatrix2D(tuple(center), angle, 1)
        if self.auto_bound:
            rot_im_center = cv2.transform(center[None, None, :] + offset, rm)[0, 0, :]
            new_center = np.array([bound_w / 2, bound_h / 2]) + offset - rot_im_center
            rm[:, 2] += new_center
        return rm

    def filter_border(self, bboxes, h, w):
        """Filter the box whose center point is outside or whose side length is
        less than 5."""
        x_ctr, y_ctr = bboxes[:, 0], bboxes[:, 1]
        w_bbox, h_bbox = bboxes[:, 2], bboxes[:, 3]
        keep_inds = (x_ctr > 0) & (x_ctr < w) & (y_ctr > 0) & (y_ctr < h) & (w_bbox > 5) & (h_bbox > 5)
        return keep_inds

    def __call__(self, img, labels):
        """Call function of PolyRandomRotate."""
        if not self.is_rotate:
            # results["rotate"] = False
            angle = 0
        else:
            # results["rotate"] = True
            if self.mode == "range":
                angle = self.angles_range * (2 * np.random.rand() - 1)
            else:
                i = np.random.randint(len(self.angles_range))
                angle = self.angles_range[i]

            class_labels = labels[..., 0:1]
            for classid in class_labels:
                if self.rect_classes:
                    if classid in self.rect_classes:
                        np.random.shuffle(self.discrete_range)
                        angle = self.discrete_range[0]
                        break

        # h, w, c = results["img_shape"]
        h, w, c = img.shape
        # img = results["img"]
        # results["rotate_angle"] = angle

        image_center = np.array((w / 2, h / 2))
        abs_cos, abs_sin = abs(np.cos(angle / 180 * np.pi)), abs(np.sin(angle / 180 * np.pi))
        if self.auto_bound:
            bound_w, bound_h = np.rint([h * abs_sin + w * abs_cos, h * abs_cos + w * abs_sin]).astype(int)
        else:
            bound_w, bound_h = w, h

        self.rm_coords = self.create_rotation_matrix(image_center, angle, bound_h, bound_w)
        self.rm_image = self.create_rotation_matrix(image_center, angle, bound_h, bound_w, offset=-0.5)

        ori_img = img.copy()
        img = self.apply_image(img, bound_h, bound_w)
        # results["img"] = img
        # results["img_shape"] = (bound_h, bound_w, c)
        # gt_bboxes = results.get("gt_bboxes", [])
        # labels = results.get("gt_labels", [])
        gt_bboxes = labels[..., 1:]

        if len(gt_bboxes):
            # gt_bboxes = np.concatenate([gt_bboxes, np.zeros((gt_bboxes.shape[0], 1))], axis=-1)
            polys = obb2poly_bp_le180(gt_bboxes).reshape(-1, 2)
            polys = self.apply_coords(polys).reshape(-1, 8)
            gt_bboxes = []
            for pt in polys:
                pt = np.array(pt, dtype=np.float32)
                obb = poly2obb_np_le180(pt) if poly2obb_np_le180(pt) is not None else [0, 0, 0, 0, 0]
                gt_bboxes.append(obb)
            gt_bboxes = np.array(gt_bboxes, dtype=np.float32)
            keep_inds = self.filter_border(gt_bboxes, bound_h, bound_w)
            gt_bboxes = gt_bboxes[keep_inds, :]
            class_labels = class_labels[keep_inds]
        if len(gt_bboxes) == 0 and not self.allow_negative:
            return img, np.zeros((1, 6))
        # results["gt_bboxes"] = gt_bboxes
        # results["gt_labels"] = labels

        labels = np.concatenate((class_labels, gt_bboxes), axis=-1)

        return img, labels

    def __repr__(self):
        repr_str = self.__class__.__name__
        repr_str += (
            f"(rotate_ratio={self.rotate_ratio}, "
            f"base_angles={self.base_angles}, "
            f"angles_range={self.angles_range}, "
            f"auto_bound={self.auto_bound})"
        )
        return repr_str


def RFlipVertical(img: np.ndarray, bboxes: np.ndarray):
    """
    # NOTE old [xmin, ymin xmax, ymax, x_center, y_center, w, h, class_id, or class_id_2 angle]
    # NOTE i   [ 0     1    2      3     4        5        6  7     8                     9or-1]
    # NOTE new [class_id, x_center, y_center, longSide, shortSide, angle]
    # NOTE i   [    0         1         2     3             4        5  ]
    # NOTE cv2: HWC
    """
    h = img.shape[0]
    img = img[::-1, :, :]
    bboxes[:, 2] = h - bboxes[:, 2]
    # * angle = 180 - angle
    bboxes[:, -1] = 180 - bboxes[:, -1]
    # * angle 180° 无定义 转到 0°
    bboxes[bboxes[:, -1] == 180, -1] = 0

    return img, bboxes


def RFlipHorizontal(img, bboxes):
    """
    # NOTE old [xmin, ymin xmax, ymax, x_center, y_center, w, h, class_id, or class_id_2 angle]
    # NOTE i   [ 0     1    2      3     4        5        6  7     8                     9or-1]
    # NOTE new [class_id, x_center, y_center, longSide, shortSide, angle]
    # NOTE i   [    0         1         2     3             4        5  ]
    # NOTE cv2: HWC
    """
    w = img.shape[1]
    img = img[:, ::-1, :]
    # * x' = w - x; y' = y
    bboxes[:, 1] = w - bboxes[:, 1]
    # * angle = 180 - angle
    bboxes[:, -1] = 180 - bboxes[:, -1]
    # * angle 180° 无定义 转到 0°
    bboxes[bboxes[:, -1] == 180, -1] = 0
    return img, bboxes


def RRotate(img, bboxes):
    # NOTE new [xmin, ymin xmax, ymax, x_center, y_center, w, h, class_id, or class_id_2 angle]
    # NOTE i   [ 0     1    2      3     4        5        6  7     8                     9or-1]
    # NOTE new [class_id, x_center, y_center, longSide, shortSide, angle]
    # NOTE i   [    0         1         2     3             4        5  ]
    # NOTE cv2: HWC
    # NOTE 0°， 90°， 180°， 270°
    degree = 90
    pn = np.random.randint(low=0, high=4, size=None, dtype="l")
    # pn = 3
    h, w = img.shape[:2]
    M = np.eye(3)
    M[:2] = cv2.getRotationMatrix2D(angle=degree * pn, center=(w / 2, h / 2), scale=1.0)

    img = cv2.warpAffine(img, M[:2], dsize=(w, h), borderValue=(128, 128, 128))

    # Transform label coordinates
    n = len(bboxes)
    # warp points
    xy = np.ones((n, 3))

    # x_center y_center
    xy[:, :2] = bboxes[:, [1, 2]].reshape(n, 2)
    xy = xy @ M.T  # transform
    xy = xy[:, :2].reshape(n, 2)

    # create new boxes
    x = xy[:, 0:1].clip(0, w)
    y = xy[:, 1:2].clip(0, h)

    bboxes[:, 1] = x[:, -1]
    bboxes[:, 2] = y[:, -1]
    # if pn == 0:
    #     bboxes[:, [5, 7, 9, 11]] = x
    #     bboxes[:, [6, 8, 10, 12]] = y
    # elif pn == 1:
    #     bboxes[:, [11, 5, 7, 9]] = x
    #     bboxes[:, [12, 6, 8, 10]] = y
    # elif pn == 2:
    #     bboxes[:, [9, 11, 5, 7]] = x
    #     bboxes[:, [10, 12, 6, 8]] = y
    # elif pn == 3:
    #     bboxes[:, [7, 9, 11, 5]] = x
    #     bboxes[:, [8, 10, 12, 6]] = y

    bboxes[:, -1] = bboxes[:, -1] + pn * 90

    bboxes[:, -1] = np.where(bboxes[:, -1] > 180, bboxes[:, -1] - 180, bboxes[:, -1])
    bboxes[:, -1] = np.where(bboxes[:, -1] > 180, bboxes[:, -1] - 180, bboxes[:, -1])

    # * angle 180° 无定义 转到 0°
    bboxes[bboxes[:, -1] == 180, -1] = 0

    # xy4 = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T
    # bboxes[:, [0, 1, 2, 3]] = xy4

    # for anno in bboxes:
    #     points = np.array(
    #         [[int(anno[5]), int(anno[6])], [int(anno[7]), int(anno[8])], [int(anno[9]), int(anno[10])],
    #          [int(anno[11]), int(anno[12])]])
    #     cv2.polylines(new_img, [points], 1, (0, 128, 255), 2)
    # import matplotlib.pyplot as plt
    # plt.figure("Image_Rot")  # 图像窗口名称
    # plt.imshow(new_img / 255, cmap='jet')
    # plt.show()
    return img, bboxes


def augment_hsv(im, hgain=0.5, sgain=0.5, vgain=0.5):
    """HSV color-space augmentation."""
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


def letterbox(im, new_shape=(640, 640), color=(144, 144, 144), auto=True, scaleup=True, stride=32, return_int=False):
    """Resize and pad image while meeting stride-multiple constraints."""
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
    if not return_int:
        return im, r, (dw, dh)
    else:
        return im, r, (left, top)


def mixup(im, labels, im2, labels2):
    """Applies MixUp augmentation https://arxiv.org/pdf/1710.09412.pdf."""
    # r = np.random.beta(32.0, 32.0)  # mixup ratio, alpha=beta=32.0
    r = 0.5
    im = (im * r + im2 * (1 - r)).astype(np.uint8)
    labels = np.concatenate((labels, labels2), 0)
    return im, labels


def box_candidates(box1, box2, wh_thr=2, ar_thr=20, area_thr=0.1, eps=1e-16):  # box1(4,n), box2(4,n)
    """Compute candidate boxes: box1 before augment, box2 after augment, wh_thr (pixels), aspect_ratio_thr, area_ratio."""
    w1, h1 = box1[2] - box1[0], box1[3] - box1[1]
    w2, h2 = box2[2] - box2[0], box2[3] - box2[1]
    ar = np.maximum(w2 / (h2 + eps), h2 / (w2 + eps))  # aspect ratio
    return (w2 > wh_thr) & (h2 > wh_thr) & (w2 * h2 / (w1 * h1 + eps) > area_thr) & (ar < ar_thr)  # candidates


def random_affine(img, labels=(), degrees=10, translate=0.1, scale=0.1, shear=10, new_shape=(640, 640)):
    """Applies Random affine transformation."""
    n = len(labels)
    height, width = new_shape

    M, s = get_transform_matrix(img.shape[:2], (height, width), degrees, scale, shear, translate)
    if (M != np.eye(3)).any():  # image changed
        img = cv2.warpAffine(img, M[:2], dsize=(width, height), borderValue=(114, 114, 114))

    # Transform label coordinates
    if n:
        new = np.zeros((n, 4))

        xy = np.ones((n * 4, 3))
        xy[:, :2] = labels[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
        xy = xy @ M.T  # transform
        xy = xy[:, :2].reshape(n, 8)  # perspective rescale or affine

        # create new boxes
        x = xy[:, [0, 2, 4, 6]]
        y = xy[:, [1, 3, 5, 7]]
        new = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

        # clip
        new[:, [0, 2]] = new[:, [0, 2]].clip(0, width)
        new[:, [1, 3]] = new[:, [1, 3]].clip(0, height)

        # filter candidates
        i = box_candidates(box1=labels[:, 1:5].T * s, box2=new.T, area_thr=0.1)
        labels = labels[i]
        labels[:, 1:5] = new[i]

    return img, labels


def get_transform_matrix(img_shape, new_shape, degrees, scale, shear, translate):
    new_height, new_width = new_shape
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


def mosaic_augmentation(img_size, imgs, hs, ws, labels, hyp):
    """Applies Mosaic augmentation."""
    assert len(imgs) == 4, "Mosaic augmentation of current version only supports 4 images."

    labels4 = []
    s = img_size
    yc, xc = (int(random.uniform(s // 2, 3 * s // 2)) for _ in range(2))  # mosaic center x, y
    for i in range(len(imgs)):
        # Load image
        img, h, w = imgs[i], hs[i], ws[i]
        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels_per_img = labels[i].copy()
        if labels_per_img.size:
            boxes = np.copy(labels_per_img[:, 1:])
            boxes[:, 0] = w * (labels_per_img[:, 1] - labels_per_img[:, 3] / 2) + padw  # top left x
            boxes[:, 1] = h * (labels_per_img[:, 2] - labels_per_img[:, 4] / 2) + padh  # top left y
            boxes[:, 2] = w * (labels_per_img[:, 1] + labels_per_img[:, 3] / 2) + padw  # bottom right x
            boxes[:, 3] = h * (labels_per_img[:, 2] + labels_per_img[:, 4] / 2) + padh  # bottom right y
            labels_per_img[:, 1:] = boxes

        labels4.append(labels_per_img)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)
    for x in labels4[:, 1:]:
        np.clip(x, 0, 2 * s, out=x)

    # Augment
    img4, labels4 = random_affine(
        img4,
        labels4,
        degrees=hyp["degrees"],
        translate=hyp["translate"],
        scale=hyp["scale"],
        shear=hyp["shear"],
        new_shape=(img_size, img_size),
    )

    return img4, labels4


def mosaic_augmentation_obb(img_size, imgs, hs, ws, labels, hyp):
    """Applies Mosaic augmentation."""
    # NOTE mosaic 对遥感场景尺度变化影响很大, 需要修改
    assert len(imgs) == 4, "Mosaic augmentation of current version only supports 4 images."

    labels4 = []
    s = img_size // 2
    yc, xc = (int(random.uniform(s // 2, 3 * s // 2)) for _ in range(2))  # mosaic center x, y
    for i in range(len(imgs)):
        # Load image
        img, h, w = imgs[i], hs[i], ws[i]
        # place img in img4
        if i == 0:  # top left
            img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
            x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
        elif i == 1:  # top right
            x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
            x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
        elif i == 2:  # bottom left
            x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, w, min(y2a - y1a, h)
        elif i == 3:  # bottom right
            x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
            x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

        img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
        padw = x1a - x1b
        padh = y1a - y1b

        # Labels
        labels_per_img = labels[i].copy()
        if labels_per_img.size:
            # NOTE 输出全部转变为真实值, [x, y, w, h, angle]
            # x [padw, padw + w]
            # y [padh, padh + h]
            boxes = np.copy(labels_per_img[:, 1:])
            boxes[:, 0] = w * boxes[:, 0] + padw  # center x
            boxes[:, 1] = h * boxes[:, 1] + padh  # center y
            boxes[:, 2] = w * boxes[:, 2]  # longSide / w
            boxes[:, 3] = h * boxes[:, 3]  # shortSide / h
            # NOTE filter
            valid_inds = filter_box_candidates(boxes, x1a, x2a, y1a, y2a, min_bbox_size=2)
            labels_per_img[:, 1:] = boxes
            labels_per_img = labels_per_img[valid_inds]

        labels4.append(labels_per_img)

    # Concat/clip labels
    labels4 = np.concatenate(labels4, 0)

    # NOTE 不做affine,一个是label不好调整, 另一个参考mmyolo的RTM, affine会造成影响
    # NOTE img5, labels4 需要重新resize
    # img4, labels4 = random_affine(img4, labels4,
    #                               degrees=hyp['degrees'],
    #                               translate=hyp['translate'],
    #                               scale=hyp['scale'],
    #                               shear=hyp['shear'],
    #                               new_shape=(img_size, img_size))

    img4 = cv2.resize(img4, (img_size, img_size))
    # labels4[:, 1:5] /= 2.0

    return img4, labels4


def filter_box_candidates(bboxes, w_min, w_max, h_min, h_max, min_bbox_size=2, ratio=0.1):
    """Filter out small bboxes and outside bboxes after Mosaic."""
    bbox_x, bbox_y, bbox_w, bbox_h = bboxes[:, 0], bboxes[:, 1], bboxes[:, 2], bboxes[:, 3]
    # TODO 截断resize情况,比较复杂, 考虑中心点到边界距离
    # 比例截断系数, 0.5 * 0.3
    ratio *= 0.5
    valid_inds = (
        (bbox_x > w_min)
        & (bbox_x < w_max)
        & (bbox_y > h_min)
        & (bbox_y < h_max)
        & ((bbox_x + ratio * bbox_w) < w_max)
        & ((bbox_x - ratio * bbox_w) > w_min)
        & ((bbox_y + ratio * bbox_h) < h_max)
        & ((bbox_y - ratio * bbox_h) > h_min)
        & (bbox_w > min_bbox_size)
        & (bbox_h > min_bbox_size)
    )
    valid_inds = np.nonzero(valid_inds)[0]
    return valid_inds
