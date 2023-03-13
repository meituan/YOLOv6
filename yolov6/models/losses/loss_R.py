#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.assigners.atss_assigner_R import ATSSAssigner
from yolov6.assigners.tal_assigner_R import TaskAlignedAssigner
from yolov6.utils.figure_iou import IOUloss
from yolov6.utils.general import bbox2dist, box_iou, dist2bbox, xywh2xyxy


class ComputeLoss:
    """Loss computation func."""

    def __init__(
        self,
        fpn_strides=[8, 16, 32],
        grid_cell_size=5.0,
        grid_cell_offset=0.5,
        num_classes=80,
        ori_img_size=640,
        warmup_epoch=4,
        use_dfl=True,
        reg_max=16,
        angle_max=180,
        angle_fitting_methods="regression",
        iou_type="giou",
        loss_weight={"class": 1.0, "iou": 2.5, "dfl": 0.5, "angle": 0.01},
    ):

        self.fpn_strides = fpn_strides  # NOTE [8, 16, 32]
        self.grid_cell_size = grid_cell_size  # NOTE 5
        self.grid_cell_offset = grid_cell_offset  # NOTE 0.5
        self.num_classes = num_classes
        self.ori_img_size = ori_img_size

        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(
            9, num_classes=self.num_classes, angle_max=angle_max, angle_fitting_methods=angle_fitting_methods
        )  # NOTE ATSS
        self.formal_assigner = TaskAlignedAssigner(
            topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0
        )  # NOTE TAL

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.angle_max = angle_max
        self.angle_fitting_methods = angle_fitting_methods

        self.proj = nn.Parameter(
            torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False
        )  # NOTE 投影矩阵 dfl 乘矩阵
        if self.angle_fitting_methods == "dfl":
            self.proj_angle = (180.0 / self.angle_max) * nn.Parameter(
                torch.linspace(0, self.angle_max, self.angle_max + 1), requires_grad=False
            )  # NOTE 投影矩阵 dfl 乘矩阵

        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type).cuda()
        self.angle_loss = AngleLoss(self.angle_max, self.angle_fitting_methods).cuda()
        self.loss_weight = loss_weight

    def __call__(self, outputs, targets, epoch_num, step_num):

        # NOTE pred_distri 相对值 [bs, 8400, 4]
        # NOTE pred_angles [bs, 8400, angle_max]
        feats, pred_scores, pred_distri, pred_angles = outputs
        # NOTE 需要角度么?
        anchors, anchor_points, n_anchors_list, stride_tensor = generate_anchors(
            feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device
        )

        assert pred_scores.type() == pred_distri.type()
        gt_bboxes_scale = torch.full((1, 4), self.ori_img_size).type_as(pred_scores)
        batch_size = pred_scores.shape[0]

        # NOTE targets 绝对值坐标 [bs, max_len, 6] [class_id, x, y, x, y, angle]
        targets = self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]  # NOTE [bs, MAX_Num_labels_per_img, 1]
        gt_bboxes = targets[:, :, 1:5]  # NOTE [x, y, x, y]
        gt_angles = targets[:, :, -1:]
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)  # NOTE 相对值 xyxy [bs, 13125/8400, 4]

        try:
            # TODO
            if epoch_num < self.warmup_epoch:
                target_labels, target_bboxes, target_angles, target_scores, fg_mask = self.warmup_assigner(
                    anchors,
                    n_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    gt_angles,
                    mask_gt,
                    pred_bboxes.detach() * stride_tensor,
                )
            else:
                # TODO
                target_labels, target_bboxes, target_angles, target_scores, fg_mask = self.formal_assigner(
                    pred_scores.detach(),
                    pred_bboxes.detach() * stride_tensor,
                    pred_angles.detach(),
                    anchor_points,
                    gt_labels,
                    gt_bboxes,
                    gt_angles,
                    mask_gt,
                )

        except RuntimeError:
            print(
                "OOM RuntimeError is raised due to the huge memory cost during label assignment. \
                    CPU mode is applied in this batch. If you want to avoid this issue, \
                    try to reduce the batch size or image size."
            )
            torch.cuda.empty_cache()
            print("------------CPU Mode for This Batch-------------")
            if epoch_num < self.warmup_epoch:
                _anchors = anchors.cpu().float()
                _n_anchors_list = n_anchors_list
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _gt_angles = gt_angles.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _stride_tensor = stride_tensor.cpu().float()

                target_labels, target_bboxes, target_angles, target_scores, fg_mask = self.warmup_assigner(
                    _anchors,
                    _n_anchors_list,
                    _gt_labels,
                    _gt_bboxes,
                    _gt_angles,
                    _mask_gt,
                    _pred_bboxes * _stride_tensor,
                )

            else:
                _pred_scores = pred_scores.detach().cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _pred_angles = pred_angles.detach().cpu().float()
                _anchor_points = anchor_points.cpu().float()
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _gt_angles = gt_angles.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _stride_tensor = stride_tensor.cpu().float()

                target_labels, target_bboxes, target_angles, target_scores, fg_mask = self.formal_assigner(
                    _pred_scores,
                    _pred_bboxes * _stride_tensor,
                    _pred_angles,
                    _anchor_points,
                    _gt_labels,
                    _gt_bboxes,
                    _gt_angles,
                    _mask_gt,
                )

            target_labels = target_labels.cuda()
            target_bboxes = target_bboxes.cuda()
            target_scores = target_scores.cuda()
            target_angles = target_angles.cuda()
            fg_mask = fg_mask.cuda()
        # Dynamic release GPU memory
        if step_num % 10 == 0:
            torch.cuda.empty_cache()

        # NOTE rescale bbox 相对特征图的值, 相对值
        target_bboxes /= stride_tensor

        # cls loss
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        smooth_one_hot_label = one_hot_label * (0.9 - 0.1 / (self.angle_max - 2)) + (0.1 / (self.angle_max - 2))
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

        target_scores_sum = target_scores.sum()
        # avoid devide zero error, devide by zero will cause loss to be inf or nan.
        # if target_scores_sum is 0, loss_cls equals to 0 alson
        if target_scores_sum > 0:
            loss_cls /= target_scores_sum

        # bbox loss
        loss_iou, loss_dfl = self.bbox_loss(
            pred_distri, pred_bboxes, anchor_points_s, target_bboxes, target_scores, target_scores_sum, fg_mask
        )

        # angle loss
        loss_angle = self.angle_loss(pred_angles, target_angles, target_scores, target_scores_sum, fg_mask)

        if isinstance(loss_angle, torch.Tensor):
            loss = (
                self.loss_weight["class"] * loss_cls
                + self.loss_weight["iou"] * loss_iou
                + self.loss_weight["dfl"] * loss_dfl
                + self.loss_weight["angle"] * loss_angle
            )
        else:
            loss = (
                self.loss_weight["class"] * loss_cls
                + self.loss_weight["iou"] * loss_iou
                + self.loss_weight["dfl"] * loss_dfl
                + self.loss_weight["MGAR_cls"] * loss_angle[0]
                + self.loss_weight["MGAR_reg"] * loss_angle[1]
            )
            loss_angle = loss_angle[0] + loss_angle[1]

        return (
            loss,
            torch.cat(
                (
                    (self.loss_weight["iou"] * loss_iou).unsqueeze(0),
                    (self.loss_weight["dfl"] * loss_dfl).unsqueeze(0),
                    (self.loss_weight["class"] * loss_cls).unsqueeze(0),
                    (self.loss_weight["angle"] * loss_angle).unsqueeze(0),
                )
            ).detach(),
        )

    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 6)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(
            np.array(list(map(lambda l: l + [[-1, 0, 0, 0, 0, 0]] * (max_len - len(l)), targets_list)))[:, 1:, :]
        ).to(targets.device)
        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:5] = xywh2xyxy(batch_target)
        # NOTE targets 绝对值坐标 [bs, max_len, 6]
        return targets

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(
                self.proj.to(pred_dist.device)
            )
        return dist2bbox(pred_dist, anchor_points)


class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score, gt_score, label, alpha=0.75, gamma=2.0):

        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction="none") * weight).sum()

        return loss


def gaussian_label(target_angles: torch.Tensor, angle_max: int, u=0, sig=6.0):
    # NOTE target_angles [nums_label, 1]
    # NOTE gaussian angles [nums_label, angle_max]
    smooth_label = (
        torch.zeros_like(target_angles).repeat(1, angle_max).to(device=target_angles.device).type_as(target_angles)
    )
    target_angles_long = target_angles.long()

    base_radius_range = torch.arange(-angle_max // 2, angle_max // 2, device=target_angles.device)
    radius_range = (base_radius_range + target_angles_long) % angle_max
    smooth_value = torch.exp(-torch.pow(base_radius_range, 2) / (2 * sig ** 2))

    if isinstance(smooth_value, torch.Tensor):
        smooth_value = smooth_value.unsqueeze(0).repeat(smooth_label.size(0), 1).type_as(target_angles)

    return smooth_label.scatter(1, radius_range, smooth_value)


class AngleLoss(nn.Module):
    def __init__(self, angle_max, angle_fitting_methods):
        super(AngleLoss, self).__init__()
        self.angle_max = angle_max
        self.angle_fitting_methods = angle_fitting_methods
        if self.angle_fitting_methods == "dfl" or self.angle_fitting_methods == "MGAR":
            self.angle_max += 1
        self.bce = nn.BCEWithLogitsLoss(reduction="none")

    def forward(self, pred_angles, target_angles, target_scores, target_scores_sum, fg_mask):
        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # angle loss
            angle_mask = fg_mask.unsqueeze(-1).repeat([1, 1, self.angle_max])
            target_angle_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 1])
            pred_angle_pos = torch.masked_select(pred_angles, angle_mask).reshape([-1, self.angle_max])
            target_angle_pos = torch.masked_select(target_angles, target_angle_mask).reshape([-1, 1])
            # NOTE 需不需要乘这个
            angle_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)

            if self.angle_fitting_methods == "regression":
                loss_angle = F.smooth_l1_loss(pred_angle_pos, target_angle_pos, reduction="none") * angle_weight

                if target_scores_sum == 0:
                    loss_angle = loss_angle.sum()
                else:
                    loss_angle = loss_angle.sum() / target_scores_sum
                return loss_angle

            elif self.angle_fitting_methods == "csl":
                csl_gaussian_target_angle_pos = gaussian_label(target_angle_pos, self.angle_max)
                loss_angle = self.bce(pred_angle_pos, csl_gaussian_target_angle_pos) * angle_weight

                if target_scores_sum == 0:
                    loss_angle = loss_angle.sum()
                else:
                    loss_angle = loss_angle.sum() / target_scores_sum
                return loss_angle

            elif self.angle_fitting_methods == "dfl":
                target_angle_pos = target_angle_pos / (180.0 / (self.angle_max - 1))
                loss_angle = self._df_loss(pred_angle_pos, target_angle_pos) * angle_weight

                if target_scores_sum == 0:
                    loss_angle = loss_angle.sum()
                else:
                    loss_angle = loss_angle.sum() / target_scores_sum
                return loss_angle

            elif self.angle_fitting_methods == "MGAR":
                # target_labels = torch.full_like(target_angle_pos, self.angle_max)
                class_id = target_angle_pos.clone() // (180 / (self.angle_max - 1))
                regression_value = target_angle_pos.clone() - class_id * (180 / (self.angle_max - 1))
                class_id = class_id.squeeze()
                one_hot_label = F.one_hot(class_id.long(), self.angle_max - 1)
                smooth_one_hot_label = one_hot_label * (0.9 - 0.1 / (self.angle_max - 2)) + (0.1 / (self.angle_max - 2))
                loss_angle_regression = (
                    F.smooth_l1_loss(pred_angle_pos[..., -1:] ** 2, regression_value, reduction="none") * angle_weight
                )
                loss_angle_class = (
                    self.bce(pred_angle_pos[:, :(self.angle_max - 1)], smooth_one_hot_label.float()) * angle_weight
                )

                if target_scores_sum == 0:
                    loss_angle_class = loss_angle_class.sum()
                    loss_angle_regression = loss_angle_regression.sum()
                else:
                    loss_angle_class = loss_angle_class.sum() / target_scores_sum
                    loss_angle_regression = loss_angle_regression.sum() / target_scores_sum
                return (loss_angle_class, loss_angle_regression)
        else:
            loss_angle = pred_angles.sum() * 0.0

        return loss_angle

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = (
            F.cross_entropy(pred_dist.view(-1, self.angle_max), target_left.view(-1), reduction="none").view(
                target_left.shape
            )
            * weight_left
        )
        loss_right = (
            F.cross_entropy(pred_dist.view(-1, self.angle_max), target_right.view(-1), reduction="none").view(
                target_left.shape
            )
            * weight_right
        )
        return (loss_left + loss_right).mean(-1, keepdim=True)


class BboxLoss(nn.Module):
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type="giou"):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format="xyxy", iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points, target_bboxes, target_scores, target_scores_sum, fg_mask):

        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape([-1, 4])
            target_bboxes_pos = torch.masked_select(target_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos, target_bboxes_pos) * bbox_weight
            if target_scores_sum == 0:
                loss_iou = loss_iou.sum()
            else:
                loss_iou = loss_iou.sum() / target_scores_sum

            # dfl loss
            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).repeat([1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
                if target_scores_sum == 0:
                    loss_dfl = loss_dfl.sum()
                else:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.0

        else:
            loss_iou = pred_dist.sum() * 0.0
            loss_dfl = pred_dist.sum() * 0.0

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = (
            F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction="none").view(
                target_left.shape
            )
            * weight_left
        )
        loss_right = (
            F.cross_entropy(pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction="none").view(
                target_left.shape
            )
            * weight_right
        )
        return (loss_left + loss_right).mean(-1, keepdim=True)
