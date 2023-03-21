#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.assigners.atss_assigner_R import ATSSAssigner
from yolov6.assigners.tal_assigner_R import TaskAlignedAssigner
from yolov6.utils.figure_iou import IOUloss
from yolov6.utils.general import bbox2dist, dist2bbox, xywh2xyxy


class ComputeLoss:
    """Loss computation func."""

    def __init__(
        self,
        fpn_strides=[8, 16, 32],
        grid_cell_size=5.0,
        grid_cell_offset=0.5,
        num_classes=80,
        ori_img_size=640,
        warmup_epoch=0,
        use_dfl=True,
        reg_max=16,
        angle_max=180,
        angle_fitting_methods="regression",
        iou_type="giou",
        loss_weight={"class": 1.0, "iou": 2.5, "dfl": 0.5, "angle": 0.01, "cwd": 10.0},
        distill_feat=False,
        distill_weight={"class": 1.0, "dfl": 1.0, "angle": 1.0},
    ):

        self.fpn_strides = fpn_strides  # NOTE [8, 16, 32]
        self.grid_cell_size = grid_cell_size  # NOTE 5
        self.grid_cell_offset = grid_cell_offset  # NOTE 0.5
        self.num_classes = num_classes
        self.ori_img_size = ori_img_size

        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)
        # NOTE TAL

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.angle_max = angle_max
        self.angle_fitting_methods = angle_fitting_methods

        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        # NOTE 投影矩阵 dfl 乘矩阵
        if self.angle_fitting_methods == "dfl":
            self.proj_angle = (180.0 / self.angle_max) * nn.Parameter(
                torch.linspace(0, self.angle_max, self.angle_max + 1), requires_grad=False
            )  # NOTE 投影矩阵 dfl 乘矩阵

        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss().cuda()

        self.loss_weight = loss_weight
        self.distill_feat = distill_feat
        self.distill_weight = distill_weight

    def __call__(
        self,
        outputs,
        t_outputs,
        s_featmaps,
        t_featmaps,
        targets,
        epoch_num,
        max_epoch,
        temperature,
        step_num,
        distill_off=False,
    ):
        # NOTE pred_distri 相对值 [bs, 8400, 4]
        # NOTE pred_angles [bs, 8400, angle_max]
        self.bbox_loss = BboxLoss(
            self.num_classes, self.reg_max, self.use_dfl, self.iou_type, distill_off=distill_off
        ).cuda()
        self.angle_loss = AngleLoss(self.angle_max, self.angle_fitting_methods, distill_off=distill_off).cuda()

        feats, pred_scores, pred_distri, pred_angles = outputs
        # NOTE 教师网络的四个参数也要进行处理

        t_feats, t_pred_scores, t_pred_distri, t_pred_angles = t_outputs

        anchors, anchor_points, n_anchors_list, stride_tensor = generate_anchors(
            feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device
        )
        t_anchors, t_anchor_points, t_n_anchors_list, t_stride_tensor = generate_anchors(
            t_feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device
        )
        assert pred_scores.type() == pred_distri.type()
        assert t_pred_scores.type() == t_pred_distri.type()

        gt_bboxes_scale = torch.full((1, 4), self.ori_img_size).type_as(pred_scores)
        batch_size = pred_scores.shape[0]

        # NOTE targets 绝对值坐标 [bs, max_len, 6] [class_id, x, y, x, y, angle]
        targets = self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]  # NOTE [bs, MAX_Num_labels_per_img, 1]
        gt_bboxes = targets[:, :, 1:5]  # NOTE [x,y,x,]
        gt_angles = targets[:, :, -1:]
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()

        # pboxes
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri)  # xyxy
        # NOTE 相对值 xyxy [bs, 13125/8400, 4] 教师网络以及学生网络
        t_anchor_points_s = t_anchor_points / t_stride_tensor
        t_pred_bboxes = self.bbox_decode(t_anchor_points_s, t_pred_distri)  # xyxy

        try:
            if epoch_num < self.warmup_epoch:
                target_labels, target_bboxes, target_scores, target_angles, fg_mask = self.warmup_assigner(
                    anchors,
                    n_anchors_list,
                    gt_labels,
                    gt_bboxes,
                    gt_angles,
                    mask_gt,
                    pred_bboxes.detach() * stride_tensor,
                )
            else:
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
                    _mask_gt,
                )

            target_labels = target_labels.cuda()
            target_bboxes = target_bboxes.cuda()
            target_angles = target_angles.cuda()
            target_scores = target_scores.cuda()
            fg_mask = fg_mask.cuda()

        # Dynamic release GPU memory
        if step_num % 10 == 0:
            torch.cuda.empty_cache()

        # rescale bbox
        target_bboxes /= stride_tensor

        # cls loss
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels, self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

        target_scores_sum = target_scores.sum()
        # avoid devide zero error, devide by zero will cause loss to be inf or nan.
        if target_scores_sum > 0:
            loss_cls /= target_scores_sum
        # NOTE
        # bbox loss
        loss_iou, loss_dfl, d_loss_dfl = self.bbox_loss(
            pred_distri,
            pred_bboxes,
            t_pred_distri,
            t_pred_bboxes,
            temperature,
            anchor_points_s,
            target_bboxes,
            target_scores,
            target_scores_sum,
            fg_mask,
        )
        # NOTE Angle Loss
        loss_angle, d_loss_angle = self.angle_loss(
            pred_angles, t_pred_angles, target_angles, target_scores, target_scores_sum, fg_mask, temperature
        )

        logits_student = pred_scores
        logits_teacher = t_pred_scores
        distill_num_classes = self.num_classes
        d_loss_cls = self.distill_loss_cls(logits_student, logits_teacher, distill_num_classes, temperature, fg_mask)
        if target_scores_sum > 0:
            d_loss_cls /= target_scores_sum
        if distill_off:
            d_loss_cls = d_loss_cls * 0.0
        if self.distill_feat:
            d_loss_cw = self.distill_loss_cw(s_featmaps, t_featmaps)
        else:
            d_loss_cw = torch.tensor(0.0).to(feats[0].device)
        import math

        # print(loss_cls,"loss_cls")
        # print(loss_dfl,"loss_dfl")
        # print(loss_angle, "loss_angle")
        # print(d_loss_cls, "d_loss_cls")
        # print(d_loss_dfl,"d_loss_dfl")
        # print(d_loss_angle, "d_loss_angle")
        distill_weightdecay = ((1 - math.cos(epoch_num * math.pi / max_epoch)) / 2) * (0.01 - 1) + 1
        d_loss_dfl *= distill_weightdecay
        d_loss_cls *= distill_weightdecay
        d_loss_cw *= distill_weightdecay
        d_loss_angle *= distill_weightdecay
        loss_cls_all = loss_cls + d_loss_cls * self.distill_weight["class"] * 0.20
        loss_dfl_all = loss_dfl + d_loss_dfl * self.distill_weight["dfl"] * 0.20
        loss_angle_all = loss_angle + d_loss_angle * self.distill_weight["angle"] * 0.20
        loss = (
            self.loss_weight["class"] * loss_cls_all
            + self.loss_weight["iou"] * loss_iou
            + self.loss_weight["dfl"] * loss_dfl_all
            + self.loss_weight["angle"] * loss_angle_all
            + self.loss_weight["cwd"] * d_loss_cw
        )

        return (
            loss,
            torch.cat(
                (
                    (self.loss_weight["iou"] * loss_iou).unsqueeze(0),
                    (self.loss_weight["dfl"] * loss_dfl_all).unsqueeze(0),
                    (self.loss_weight["class"] * loss_cls_all).unsqueeze(0),
                    (self.loss_weight["angle"] * loss_angle_all).unsqueeze(0),
                    (self.loss_weight["cwd"] * d_loss_cw).unsqueeze(0),
                )
            ).detach(),
        )

    def distill_loss_cls(self, logits_student, logits_teacher, num_classes, temperature=20, fg_mask=None):
        # NOTE 只对正样本做KL Divergance 负样本不进行处理
        cls_mask = fg_mask.unsqueeze(-1).repeat([1, 1, num_classes])
        s_logits_pred = torch.masked_select(logits_student, cls_mask).reshape(-1, num_classes)
        t_logits_pred = torch.masked_select(logits_teacher, cls_mask).reshape(-1, num_classes)
        # logits_student = logits_student.view(-1, num_classes)
        # logits_teacher = logits_teacher.view(-1, num_classes)
        # pred_student = F.softmax(logits_student / temperature, dim=1)
        # pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        s_logits_pred = F.softmax(s_logits_pred / temperature, dim=1)
        t_logits_pred = F.softmax(t_logits_pred / temperature, dim=1)
        log_pred_student = torch.log(s_logits_pred)
        # d_loss_cls = F.kl_div(log_pred_student, pred_teacher, reduction="sum")
        # print(log_pred_student.shape)
        d_loss_cls = F.kl_div(log_pred_student, t_logits_pred, reduction="sum")

        d_loss_cls *= temperature**2
        return d_loss_cls

    def distill_loss_cw(self, s_feats, t_feats, temperature=1):
        N, C, H, W = s_feats[0].shape
        # print(N,C,H,W)
        loss_cw = (
            F.kl_div(
                F.log_softmax(s_feats[0].view(N, C, H * W) / temperature, dim=2),
                F.log_softmax(t_feats[0].view(N, C, H * W).detach() / temperature, dim=2),
                reduction="sum",
                log_target=True,
            )
            * (temperature * temperature)
            / (N * C)
        )

        N, C, H, W = s_feats[1].shape
        # print(N,C,H,W)
        loss_cw += (
            F.kl_div(
                F.log_softmax(s_feats[1].view(N, C, H * W) / temperature, dim=2),
                F.log_softmax(t_feats[1].view(N, C, H * W).detach() / temperature, dim=2),
                reduction="sum",
                log_target=True,
            )
            * (temperature * temperature)
            / (N * C)
        )

        N, C, H, W = s_feats[2].shape
        # print(N,C,H,W)
        loss_cw += (
            F.kl_div(
                F.log_softmax(s_feats[2].view(N, C, H * W) / temperature, dim=2),
                F.log_softmax(t_feats[2].view(N, C, H * W).detach() / temperature, dim=2),
                reduction="sum",
                log_target=True,
            )
            * (temperature * temperature)
            / (N * C)
        )
        # print(loss_cw)
        return loss_cw

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


class BboxLoss(nn.Module):
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type="giou", distill_off=False):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format="xyxy", iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl
        self.distill_off = distill_off

    def forward(
        self,
        pred_dist,
        pred_bboxes,
        t_pred_dist,
        t_pred_bboxes,
        temperature,
        anchor_points,
        target_bboxes,
        target_scores,
        target_scores_sum,
        fg_mask,
    ):
        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes, bbox_mask).reshape([-1, 4])
            t_pred_bboxes_pos = torch.masked_select(t_pred_bboxes, bbox_mask).reshape([-1, 4])
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
                t_pred_dist_pos = torch.masked_select(t_pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = torch.masked_select(target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos, target_ltrb_pos) * bbox_weight
                d_loss_dfl = self.distill_loss_dfl(pred_dist_pos, t_pred_dist_pos, temperature) * bbox_weight
                if target_scores_sum == 0:
                    loss_dfl = loss_dfl.sum()
                    d_loss_dfl = d_loss_dfl.sum()
                else:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
                    d_loss_dfl = d_loss_dfl.sum() / target_scores_sum
            else:
                loss_dfl = pred_dist.sum() * 0.0
                d_loss_dfl = pred_dist.sum() * 0.0

        else:

            loss_iou = pred_dist.sum() * 0.0
            loss_dfl = pred_dist.sum() * 0.0
            d_loss_dfl = pred_dist.sum() * 0.0
        if not self.distill_off:
            return loss_iou, loss_dfl, d_loss_dfl
        else:
            d_loss_dfl = d_loss_dfl * 0.0
            return loss_iou, loss_dfl, d_loss_dfl

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

    def distill_loss_dfl(self, logits_student, logits_teacher, temperature=20):

        logits_student = logits_student.view(-1, 17)
        logits_teacher = logits_teacher.view(-1, 17)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        log_pred_student = torch.log(pred_student)

        d_loss_dfl = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        d_loss_dfl *= temperature**2
        return d_loss_dfl


def gaussian_label(target_angles: torch.Tensor, angle_max: int, u=0, sig=6.0):
    # NOTE target_angles [nums_label, 1]
    # NOTE gaussian angles [nums_label, angle_max]
    smooth_label = (
        torch.zeros_like(target_angles).repeat(1, angle_max).to(device=target_angles.device).type_as(target_angles)
    )
    target_angles_long = target_angles.long()

    base_radius_range = torch.arange(-angle_max // 2, angle_max // 2, device=target_angles.device)
    radius_range = (base_radius_range + target_angles_long) % angle_max
    smooth_value = torch.exp(-torch.pow(base_radius_range, 2) / (2 * sig**2))

    if isinstance(smooth_value, torch.Tensor):
        smooth_value = smooth_value.unsqueeze(0).repeat(smooth_label.size(0), 1).type_as(target_angles)

    return smooth_label.scatter(1, radius_range, smooth_value)


class AngleLoss(nn.Module):
    def __init__(self, angle_max, angle_fitting_methods, distill_off=False):
        super(AngleLoss, self).__init__()
        self.angle_max = angle_max
        self.angle_fitting_methods = angle_fitting_methods
        if self.angle_fitting_methods == "dfl":
            self.angle_max += 1
        self.bce = nn.BCEWithLogitsLoss(reduction="none")
        self.distill_off = distill_off

    def distill_loss_angle(self, logits_student, logits_teacher, temperature):
        logits_student = logits_student.view(-1, self.angle_max)
        logits_teacher = logits_teacher.view(-1, self.angle_max)
        pred_student = F.softmax(logits_student / temperature, dim=1)
        pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
        log_pred_student = torch.log(pred_student)
        d_loss_angle = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
        d_loss_angle *= temperature**2
        return d_loss_angle

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

    def forward(
        self, pred_angles, t_pred_angles, target_angles, target_scores, target_scores_sum, fg_mask, temperature=20
    ):
        # NOTE 参考分类的KL Divergence
        num_pos = fg_mask.sum()
        if num_pos > 0:
            angle_mask = fg_mask.unsqueeze(-1).repeat([1, 1, self.angle_max])
            target_angle_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 1])
            pred_angle_pos = torch.masked_select(pred_angles, angle_mask).reshape([-1, self.angle_max])
            t_pred_angle_pos = torch.masked_select(t_pred_angles, angle_mask).reshape([-1, self.angle_max])
            target_angle_pos = torch.masked_select(target_angles, target_angle_mask).reshape([-1, 1])
            angle_weight = torch.masked_select(target_scores.sum(-1), fg_mask).unsqueeze(-1)
            if self.angle_fitting_methods == "regression":
                loss_angle = F.smooth_l1_loss(pred_angle_pos, target_angle_pos, reduction="none") * angle_weight
                d_loss_angle = F.smooth_l1_loss(pred_angle_pos, t_pred_angle_pos, reduction="none") * angle_weight
            elif self.angle_fitting_methods == "csl":
                csl_gaussian_target_angle_pos = gaussian_label(target_angle_pos, self.angle_max)
                loss_angle = self.bce(pred_angle_pos, csl_gaussian_target_angle_pos) * angle_weight
                d_loss_angle = self.distill_loss_angle(pred_angle_pos, t_pred_angle_pos, temperature) * angle_weight
            # dfl loss
            elif self.angle_fitting_methods == "dfl":
                target_angle_pos = target_angle_pos / (180.0 / (self.angle_max - 1))
                loss_angle = self._df_loss(pred_angle_pos, target_angle_pos) * angle_weight
                d_loss_angle = self.distill_loss_angle(pred_angle_pos, t_pred_angle_pos, temperature) * angle_weight
            if target_scores_sum == 0:
                loss_angle = loss_angle.sum()
                d_loss_angle = d_loss_angle.sum()
            else:
                loss_angle = loss_angle.sum() / target_scores_sum
                d_loss_angle = d_loss_angle.sum() / target_scores_sum
        else:
            loss_angle = pred_angles.sum() * 0.0
            d_loss_angle = pred_angles.sum() * 0.0
        if self.distill_off:
            d_loss_angle = d_loss_angle * 0.0
        return loss_angle, d_loss_angle
