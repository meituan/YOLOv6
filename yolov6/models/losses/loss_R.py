#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import torch
import torch.nn as nn
import numpy as np
from loguru import logger
import torch.nn.functional as F
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox, bbox2dist, xywh2xyxy
from yolov6.utils.figure_iou import IOUloss
from yolov6.assigners.atss_assigner import ATSSAssigner
from yolov6.assigners.rotated_tal_assigner import RotatedTaskAlignedAssigner
import ipdb
class ComputeLoss:
    '''Loss computation func.'''
    def __init__(self,
                 fpn_strides=[8, 16, 32],
                 grid_cell_size=5.0,
                 grid_cell_offset=0.5,
                 num_classes=5,
                 ori_img_size=640,
                 warmup_epoch=5,
                 use_reg_dfl=True,
                 use_angle_dfl=True,
                 reg_max=16,
                 angle_max = 90,
                 iou_type='giou',
                 loss_weight={
                     'class': 1,
                     'iou': 2.5,
                     'reg_dfl': 0.1,
                     'angle_dfl':0.05}
                 ):

        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.ori_img_size = ori_img_size
        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = RotatedTaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)
        
        
        self.use_reg_dfl = use_reg_dfl
        self.use_angle_dfl = use_angle_dfl
        self.reg_max = reg_max
        self.angle_max =  angle_max
        self.reg_proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.half_pi = torch.tensor(
            [1.5707963267948966],dtype=torch.float32)
        self.half_pi_bin = self.half_pi/self.angle_max
        self.angle_proj = self.half_pi_bin*nn.Parameter(torch.linspace(0, self.angle_max, self.angle_max + 1), requires_grad=False)
        
        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, reg_max=self.reg_max,angle_max=self.angle_max,use_reg_dfl=self.use_reg_dfl,use_angle_dfl=self.use_angle_dfl).cuda()
        self.loss_weight = loss_weight

    def __call__(
        self,
        outputs,
        targets,
        epoch_num,
        step_num
    ):
        # feats [x[0]x[1]x[2]] [bs,64,80,80] [bs,128,40,40] [bs,256,20,20]
        # pred_scores [bs,8400,10]
        # cls_score [bs,8400,15]
        feats, pred_scores, pred_reg_dist, pred_angle_dist = outputs
        anchors, anchor_points, n_anchors_list, stride_tensor = \
               generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)
        assert pred_scores.type() == pred_reg_dist.type()
    
        gt_bboxes_scale = torch.full((1,4), self.ori_img_size).type_as(pred_scores)
        batch_size = pred_scores.shape[0]
        # 获取对应的batch size
        # [class_id,xmin,ymin,xmax,ymax,a1,a2,a3,a4,angle,c_x,c_y,box_w,box_h] 传入的label
        targets = self.preprocess(targets, batch_size, gt_bboxes_scale)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:] # x y w h theta
        # NOTE: xywh2xyx
        gt_bboxes_atss = xywh2xyxy(gt_bboxes[:,:,:-1])
        mask_gt = (gt_bboxes[:,:,1:-2].sum(-1, keepdim=True) > 0).float()
        anchor_points_s = anchor_points/stride_tensor
        pred_reg_atss, pred_angle = self.obb_decode(anchor_points_s, pred_reg_dist,\
                pred_angle_dist,stride_tensor,mode="xyxy")
        pred_reg_tal, pred_angle = self.obb_decode(anchor_points_s, pred_reg_dist,\
                pred_angle_dist,stride_tensor,mode="xywh")
        pred_bboxes = torch.concat([pred_reg_tal.clone()*stride_tensor,pred_angle],dim=-1)
        try:
            if epoch_num < self.warmup_epoch:
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.warmup_assigner(
                        anchors,
                        n_anchors_list,
                        gt_labels,
                        gt_bboxes_atss,
                        gt_bboxes,
                        mask_gt,
                        pred_reg_atss.detach()*stride_tensor)
            else:
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.formal_assigner(
                        pred_scores.detach().type_as(gt_labels),
                        torch.cat([pred_reg_tal.detach()*stride_tensor,pred_angle.detach()],axis=-1).type_as(gt_labels),
                        anchor_points.type_as(gt_labels),
                        gt_labels,
                        gt_bboxes,
                        mask_gt.type_as(gt_labels))
                    # mmcv 需要fp32 进行计算
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
                _mask_gt = mask_gt.cpu().float()
                _pred_reg_atss = pred_reg_atss.detach().cpu().float()
                _stride_tensor = stride_tensor.cpu().float()
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.warmup_assigner(
                        _anchors,
                        _n_anchors_list,
                        _gt_labels,
                        _gt_bboxes,
                        _mask_gt,
                        _pred_reg_atss * _stride_tensor)

            else:
                _pred_scores = pred_scores.detach().cpu().float()
                _pred_reg_tal = pred_reg_tal.detach().cpu().float()
                _pred_angle = pred_angle.detach().cpu().float()
                _anchor_points = anchor_points.cpu().float()
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _stride_tensor = stride_tensor.cpu().float()
               
                target_labels, target_bboxes, target_scores, fg_mask = \
                    self.formal_assigner(
                        _pred_scores,
                        torch.cat([_pred_reg_tal*_stride_tensor, _pred_angle],dim=-1),
                        _anchor_points,
                        _gt_labels,
                        _gt_bboxes,
                        _mask_gt)

            target_labels = target_labels.cuda()
            target_bboxes = target_bboxes.cuda()
            target_scores = target_scores.cuda()
            fg_mask = fg_mask.cuda()
        #Dynamic release GPU memory
        if step_num % 10 == 0:
            torch.cuda.empty_cache()

        # TODO rescale bbox 这个目前先不需要进行rescale操作，DFL loss会加上这么一个东西
      #  target_bboxes /= stride_tensor
        # cls loss
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)

        target_scores_sum = target_scores.sum()
        # avoid devide zero error, devide by zero will cause loss to be inf or nan.
        # if target_scores_sum is 0, loss_cls equals to 0 alson 
        if target_scores_sum > 0:
            loss_cls /= target_scores_sum
        
        # bbox loss
        """

            def forward(self, pred_angle_dist, pred_reg_dist,pred_bboxes,anchor_points,assigned_bboxes, assigned_scores,assigned_scores_sum,stride_tensor,\
            fg_mask)
        """
        
        """
             def forward(self, pred_angle_dist, pred_reg_dist,pred_bboxes,anchor_points,assigned_bboxes, assigned_scores,assigned_scores_sum,stride_tensor,\
            fg_mask):
        """
       # ipdb.set_trace()
        loss_iou, loss_reg_dfl, loss_angle_dfl = self.bbox_loss(pred_angle_dist,pred_reg_dist,pred_bboxes,anchor_points_s, target_bboxes,target_scores,\
            target_scores_sum,stride_tensor,fg_mask)
        
        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['reg_dfl'] * loss_reg_dfl +\
               self.loss_weight['angle_dfl'] * loss_angle_dfl

        return loss, \
            torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0),
                         (self.loss_weight['reg_dfl'] * loss_reg_dfl).unsqueeze(0),
                         (self.loss_weight['class'] * loss_cls).unsqueeze(0),
                      (self.loss_weight['angle_dfl']* loss_angle_dfl).unsqueeze(0))).detach()

    def preprocess(self, targets, batch_size, scale_tensor):
        targets_list = np.zeros((batch_size, 1, 6)).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            targets_list[int(item[0])].append(item[1:])
        max_len = max((len(l) for l in targets_list))
        targets = torch.from_numpy(np.array(list(map(lambda l:l + [[-1,0,0,0,0,0]]*(max_len - len(l)), targets_list)))[:,1:,:]).to(targets.device)
        batch_target = targets[:,:,1:5]
        targets[...,1:5] = batch_target
        targets = targets.float()
        return targets
    def obb_decode(self, points, pred_dist, pred_angle, stride_tensor, mode="xyxy"):
        if self.use_reg_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(self.reg_proj.to(pred_dist.device))
        else:
            pred_dist = pred_dist
            
        if self.use_angle_dfl:
            batch_size, n_anchors, _ = pred_angle.shape
            pred_angle = F.softmax(pred_angle.view(batch_size, n_anchors, 1, self.angle_max + 1), dim=-1).matmul(self.angle_proj.to(pred_angle.device))
        else:
            pred_angle = pred_angle
        # 将DFL的蒸馏去除
        pred_dist = dist2bbox(pred_dist, points, box_format= mode) 
        return pred_dist, pred_angle
                 
    # def bbox_decode(self, anchor_points, pred_dist):
    #     if self.use_dfl:
    #         batch_size, n_anchors, _ = pred_dist.shape
    #         pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(self.proj.to(pred_dist.device))
    #     return dist2bbox(pred_dist, anchor_points)


class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score,gt_score, label, alpha=0.75, gamma=2.0):

        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()

        return loss





class BboxLoss(nn.Module):
    def __init__(self, num_classes, reg_max, angle_max,use_reg_dfl=False, use_angle_dfl=False,iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format="xywh",iou_type="siou")
        # self.iou_loss = ProbIoUloss()
        self.reg_max = reg_max
        self.angle_max = angle_max
        self.use_reg_dfl = use_reg_dfl
        self.use_angle_dfl = use_angle_dfl
    def forward(self, pred_angle_dist, pred_reg_dist,pred_bboxes,anchor_points,assigned_bboxes, assigned_scores,assigned_scores_sum,stride_tensor,\
            fg_mask):
        """
            pred_angle:
            pred_dist: 
        """
        self.half_pi = torch.tensor(
            [1.5707963267948966],dtype=torch.float32).to(pred_angle_dist.device)
        self.half_pi_bin = self.half_pi/self.angle_max

        # select positive samples mask
        # ipdb.set_trace()
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            # ipdb.set_trace()
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 5])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,\
                                                  bbox_mask).reshape([-1, 5])
            assigned_bboxes_pos = torch.masked_select(
                assigned_bboxes, bbox_mask).reshape([-1, 5])
            bbox_weight = torch.masked_select(
                assigned_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos[...,:-1],
                                     assigned_bboxes_pos[...,:-1]) * bbox_weight
            # loss_iou = self.iou_loss(pred_bboxes_pos,
            #                          assigned_bboxes_pos) * bbox_weight
            if assigned_scores_sum == 0:
                loss_iou = loss_iou.sum()
            else:
                loss_iou = loss_iou.sum() / assigned_scores_sum
            if self.use_angle_dfl:
                angle_mask = fg_mask.unsqueeze(-1).repeat([1,1,self.angle_max+1])
                pred_angle_pos = torch.masked_select(pred_angle_dist, angle_mask).reshape(
                    [-1, self.angle_max + 1]
                )
                # TODO clip(self.angle_max)
                assigned_angle_pos = (
                    assigned_bboxes_pos[:, 4] /
                self.half_pi_bin)
                loss_angle_dfl = self._df_loss_angle(pred_angle_pos, assigned_angle_pos) * bbox_weight
                if assigned_scores_sum == 0:
                    loss_angle_dfl = loss_angle_dfl.sum()*0.0
                else:
                    # loss_angle_dfl = loss_angle_dfl.sum() / assigned_scores_sum
                    loss_angle_dfl = loss_angle_dfl.sum()/assigned_scores_sum
            else:
                # ipdb.set_trace()
                angle_mask = fg_mask.unsqueeze(-1).repeat([1,1,self.angle_max+1])
                pred_angle_pos = torch.masked_select(pred_angle_dist, angle_mask).reshape(
                    [-1, self.angle_max + 1]
                )
                # TODO 拟合到0-90°
                assigned_angle_pos = (
                    assigned_bboxes_pos[:, 4] /
                self.half_pi_bin).reshape([-1,self.angle_max+1])
                loss_angle_dfl = F.smooth_l1_loss(assigned_angle_pos, pred_angle_pos).sum()/assigned_scores_sum
            if self.use_reg_dfl:
                bbox_dfl_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
                
                dfl_bboxes = xywh2xyxy(assigned_bboxes[:,:,:4].clone())/stride_tensor
                pred_t = pred_reg_dist
                dist_mask = fg_mask.unsqueeze(-1).repeat(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(
                    pred_t, dist_mask).reshape([-1, 4, self.reg_max + 1])
                assigned_ltrb = bbox2dist(anchor_points, dfl_bboxes, self.reg_max)
                assigned_ltrb_pos = torch.masked_select(
                    assigned_ltrb, bbox_dfl_mask).reshape([-1, 4])
                loss_reg_dfl = self._df_loss_reg(pred_dist_pos,
                                        assigned_ltrb_pos) * bbox_weight
                if assigned_scores_sum == 0:
                    loss_reg_dfl = loss_reg_dfl.sum()
                else:
                    loss_reg_dfl = loss_reg_dfl.sum() / assigned_scores_sum
            else:
                loss_reg_dfl =  torch.tensor(0.).to(pred_reg_dist.device)
        else:
            loss_iou = torch.tensor(0.).to(pred_bboxes.device)
            loss_reg_dfl = torch.tensor(0.).to(pred_reg_dist.device)
            loss_angle_dfl = torch.tensor(0.).to(pred_angle_dist.device)

        return loss_iou, loss_reg_dfl, loss_angle_dfl
    def _df_loss_angle(self, pred_angle_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_angle_dist.view(-1, self.angle_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(
            pred_angle_dist.view(-1, self.angle_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

    def _df_loss_reg(self, pred_reg_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_reg_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(
            pred_reg_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)
