#!/usr/bin/env python3
# -*- coding:utf-8 -*-

import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox, bbox2dist, xywh2xyxy, box_iou
from yolov6.utils.figure_iou import IOUloss
from yolov6.assigners.atss_assigner_seg import ATSSAssigner
from yolov6.assigners.tal_assigner_seg import TaskAlignedAssigner
import time
import pickle

class ComputeLoss:
    '''Loss computation func.'''
    def __init__(self,
                 fpn_strides=[8, 16, 32],
                 grid_cell_size=5.0,
                 grid_cell_offset=0.5,
                 num_classes=80,
                 ori_img_size=640,
                 warmup_epoch=4,
                 use_dfl=True,
                 reg_max=16,
                 nm=32,
                 iou_type='giou',
                 loss_weight={
                     'class': 1.0,
                     'iou': 2.5,
                     'dfl': 0.5,
                     'seg': 2.5},
                 ):

        self.fpn_strides = fpn_strides
        self.grid_cell_size = grid_cell_size
        self.grid_cell_offset = grid_cell_offset
        self.num_classes = num_classes
        self.ori_img_size = ori_img_size
        self.nm = nm
        self.tt = nm
        self.warmup_epoch = warmup_epoch
        self.warmup_assigner = ATSSAssigner(9, num_classes=self.num_classes)
        self.formal_assigner = TaskAlignedAssigner(topk=13, num_classes=self.num_classes, alpha=1.0, beta=6.0)

        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.iou_type = iou_type
        self.varifocal_loss = VarifocalLoss().cuda()
        self.bbox_loss = BboxLoss(self.num_classes, self.reg_max, self.use_dfl, self.iou_type).cuda()
        self.loss_weight = loss_weight

    def __call__(
        self,
        outputs,
        targets,
        epoch_num,
        step_num,
        batch_height,
        batch_width,
        segmasks,
        img=None,
    ):
        
        feats, pred_scores, pred_distri, pred_seg  = outputs # seg_list:shape(3)(b, nm, mw, mh) seg_conf_list:shape(3):(b, l ,nm)
        seg_cf, seg_proto = pred_seg
        anchors, anchor_points, n_anchors_list, stride_tensor = \
               generate_anchors(feats, self.fpn_strides, self.grid_cell_size, self.grid_cell_offset, device=feats[0].device)

        assert pred_scores.type() == pred_distri.type()
        gt_bboxes_scale = torch.tensor([batch_width, batch_height, batch_width, batch_height]).type_as(pred_scores)
        batch_size = pred_scores.shape[0]

        targets, gt_segmasks =self.preprocess(targets, batch_size, gt_bboxes_scale, segmasks)
        gt_labels = targets[:, :, :1]
        gt_bboxes = targets[:, :, 1:] #xyxy
        mask_gt = (gt_bboxes.sum(-1, keepdim=True) > 0).float()
        
        anchor_points_s = anchor_points / stride_tensor
        pred_bboxes = self.bbox_decode(anchor_points_s, pred_distri) #xyxy
        try:
            if epoch_num < self.warmup_epoch:
                target_labels, target_bboxes, target_scores, fg_mask, target_segmasks = \
                    self.warmup_assigner(
                        anchors,
                        n_anchors_list,
                        gt_labels,
                        gt_bboxes,
                        mask_gt,
                        pred_bboxes.detach() * stride_tensor,
                        gt_segmasks)
            else:
                target_labels, target_bboxes, target_scores, fg_mask, idx_lst = \
                    self.formal_assigner(
                        pred_scores.detach(),
                        pred_bboxes.detach() * stride_tensor,
                        anchor_points,
                        gt_labels,
                        gt_bboxes,
                        mask_gt,
                        gt_segmasks)

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
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _stride_tensor = stride_tensor.cpu().float()
                _segmasks = gt_segmasks.cpu().float()

                target_labels, target_bboxes, target_scores, fg_mask, target_segmasks = \
                    self.warmup_assigner(
                        _anchors,
                        _n_anchors_list,
                        _gt_labels,
                        _gt_bboxes,
                        _mask_gt,
                        _pred_bboxes * _stride_tensor,
                        _segmasks)

            else:
                _pred_scores = pred_scores.detach().cpu().float()
                _pred_bboxes = pred_bboxes.detach().cpu().float()
                _anchor_points = anchor_points.cpu().float()
                _gt_labels = gt_labels.cpu().float()
                _gt_bboxes = gt_bboxes.cpu().float()
                _mask_gt = mask_gt.cpu().float()
                _stride_tensor = stride_tensor.cpu().float()
                _segmasks = gt_segmasks.cpu().float()

                target_labels, target_bboxes, target_scores, fg_mask, idx_lst = \
                    self.formal_assigner(
                        _pred_scores,
                        _pred_bboxes * _stride_tensor,
                        _anchor_points,
                        _gt_labels,
                        _gt_bboxes,
                        _mask_gt,
                        _segmasks)

            target_labels = target_labels.cuda()
            target_bboxes = target_bboxes.cuda()
            target_scores = target_scores.cuda()
            fg_mask = fg_mask.cuda()
            for _ in idx_lst:
                _ = _.cuda()

        
        if step_num % 10 == 0:
            torch.cuda.empty_cache()

        # rescale bbox
        target_bboxes /= stride_tensor

        # cls loss
        target_labels = torch.where(fg_mask > 0, target_labels, torch.full_like(target_labels, self.num_classes))
        one_hot_label = F.one_hot(target_labels.long(), self.num_classes + 1)[..., :-1]
        loss_cls = self.varifocal_loss(pred_scores, target_scores, one_hot_label)


        target_scores_sum = target_scores.sum()
        
		# avoid devide zero error, devide by zero will cause loss to be inf or nan.
        # if target_scores_sum is 0, loss_cls equals to 0 alson
        if target_scores_sum > 1:
            loss_cls /= target_scores_sum

        # bbox loss
        loss_iou, loss_dfl = self.bbox_loss(pred_distri, pred_bboxes, anchor_points_s, target_bboxes,
                                            target_scores, target_scores_sum, fg_mask)

        loss_seg = self.mask_loss(gt_segmasks, seg_cf, seg_proto, target_bboxes, fg_mask, idx_lst, target_scores, target_scores_sum)

        loss = self.loss_weight['class'] * loss_cls + \
               self.loss_weight['iou'] * loss_iou + \
               self.loss_weight['dfl'] * loss_dfl + \
               self.loss_weight['seg'] * loss_seg


        return loss, \
            torch.cat(((self.loss_weight['iou'] * loss_iou).unsqueeze(0),
                         (self.loss_weight['dfl'] * loss_dfl).unsqueeze(0),
                         (self.loss_weight['class'] * loss_cls).unsqueeze(0),
                         (self.loss_weight['seg'] * loss_seg).unsqueeze(0))).detach()

    def preprocess(self, targets, batch_size, scale_tensor, segmask):
        targets_list = np.zeros((batch_size, 1, 5)).tolist()
        cu = []
        already = []
        # seg_list = np.zeros((batch_size, 1, *segmask.shape[1:])).tolist()
        for i, item in enumerate(targets.cpu().numpy().tolist()):
            index = int(item[0])
            targets_list[index].append(item[1:])
            if index not in already:
                already.append(index)
                cu.append(i)
        cu.append(segmask.shape[0])
        max_len = max((len(l) for l in targets_list))
        segmasks = torch.zeros(batch_size, max_len - 1, segmask.shape[-2], segmask.shape[-1]).cuda()
        if len(already) != 0:
            for i in range(len(already)):
                j = already[i]
                start = cu[i]
                end = cu[i+1]
                segmasks[j, : end - start] = segmask[start: end].clone()
        targets = torch.from_numpy(np.array(list(map(lambda l:l + [[-1,0,0,0,0]]*(max_len - len(l)), targets_list)))[:,1:,:]).to(targets.device)

        batch_target = targets[:, :, 1:5].mul_(scale_tensor)
        targets[..., 1:] = xywh2xyxy(batch_target)
        return targets, segmasks

    def bbox_decode(self, anchor_points, pred_dist):
        if self.use_dfl:
            batch_size, n_anchors, _ = pred_dist.shape
            pred_dist = F.softmax(pred_dist.view(batch_size, n_anchors, 4, self.reg_max + 1), dim=-1).matmul(self.proj.to(pred_dist.device))
        return dist2bbox(pred_dist, anchor_points)

    def mask_loss(self, gt_segmasks, seg_cf, seg_proto, txyxy_ori, fg_mask, idx_lst, target_scores=None, target_scores_sum=None):
            # pred_mask_lst -> list
            '''
            pred_mask -> Shape(n1, w, h)
            gt_mask -> Shape(n, img_w, img_h)
            xyxy -> Shape(n, 4)
            sum(n1, n2, n3, ...) = n
            torch.abs((xyxy[..., 3] - xyxy[..., 1]) * (xyxy[..., 4] - xyxy[..., 2])) -> area
            fg_mask --> (bs, tsize)
            idx -> (bs, tsize)
            gt_segmasks -> (bs, labelsize, w, h)
            '''
            sl = 0
            sl2 = 0
            bl = [2, 4, 8]
            num_pos = fg_mask.sum()
            tloss = torch.zeros(1).float().cuda()
            if num_pos<=0:
                for ipred in seg_proto:
                    tloss += (ipred.sum() * 0.)
                for ipred in seg_cf:
                    tloss += (ipred.sum() * 0.)
                return tloss[0]


            xyxy_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            mtarget_scores = target_scores.sum(-1) # (bs, nl, 1)

            sl = 0
            qf = len(idx_lst) == 1 and len(idx_lst[0].shape) == 2
            if qf:
                idx_lst = idx_lst[0]
            for j in range(len(seg_cf)):
                ishape = 0
                pshape = 0

                iseg_proto = seg_proto[0] # (bs, 32, h, w)
                bs = iseg_proto.shape[0]
                iseg_cf = seg_cf[j] # (bs, part_n, 32)

                pshape = iseg_proto.shape[-1]
                ishape = iseg_cf.shape[1] # (1) = part_n
                idx = idx_lst[:, sl: sl + ishape] # (bs, part_n)

                ifg_mask = fg_mask[:, sl: sl + ishape] # (n) --> (bs, part_n)
                itarget_scores = mtarget_scores[:, sl: sl + ishape]
                if ifg_mask.sum() <= 0:
                     tloss += (iseg_proto.sum() * 0.)
                     tloss += (iseg_cf.sum() * 0.)
                     continue
                target_sg = []
                pred_sg = []
                ixyxy_lst = []
                mask_weight = []
                for i in range(bs):
                    idx_thisbatch = torch.masked_select(idx[i], ifg_mask[i]) #(casize)
                    igt_segmasks = gt_segmasks.reshape(-1, *gt_segmasks.shape[-2:])[idx_thisbatch] # (?1, h?, w?) --> (?2, h?, w?)
                    imask_weight = torch.masked_select(itarget_scores[i], ifg_mask[i]).unsqueeze(-1)
                    mask_weight.append(imask_weight)
                    target_sg.append(igt_segmasks)
                    tiseg_cf = torch.masked_select(iseg_cf[i], ifg_mask[i].unsqueeze(-1).repeat(1, self.tt)) # (?2, 32)
                    tiseg_cf = tiseg_cf.reshape(-1, self.tt)
                    ipred_seg = (tiseg_cf@iseg_proto[i].reshape(self.tt, -1)).reshape(-1, pshape, pshape) # (?2, h, w)
                    ixyxy = torch.masked_select(txyxy_ori[i, sl: sl + ishape], xyxy_mask[i, sl: sl + ishape, :]).reshape(-1, 4) # (n, 4) --> (part_n, 4) --> (?2, 4)
                    ixyxy_lst.append(ixyxy)
                    pred_sg.append(ipred_seg)
                
                


                bxyxy = torch.cat(ixyxy_lst, dim = 0) * bl[j]
                bpred_seg = torch.cat(pred_sg, dim = 0)
                bgt_seg = torch.cat(target_sg, dim = 0)
                masks_weight = torch.cat(mask_weight, dim = 0).reshape(-1)
                if tuple(bgt_seg.shape[-2:]) != (pshape, pshape):  # downsample
                    bgt_seg = F.interpolate(bgt_seg[None], (pshape, pshape), mode='nearest')[0]
                area = torch.abs((bxyxy[..., 2] - bxyxy[..., 0]) * (bxyxy[..., 3] - bxyxy[..., 1]))
                area = area / (pshape)
                area = area / (pshape)

                
                
                
                    
                sl += ishape
                loss = F.binary_cross_entropy_with_logits(bpred_seg, bgt_seg, reduction='none')
                
                loss = (self.crop_mask(loss, bxyxy).mean(dim=(1, 2)) / area) * masks_weight
                loss = loss.sum()
                tloss += loss
            if target_scores_sum > 1:
                tloss[0] = tloss[0] / target_scores_sum
            return tloss[0] / len(seg_cf)


    @staticmethod
    def crop_mask(masks, boxes):
        """
        "Crop" predicted masks by zeroing out everything not in the predicted bbox.
        Vectorized by Chong (thanks Chong).

        Args:
            - masks should be a size [n, h, w] tensor of masks
            - boxes should be a size [n, 4] tensor of bbox coords in relative point form
        """

        n, h, w = masks.shape
        x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
        r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
        c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)

        return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))


class VarifocalLoss(nn.Module):
    def __init__(self):
        super(VarifocalLoss, self).__init__()

    def forward(self, pred_score,gt_score, label, alpha=0.75, gamma=2.0):

        weight = alpha * pred_score.pow(gamma) * (1 - label) + gt_score * label
        with torch.cuda.amp.autocast(enabled=False):
            loss = (F.binary_cross_entropy(pred_score.float(), gt_score.float(), reduction='none') * weight).sum()

        return loss


class BboxLoss(nn.Module):
    def __init__(self, num_classes, reg_max, use_dfl=False, iou_type='giou'):
        super(BboxLoss, self).__init__()
        self.num_classes = num_classes
        self.iou_loss = IOUloss(box_format='xyxy', iou_type=iou_type, eps=1e-10)
        self.reg_max = reg_max
        self.use_dfl = use_dfl

    def forward(self, pred_dist, pred_bboxes, anchor_points,
                target_bboxes, target_scores, target_scores_sum, fg_mask):

        # select positive samples mask
        num_pos = fg_mask.sum()
        if num_pos > 0:
            # iou loss
            bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
            pred_bboxes_pos = torch.masked_select(pred_bboxes,
                                                  bbox_mask).reshape([-1, 4])
            target_bboxes_pos = torch.masked_select(
                target_bboxes, bbox_mask).reshape([-1, 4])
            bbox_weight = torch.masked_select(
                target_scores.sum(-1), fg_mask).unsqueeze(-1)
            loss_iou = self.iou_loss(pred_bboxes_pos,
                                     target_bboxes_pos) * bbox_weight
            if target_scores_sum > 1:
                loss_iou = loss_iou.sum() / target_scores_sum
            else:
                loss_iou = loss_iou.sum()

            # dfl loss
            if self.use_dfl:
                dist_mask = fg_mask.unsqueeze(-1).repeat(
                    [1, 1, (self.reg_max + 1) * 4])
                pred_dist_pos = torch.masked_select(
                    pred_dist, dist_mask).reshape([-1, 4, self.reg_max + 1])
                target_ltrb = bbox2dist(anchor_points, target_bboxes, self.reg_max)
                target_ltrb_pos = torch.masked_select(
                    target_ltrb, bbox_mask).reshape([-1, 4])
                loss_dfl = self._df_loss(pred_dist_pos,
                                        target_ltrb_pos) * bbox_weight
                if target_scores_sum > 1:
                    loss_dfl = loss_dfl.sum() / target_scores_sum
                else:
                    loss_dfl = loss_dfl.sum()
            else:
                loss_dfl = pred_dist.sum() * 0.

        else:
            loss_iou = pred_dist.sum() * 0.
            loss_dfl = pred_dist.sum() * 0.

        return loss_iou, loss_dfl

    def _df_loss(self, pred_dist, target):
        target_left = target.to(torch.long)
        target_right = target_left + 1
        weight_left = target_right.to(torch.float) - target
        weight_right = 1 - weight_left
        loss_left = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_left.view(-1), reduction='none').view(
            target_left.shape) * weight_left
        loss_right = F.cross_entropy(
            pred_dist.view(-1, self.reg_max + 1), target_right.view(-1), reduction='none').view(
            target_left.shape) * weight_right
        return (loss_left + loss_right).mean(-1, keepdim=True)

def dice_loss(pred,
              target,
              weight=None,
              eps=1e-3,
              reduction='mean',
              naive_dice=False,
              avg_factor=None):
    """Calculate dice loss, there are two forms of dice loss is supported:

        - the one proposed in `V-Net: Fully Convolutional Neural
            Networks for Volumetric Medical Image Segmentation
            <https://arxiv.org/abs/1606.04797>`_.
        - the dice loss in which the power of the number in the
            denominator is the first power instead of the second
            power.

    Args:
        pred (torch.Tensor): The prediction, has a shape (n, *)
        target (torch.Tensor): The learning label of the prediction,
            shape (n, *), same shape of pred.
        weight (torch.Tensor, optional): The weight of loss for each
            prediction, has a shape (n,). Defaults to None.
        eps (float): Avoid dividing by zero. Default: 1e-3.
        reduction (str, optional): The method used to reduce the loss into
            a scalar. Defaults to 'mean'.
            Options are "none", "mean" and "sum".
        naive_dice (bool, optional): If false, use the dice
                loss defined in the V-Net paper, otherwise, use the
                naive dice loss in which the power of the number in the
                denominator is the first power instead of the second
                power.Defaults to False.
        avg_factor (int, optional): Average factor that is used to average
            the loss. Defaults to None.
    """

    input = pred.flatten(1)
    target = target.flatten(1).float()

    a = torch.sum(input * target, 1)
    if naive_dice:
        b = torch.sum(input, 1)
        c = torch.sum(target, 1)
        d = (2 * a + eps) / (b + c + eps)
    else:
        b = torch.sum(input * input, 1) + eps
        c = torch.sum(target * target, 1) + eps
        d = (2 * a) / (b + c)

    loss = 1 - d
    if weight is not None:
        assert weight.ndim == loss.ndim
        assert len(weight) == len(pred)
    loss = weight_reduce_loss(loss, weight, reduction, avg_factor)
    return loss

def weight_reduce_loss(loss,
                       weight=None,
                       reduction='mean',
                       avg_factor=None):
    """Apply element-wise weight and reduce loss.

    Args:
        loss (Tensor): Element-wise loss.
        weight (Optional[Tensor], optional): Element-wise weights.
            Defaults to None.
        reduction (str, optional): Same as built-in losses of PyTorch.
            Defaults to 'mean'.
        avg_factor (Optional[float], optional): Average factor when
            computing the mean of losses. Defaults to None.

    Returns:
        Tensor: Processed loss values.
    """
    # if weight is specified, apply element-wise weight
    if weight is not None:
        loss = loss * weight

    # if avg_factor is not specified, just reduce the loss
    if avg_factor is None:
        loss = reduce_loss(loss, reduction)
    else:
        # if reduction is mean, then average the loss by avg_factor
        if reduction == 'mean':
            # Avoid causing ZeroDivisionError when avg_factor is 0.0,
            # i.e., all labels of an image belong to ignore index.
            eps = torch.finfo(torch.float32).eps
            loss = loss.sum() / (avg_factor + eps)
        # if reduction is 'none', then do nothing, otherwise raise an error
        elif reduction != 'none':
            raise ValueError('avg_factor can not be used with reduction="sum"')
    return loss

def reduce_loss(loss, reduction):
    """Reduce loss as specified.

    Args:
        loss (Tensor): Elementwise loss tensor.
        reduction (str): Options are "none", "mean" and "sum".

    Return:
        Tensor: Reduced loss tensor.
    """
    reduction_enum = F._Reduction.get_enum(reduction)
    # none: 0, elementwise_mean:1, sum: 2
    if reduction_enum == 0:
        return loss
    elif reduction_enum == 1:
        return loss.mean()
    elif reduction_enum == 2:
        return loss.sum()