import torch
import torch.nn as nn
import torch.nn.functional as F

from yolov6.assigners.assigner_utils import (dist_calculator, iou_calculator,
                                             select_candidates_in_gts,
                                             select_highest_overlaps)


class TaskAlignedAssigner(nn.Module):
    def __init__(self, topk=13, num_classes=80, alpha=1.0, beta=6.0, eps=1e-9):
        super(TaskAlignedAssigner, self).__init__()
        self.topk = topk
        self.alpha = alpha
        self.beta = beta
        self.eps = eps
        self.num_classes = num_classes
    @torch.no_grad()
    def forward(self, pd_scores, pd_bboxes, pd_angles, anc_points, gt_labels, gt_bboxes, gt_angles, mask_gt):
        """This code referenced to
           https://github.com/Nioolek/PPYOLOE_pytorch/blob/master/ppyoloe/assigner/tal_assigner.py

        Args:
            pred_scores (Tensor, float32): predicted class probability, shape(B, L, C)
            pred_bboxes (Tensor, float32): predicted bounding boxes, shape(B, L, 5)
            anchor_points (Tensor, float32): pre-defined anchors, shape(1, L, 2), "cxcy" format
            num_anchors_list (List): num of anchors in each level, shape(L)
            gt_labels (Tensor, int64|int32): Label of gt_bboxes, shape(B, n, 1)
            gt_bboxes (Tensor, float32): Ground truth bboxes, shape(B, n, 5)
            pad_gt_mask (Tensor, float32): 1 means bbox, 0 means no bbox, shape(B, n, 1)
            bg_index (int): background index
            gt_scores (Tensor|None, float32) Score of gt_bboxes, shape(B, n, 1)
        Returns:
            assigned_labels (Tensor): (B, L)
            assigned_bboxes (Tensor): (B, L, 5)
            assigned_scores (Tensor): (B, L, C)
            fg_mask bool
        """
        self.bs = pred_scores.shape[0]
        self.n_max_boxes = gt_bboxes.size(1)
        assert pred_scores.ndim == pred_bboxes.ndim
        assert gt_labels.ndim == gt_bboxes.ndim and \
               gt_bboxes.ndim == 3
        batch_size, num_anchors, num_classes = pred_scores.shape
        _, num_max_boxes, _ = gt_bboxes.shape
        if num_max_boxes == 0:
            device = gt_bboxes.device
            return (
                torch.full_like(pd_scores[..., 0], self.bg_idx).to(device),
                torch.zeros_like(pd_bboxes).to(device),
                torch.zeros_like(pd_angles).to(device),
                torch.zeros_like(pd_scores).to(device),
                torch.zeros_like(pd_scores[..., 0]).to(device),
            )

        cycle, step, self.bs = (1, self.bs, self.bs) if self.n_max_boxes <= 100 else (self.bs, 1, 1)
        target_labels_lst, target_bboxes_lst, target_scores_lst, fg_mask_lst = [], [], [], []
        # loop batch dim in case of numerous object box
        for i in range(cycle):  # TODO change this
            start, end = i * step, (i + 1) * step
            pd_scores_ = pd_scores[start:end, ...]
            pd_bboxes_ = pd_bboxes[start:end, ...]
            gt_labels_ = gt_labels[start:end, ...]
            gt_bboxes_ = gt_bboxes[start:end, ...]
            gt_angles_ = gt_angles[start:end, ...]
            mask_gt_ = mask_gt[start:end, ...]

            mask_pos, align_metric, overlaps = self.get_pos_mask(
                pd_scores_, pd_bboxes_, gt_labels_, gt_bboxes_, anc_points, mask_gt_
            )

            target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(mask_pos, overlaps, self.n_max_boxes)

            # assigned target
            target_labels, target_bboxes, target_angles, target_scores = self.get_targets(
                gt_labels_, gt_bboxes_, gt_angles_, target_gt_idx, fg_mask
            )

            # normalize
            align_metric *= mask_pos
            pos_align_metrics = align_metric.max(axis=-1, keepdim=True)[0]
            pos_overlaps = (overlaps * mask_pos).max(axis=-1, keepdim=True)[0]
            norm_align_metric = (align_metric * pos_overlaps / (pos_align_metrics + self.eps)).max(-2)[0].unsqueeze(-1)
            target_scores = target_scores * norm_align_metric

            # append
            target_labels_lst.append(target_labels)
            target_bboxes_lst.append(target_bboxes)
            target_scores_lst.append(target_scores)
            fg_mask_lst.append(fg_mask)

        # concat
        target_labels = torch.cat(target_labels_lst, 0)
        target_bboxes = torch.cat(target_bboxes_lst, 0)
        target_scores = torch.cat(target_scores_lst, 0)
        fg_mask = torch.cat(fg_mask_lst, 0)

        return target_labels, target_bboxes, target_angles, target_scores, fg_mask.bool()

    def get_pos_mask(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes, anc_points, mask_gt):

        # get anchor_align metric
        align_metric, overlaps = self.get_box_metrics(pd_scores, pd_bboxes, gt_labels, gt_bboxes)
        # get in_gts mask
        mask_in_gts = select_candidates_in_gts(anc_points, gt_bboxes)
        # get topk_metric mask
        mask_topk = self.select_topk_candidates(
            align_metric * mask_in_gts, topk_mask=mask_gt.repeat([1, 1, self.topk]).bool()
        )
        # merge all mask to a final mask
        mask_pos = mask_topk * mask_in_gts * mask_gt
        return mask_pos, align_metric, overlaps

    def get_box_metrics(self, pd_scores, pd_bboxes, gt_labels, gt_bboxes):

        pd_scores = pd_scores.permute(0, 2, 1)
        gt_labels = gt_labels.to(torch.long)
        ind = torch.zeros([2, self.bs, self.n_max_boxes], dtype=torch.long)
        ind[0] = torch.arange(end=self.bs).view(-1, 1).repeat(1, self.n_max_boxes)
        ind[1] = gt_labels.squeeze(-1)
        bbox_scores = pred_scores[ind[0], ind[1]]
        overlaps = rotated_iou_similarity(gt_bboxes, pred_bboxes)
        align_metric = bbox_scores.pow(self.alpha) * overlaps.pow(self.beta)
        return align_metric, overlaps

    def select_topk_candidates(self, metrics, largest=True, topk_mask=None):

        num_anchors = metrics.shape[-1]
        topk_metrics, topk_idxs = torch.topk(metrics, self.topk, axis=-1, largest=largest)
        if topk_mask is None:
            topk_mask = (topk_metrics.max(axis=-1, keepdim=True) > self.eps).tile([1, 1, self.topk])
        topk_idxs = torch.where(topk_mask, topk_idxs, torch.zeros_like(topk_idxs))
        is_in_topk = F.one_hot(topk_idxs, num_anchors).sum(axis=-2)
        is_in_topk = torch.where(is_in_topk > 1, torch.zeros_like(is_in_topk), is_in_topk)
        return is_in_topk.to(metrics.dtype)

    def get_targets(self, gt_labels, gt_bboxes, gt_angles, target_gt_idx, fg_mask):

        # assigned target labels
        batch_ind = torch.arange(end=self.bs, dtype=torch.int64, device=gt_labels.device)[..., None]
        target_gt_idx = target_gt_idx + batch_ind * self.n_max_boxes
        target_labels = gt_labels.long().flatten()[target_gt_idx]

        # assigned target boxes
        target_bboxes = gt_bboxes.reshape([-1, 5])[target_gt_idx]

        # assigned target scores
        target_labels[target_labels < 0] = 0
        target_scores = F.one_hot(target_labels, self.num_classes)
        fg_scores_mask = fg_mask[:, :, None].repeat(1, 1, self.num_classes)
        target_scores = torch.where(fg_scores_mask > 0, target_scores, torch.full_like(target_scores, 0))

        # assigned angle boxes
        target_angles = gt_angles.reshape([-1, 1])[target_gt_idx]

        return target_labels, target_bboxes, target_angles, target_scores
