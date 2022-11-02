import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.assigners.iou2d_calculator import iou2d_calculator
from yolov6.assigners.assigner_utils import dist_calculator, select_candidates_in_gts, select_highest_overlaps, iou_calculator

class ATSSAssigner(nn.Module):
    '''Adaptive Training Sample Selection Assigner'''
    def __init__(self,
                 topk=9,
                 num_classes=80):
        super(ATSSAssigner, self).__init__()
        self.topk = topk
        self.num_classes = num_classes
        self.bg_idx = num_classes

    @torch.no_grad()
    def forward(self,
                anc_bboxes,
                n_level_bboxes,
                gt_labels,
                gt_bboxes,
                mask_gt,
                pd_bboxes):
        r"""This code is based on
            https://github.com/fcjian/TOOD/blob/master/mmdet/core/bbox/assigners/atss_assigner.py

        Args:
            anc_bboxes (Tensor): shape(num_total_anchors, 4)
            n_level_bboxes (List):len(3)
            gt_labels (Tensor): shape(bs, n_max_boxes, 1)
            gt_bboxes (Tensor): shape(bs, n_max_boxes, 4)
            mask_gt (Tensor): shape(bs, n_max_boxes, 1)
            pd_bboxes (Tensor): shape(bs, n_max_boxes, 4)
        Returns:
            target_labels (Tensor): shape(bs, num_total_anchors)
            target_bboxes (Tensor): shape(bs, num_total_anchors, 4)
            target_scores (Tensor): shape(bs, num_total_anchors, num_classes)
            fg_mask (Tensor): shape(bs, num_total_anchors)
        """
        self.n_anchors = anc_bboxes.size(0)
        self.bs = gt_bboxes.size(0)
        self.n_max_boxes = gt_bboxes.size(1)

        if self.n_max_boxes == 0:
            device = gt_bboxes.device
            return torch.full( [self.bs, self.n_anchors], self.bg_idx).to(device), \
                   torch.zeros([self.bs, self.n_anchors, 4]).to(device), \
                   torch.zeros([self.bs, self.n_anchors, self.num_classes]).to(device), \
                   torch.zeros([self.bs, self.n_anchors]).to(device)


        overlaps = iou2d_calculator(gt_bboxes.reshape([-1, 4]), anc_bboxes)
        overlaps = overlaps.reshape([self.bs, -1, self.n_anchors])

        distances, ac_points = dist_calculator(gt_bboxes.reshape([-1, 4]), anc_bboxes)
        distances = distances.reshape([self.bs, -1, self.n_anchors])

        is_in_candidate, candidate_idxs = self.select_topk_candidates(
            distances, n_level_bboxes, mask_gt)

        overlaps_thr_per_gt, iou_candidates = self.thres_calculator(
            is_in_candidate, candidate_idxs, overlaps)

        # select candidates iou >= threshold as positive
        is_pos = torch.where(
            iou_candidates > overlaps_thr_per_gt.repeat([1, 1, self.n_anchors]),
            is_in_candidate, torch.zeros_like(is_in_candidate))

        is_in_gts = select_candidates_in_gts(ac_points, gt_bboxes)
        mask_pos = is_pos * is_in_gts * mask_gt

        target_gt_idx, fg_mask, mask_pos = select_highest_overlaps(
            mask_pos, overlaps, self.n_max_boxes)

        # assigned target
        target_labels, target_bboxes, target_scores = self.get_targets(
            gt_labels, gt_bboxes, target_gt_idx, fg_mask)

        # soft label with iou
        if pd_bboxes is not None:
            ious = iou_calculator(gt_bboxes, pd_bboxes) * mask_pos
            ious = ious.max(axis=-2)[0].unsqueeze(-1)
            target_scores *= ious

        return target_labels.long(), target_bboxes, target_scores, fg_mask.bool()

    def select_topk_candidates(self,
                               distances,
                               n_level_bboxes,
                               mask_gt):

        mask_gt = mask_gt.repeat(1, 1, self.topk).bool()
        level_distances = torch.split(distances, n_level_bboxes, dim=-1)
        is_in_candidate_list = []
        candidate_idxs = []
        start_idx = 0
        for per_level_distances, per_level_boxes in zip(level_distances, n_level_bboxes):

            end_idx = start_idx + per_level_boxes
            selected_k = min(self.topk, per_level_boxes)
            _, per_level_topk_idxs = per_level_distances.topk(selected_k, dim=-1, largest=False)
            candidate_idxs.append(per_level_topk_idxs + start_idx)
            per_level_topk_idxs = torch.where(mask_gt,
                per_level_topk_idxs, torch.zeros_like(per_level_topk_idxs))
            is_in_candidate = F.one_hot(per_level_topk_idxs, per_level_boxes).sum(dim=-2)
            is_in_candidate = torch.where(is_in_candidate > 1,
                torch.zeros_like(is_in_candidate), is_in_candidate)
            is_in_candidate_list.append(is_in_candidate.to(distances.dtype))
            start_idx = end_idx

        is_in_candidate_list = torch.cat(is_in_candidate_list, dim=-1)
        candidate_idxs = torch.cat(candidate_idxs, dim=-1)

        return is_in_candidate_list, candidate_idxs

    def thres_calculator(self,
                         is_in_candidate,
                         candidate_idxs,
                         overlaps):

        n_bs_max_boxes = self.bs * self.n_max_boxes
        _candidate_overlaps = torch.where(is_in_candidate > 0,
            overlaps, torch.zeros_like(overlaps))
        candidate_idxs = candidate_idxs.reshape([n_bs_max_boxes, -1])
        assist_idxs = self.n_anchors * torch.arange(n_bs_max_boxes, device=candidate_idxs.device)
        assist_idxs = assist_idxs[:,None]
        faltten_idxs = candidate_idxs + assist_idxs
        candidate_overlaps = _candidate_overlaps.reshape(-1)[faltten_idxs]
        candidate_overlaps = candidate_overlaps.reshape([self.bs, self.n_max_boxes, -1])

        overlaps_mean_per_gt = candidate_overlaps.mean(axis=-1, keepdim=True)
        overlaps_std_per_gt = candidate_overlaps.std(axis=-1, keepdim=True)
        overlaps_thr_per_gt = overlaps_mean_per_gt + overlaps_std_per_gt

        return overlaps_thr_per_gt, _candidate_overlaps

    def get_targets(self,
                    gt_labels,
                    gt_bboxes,
                    target_gt_idx,
                    fg_mask):

        # assigned target labels
        batch_idx = torch.arange(self.bs, dtype=gt_labels.dtype, device=gt_labels.device)
        batch_idx = batch_idx[...,None]
        target_gt_idx = (target_gt_idx + batch_idx * self.n_max_boxes).long()
        target_labels = gt_labels.flatten()[target_gt_idx.flatten()]
        target_labels = target_labels.reshape([self.bs, self.n_anchors])
        target_labels = torch.where(fg_mask > 0,
            target_labels, torch.full_like(target_labels, self.bg_idx))

        # assigned target boxes
        target_bboxes = gt_bboxes.reshape([-1, 4])[target_gt_idx.flatten()]
        target_bboxes = target_bboxes.reshape([self.bs, self.n_anchors, 4])

        # assigned target scores
        target_scores = F.one_hot(target_labels.long(), self.num_classes + 1).float()
        target_scores = target_scores[:, :, :self.num_classes]

        return target_labels, target_bboxes, target_scores
