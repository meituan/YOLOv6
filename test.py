from mmcv.ops import diff_iou_rotated_2d

import torch


a = torch.tensor([[[1, 1, 30, 10, 0 / 180.0 * torch.pi]]]).to("cuda:0")
b = torch.tensor([[[1, 1, 30, 10, 179 / 180.0 * torch.pi]]]).to("cuda:0")

# 179 - 180

iou = diff_iou_rotated_2d(a, b)
print(iou)
