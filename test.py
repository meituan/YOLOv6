from mmcv.ops import diff_iou_rotated_2d

import torch

s = 32
a1 = 180
a2 = 180
a = torch.tensor([[[1 * s, 1 * s, 30 * s, 10 * s, a1 / 180.0 * torch.pi]]]).to("cuda:0")
b = torch.tensor([[[1 * s, 1 * s, 30 * s, 10 * s, a2 / 180.0 * torch.pi]]]).to("cuda:0")

# 179 - 180

iou = diff_iou_rotated_2d(a, b)
print(iou)
