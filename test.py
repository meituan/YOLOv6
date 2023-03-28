from mmcv.ops import diff_iou_rotated_2d

import torch


a = torch.tensor([[[68.83703, 72.36289, 27.58253, 5.13454, 1.86161]]]).to("cuda:0")
b = torch.tensor([[[69.31250, 73.12500, 15.12500, 6.18750, 0.28749]]]).to("cuda:0")
print(a.dtype)
print(b.dtype)

iou = diff_iou_rotated_2d(a, b)
print(iou)
