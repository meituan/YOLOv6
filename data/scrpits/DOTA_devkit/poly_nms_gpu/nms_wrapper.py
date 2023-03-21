# --------------------------------------------------------
# Fast R-CNN
# Copyright (c) 2015 Microsoft
# Licensed under The MIT License [see LICENSE for details]
# Written by Ross Girshick
# --------------------------------------------------------

# from nms.gpu_nms import gpu_nms
# from nms.cpu_nms import cpu_nms
from .poly_nms import poly_gpu_nms
def poly_nms_gpu(dets, thresh, force_cpu=False):
    """Dispatch to either CPU or GPU NMS implementations."""

    if dets.shape[0] == 0:
        return []
    return poly_gpu_nms(dets, thresh, device_id=0)

