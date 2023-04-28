#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import time
import sys
import os
import torch
import torch.nn as nn

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.models.yolo import *
from yolov6.models.effidehead import Detect as NormDetect
from yolov6.models.heads.effidehead_lite import Detect as LiteDetect
from yolov6.layers.common import *
from yolov6.utils.events import LOGGER
from yolov6.utils.checkpoint import load_checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov6lite_s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[320, 320], help='image size, the order is: height width')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1  # expand
    print(args)
    t = time.time()

    # Check device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device(f'cuda:{args.device}' if cuda else 'cpu')
    assert not (device.type == 'cpu' and args.half), '--half only compatible with GPU export, i.e. use --device 0'
    # Load PyTorch model
    model = load_checkpoint(args.weights, map_location=device, inplace=True, fuse=True)  # load FP32 model
    # Switch export mode
    model.export = True
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()
        elif isinstance(layer, nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
            layer.recompute_scale_factor = None  # torch 1.11.0 compatibility
    # Input
    img = torch.zeros(args.batch_size, 3, *args.img_size).to(device)  # image size(1,3,320,192) iDetection

    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, (NormDetect, LiteDetect)):
            m.export = True

    print("===================")
    print(model)
    print("===================")

    y = model(img)  # dry run

    # TorchScript export
    try:
        LOGGER.info('\nStarting to export TorchScript...')
        export_file = args.weights.replace('.pt', '.torchscript')  # filename
        trace_model = torch.jit.trace(model, img)

        trace_model.save(export_file)

    except Exception as e:
        LOGGER.info(f'TorchScript export failure: {e}')

    # Finish
    LOGGER.info('\nExport complete (%.2fs)' % (time.time() - t))
