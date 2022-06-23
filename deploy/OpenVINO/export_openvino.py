#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import time
import sys
import os
import torch
import torch.nn as nn
import onnx
import subprocess

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.models.yolo import *
from yolov6.models.effidehead import Detect
from yolov6.layers.common import *
from yolov6.utils.events import LOGGER
from yolov6.utils.checkpoint import load_checkpoint


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov6s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1  # expand
    print(args)
    t = time.time()

    # Check device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    assert not (device.type == 'cpu' and args.half), '--half only compatible with GPU export, i.e. use --device 0'
    # Load PyTorch model
    model = load_checkpoint(args.weights, map_location=device, inplace=True, fuse=True)  # load FP32 model
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()

    # Input
    img = torch.zeros(args.batch_size, 3, *args.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    if args.half:
        img, model = img.half(), model.half()  # to FP16
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = args.inplace

    y = model(img)  # dry run

    # ONNX export
    try:
        LOGGER.info('\nStarting to export ONNX...')
        export_file = args.weights.replace('.pt', '.onnx')  # filename
        torch.onnx.export(model, img, export_file, verbose=False, opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=['image_arrays'],
                          output_names=['outputs'],
                         )

        # Checks
        onnx_model = onnx.load(export_file)  # load onnx model
        onnx.checker.check_model(onnx_model)  # check onnx model
        LOGGER.info(f'ONNX export success, saved as {export_file}')
    except Exception as e:
        LOGGER.info(f'ONNX export failure: {e}')

    # OpenVINO export
    try:
        LOGGER.info('\nStarting to export OpenVINO...')
        import_file = args.weights.replace('.pt', '.onnx')
        export_dir = str(import_file).replace('.onnx', '_openvino')
        cmd = f"mo --input_model {import_file} --output_dir {export_dir} --data_type {'FP16' if args.half else 'FP32'}"
        subprocess.check_output(cmd.split())
        LOGGER.info(f'OpenVINO export success, saved as {export_dir}')
    except Exception as e:
        LOGGER.info(f'OpenVINO export failure: {e}')

    # Finish
    LOGGER.info('\nExport complete (%.2fs)' % (time.time() - t))
