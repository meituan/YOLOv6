#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import time
import sys
import os
import torch
import torch.nn as nn
import onnx

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.models.yolo import *
from yolov6.models.effidehead import Detect
from yolov6.layers.common import *
from yolov6.utils.events import LOGGER
from yolov6.utils.checkpoint import load_checkpoint
from io import BytesIO


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov6s.pt', help='weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size, the order is: height width')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--simplify', action='store_true', help='simplify onnx model')
    parser.add_argument('--dynamic-batch', action='store_true', help='export dynamic batch onnx model')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--trt-version', type=int, default=8, help='tensorrt version')
    parser.add_argument('--ort', action='store_true', help='export onnx for onnxruntime')
    parser.add_argument('--with-preprocess', action='store_true', help='export bgr2rgb and normalize')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.5, help='conf threshold for NMS')
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
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()
        elif isinstance(layer, nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
            layer.recompute_scale_factor = None  # torch 1.11.0 compatibility
    # Input
    img = torch.zeros(args.batch_size, 3, *args.img_size).to(device)  # image size(1,3,320,192) iDetection

    # Update model
    if args.half:
        img, model = img.half(), model.half()  # to FP16
    model.eval()
    for k, m in model.named_modules():
        if isinstance(m, ConvModule):  # assign export-friendly activations
            if hasattr(m, 'act') and isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = args.inplace
    dynamic_axes = None
    if args.dynamic_batch:
        args.batch_size = 'batch'
        dynamic_axes = {
            'images' :{
                0:'batch',
            },}
        if args.end2end:
            output_axes = {
                'num_dets': {0: 'batch'},
                'det_boxes': {0: 'batch'},
                'det_scores': {0: 'batch'},
                'det_classes': {0: 'batch'},
            }
        else:
            output_axes = {
                'outputs': {0: 'batch'},
            }
        dynamic_axes.update(output_axes)


    if args.end2end:
        from yolov6.models.end2end import End2End
        model = End2End(model, max_obj=args.topk_all, iou_thres=args.iou_thres,score_thres=args.conf_thres,
                        device=device, ort=args.ort, trt_version=args.trt_version, with_preprocess=args.with_preprocess)

    print("===================")
    print(model)
    print("===================")

    y = model(img)  # dry run

    # ONNX export
    try:
        LOGGER.info('\nStarting to export ONNX...')
        export_file = args.weights.replace('.pt', '.onnx')  # filename
        with BytesIO() as f:
            torch.onnx.export(model, img, f, verbose=False, opset_version=13,
                              training=torch.onnx.TrainingMode.EVAL,
                              do_constant_folding=True,
                              input_names=['images'],
                              output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                              if args.end2end else ['outputs'],
                              dynamic_axes=dynamic_axes)
            f.seek(0)
            # Checks
            onnx_model = onnx.load(f)  # load onnx model
            onnx.checker.check_model(onnx_model)  # check onnx model
            # Fix output shape
            if args.end2end and not args.ort:
                shapes = [args.batch_size, 1, args.batch_size, args.topk_all, 4,
                          args.batch_size, args.topk_all, args.batch_size, args.topk_all]
                for i in onnx_model.graph.output:
                    for j in i.type.tensor_type.shape.dim:
                        j.dim_param = str(shapes.pop(0))
        if args.simplify:
            try:
                import onnxsim
                LOGGER.info('\nStarting to simplify ONNX...')
                onnx_model, check = onnxsim.simplify(onnx_model)
                assert check, 'assert check failed'
            except Exception as e:
                LOGGER.info(f'Simplifier failure: {e}')
        onnx.save(onnx_model, export_file)
        LOGGER.info(f'ONNX export success, saved as {export_file}')
    except Exception as e:
        LOGGER.info(f'ONNX export failure: {e}')

    # Finish
    LOGGER.info('\nExport complete (%.2fs)' % (time.time() - t))
    if args.end2end:
        if not args.ort:
            info = f'trtexec --onnx={export_file} --saveEngine={export_file.replace(".onnx",".engine")}'
            if args.dynamic_batch:
                LOGGER.info('Dynamic batch export should define min/opt/max batchsize\n'+
                            'We set min/opt/max = 1/16/32 default!')
                wandh = 'x'.join(list(map(str,args.img_size)))
                info += (f' --minShapes=images:1x3x{wandh}'+
                f' --optShapes=images:16x3x{wandh}'+
                f' --maxShapes=images:32x3x{wandh}'+
                f' --shapes=images:16x3x{wandh}')
            LOGGER.info('\nYou can export tensorrt engine use trtexec tools.\nCommand is:')
            LOGGER.info(info)
