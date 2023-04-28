#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from yolov6.layers.common import *
from yolov6.utils.torch_utils import initialize_weights
from yolov6.models.reppan import *
from yolov6.models.efficientrep import *
from yolov6.utils.events import LOGGER
from yolov6.models.heads.effidehead_lite import Detect, build_effidehead_layer

class Model(nn.Module):
    export = False
    '''YOLOv6 model with backbone, neck and head.
    The default parts are EfficientRep Backbone, Rep-PAN and
    Efficient Decoupled Head.
    '''
    def __init__(self, config, channels=3, num_classes=None):  # model, input channels, number of classes
        super().__init__()
        # Build network
        self.backbone, self.neck, self.detect = build_network(config, channels, num_classes)

        # Init Detect head
        self.stride = self.detect.stride
        self.detect.initialize_biases()

        # Init weights
        initialize_weights(self)

    def forward(self, x):
        export_mode = torch.onnx.is_in_onnx_export() or self.export
        x = self.backbone(x)
        x = self.neck(x)
        if not export_mode:
            featmaps = []
            featmaps.extend(x)
        x = self.detect(x)
        return x if export_mode or self.export is True else [x, featmaps]

    def _apply(self, fn):
        self = super()._apply(fn)
        self.detect.stride = fn(self.detect.stride)
        self.detect.grid = list(map(fn, self.detect.grid))
        return self

def build_network(config, in_channels, num_classes):
    width_mul = config.model.width_multiple

    num_repeat_backbone = config.model.backbone.num_repeats
    out_channels_backbone = config.model.backbone.out_channels
    scale_size_backbone = config.model.backbone.scale_size
    in_channels_neck = config.model.neck.in_channels
    unified_channels_neck = config.model.neck.unified_channels
    in_channels_head = config.model.head.in_channels
    num_layers = config.model.head.num_layers

    BACKBONE = eval(config.model.backbone.type)
    NECK = eval(config.model.neck.type)

    out_channels_backbone = [make_divisible(i * width_mul)
                            for i in out_channels_backbone]
    mid_channels_backbone = [make_divisible(int(i * scale_size_backbone), divisor=8)
                            for i in out_channels_backbone]
    in_channels_neck = [make_divisible(i * width_mul)
                       for i in in_channels_neck]

    backbone = BACKBONE(in_channels,
                        mid_channels_backbone,
                        out_channels_backbone,
                        num_repeat=num_repeat_backbone)
    neck = NECK(in_channels_neck, unified_channels_neck)
    head_layers = build_effidehead_layer(in_channels_head, 1, num_classes, num_layers)
    head = Detect(num_classes, num_layers, head_layers=head_layers)

    return backbone, neck, head


def build_model(cfg, num_classes, device):
    model = Model(cfg, channels=3, num_classes=num_classes).to(device)
    return model

def make_divisible(v, divisor=16):
    new_v = max(divisor, int(v + divisor / 2) // divisor * divisor)
    if new_v < 0.9 * v:
        new_v += divisor
    return new_v
