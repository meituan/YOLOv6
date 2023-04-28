import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from yolov6.layers.common import DPBlock
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox


class Detect(nn.Module):
    export = False
    '''Efficient Decoupled Head
    With hardware-aware degisn, the decoupled head is optimized with
    hybridchannels methods.
    '''
    def __init__(self, num_classes=80, num_layers=3, inplace=True, head_layers=None):  # detection layer
        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64] # strides computed during build
        self.stride = torch.tensor(stride)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0

        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i*5
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx+1])
            self.reg_convs.append(head_layers[idx+2])
            self.cls_preds.append(head_layers[idx+3])
            self.reg_preds.append(head_layers[idx+4])

    def initialize_biases(self):

        for conv in self.cls_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.reg_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

    def forward(self, x):
        if self.training:
            cls_score_list = []
            reg_distri_list = []

            for i in range(self.nl):
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_distri_list.append(reg_output.flatten(2).permute((0, 2, 1)))

            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_distri_list = torch.cat(reg_distri_list, axis=1)

            return x, cls_score_list, reg_distri_list
        else:
            cls_score_list = []
            reg_dist_list = []

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = torch.sigmoid(cls_output)

                if self.export:
                    cls_score_list.append(cls_output)
                    reg_dist_list.append(reg_output)
                else:
                    cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                    reg_dist_list.append(reg_output.reshape([b, 4, l]))


            if self.export:
                return tuple(torch.cat([cls, reg], 1) for cls, reg in zip(cls_score_list, reg_dist_list))

            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)


            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode='af')

            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
            return torch.cat(
                [
                    pred_bboxes,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list
                ],
                axis=-1)

def build_effidehead_layer(channels_list, num_anchors, num_classes, num_layers):

    head_layers = nn.Sequential(
        # stem0
        DPBlock(
            in_channel=channels_list[0],
            out_channel=channels_list[0],
            kernel_size=5,
            stride=1
        ),
        # cls_conv0
        DPBlock(
            in_channel=channels_list[0],
            out_channel=channels_list[0],
            kernel_size=5,
            stride=1
        ),
        # reg_conv0
        DPBlock(
            in_channel=channels_list[0],
            out_channel=channels_list[0],
            kernel_size=5,
            stride=1
        ),
        # cls_pred0
        nn.Conv2d(
            in_channels=channels_list[0],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred0
        nn.Conv2d(
            in_channels=channels_list[0],
            out_channels=4 * num_anchors,
            kernel_size=1
        ),
        # stem1
        DPBlock(
            in_channel=channels_list[1],
            out_channel=channels_list[1],
            kernel_size=5,
            stride=1
        ),
        # cls_conv1
        DPBlock(
            in_channel=channels_list[1],
            out_channel=channels_list[1],
            kernel_size=5,
            stride=1
        ),
        # reg_conv1
        DPBlock(
            in_channel=channels_list[1],
            out_channel=channels_list[1],
            kernel_size=5,
            stride=1
        ),
        # cls_pred1
        nn.Conv2d(
            in_channels=channels_list[1],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred1
        nn.Conv2d(
            in_channels=channels_list[1],
            out_channels=4 * num_anchors,
            kernel_size=1
        ),
        # stem2
        DPBlock(
            in_channel=channels_list[2],
            out_channel=channels_list[2],
            kernel_size=5,
            stride=1
        ),
        # cls_conv2
        DPBlock(
            in_channel=channels_list[2],
            out_channel=channels_list[2],
            kernel_size=5,
            stride=1
        ),
        # reg_conv2
        DPBlock(
            in_channel=channels_list[2],
            out_channel=channels_list[2],
            kernel_size=5,
            stride=1
        ),
        # cls_pred2
        nn.Conv2d(
            in_channels=channels_list[2],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred2
        nn.Conv2d(
            in_channels=channels_list[2],
            out_channels=4 * num_anchors,
            kernel_size=1
        )
    )

    if num_layers == 4:
        head_layers.add_module('stem3',
            # stem3
            DPBlock(
                in_channel=channels_list[3],
                out_channel=channels_list[3],
                kernel_size=5,
                stride=1
            )
        )
        head_layers.add_module('cls_conv3',
            # cls_conv3
            DPBlock(
                in_channel=channels_list[3],
                out_channel=channels_list[3],
                kernel_size=5,
                stride=1
            )
        )
        head_layers.add_module('reg_conv3',
            # reg_conv3
            DPBlock(
                in_channel=channels_list[3],
                out_channel=channels_list[3],
                kernel_size=5,
                stride=1
            )
        )
        head_layers.add_module('cls_pred3',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[3],
                out_channels=num_classes * num_anchors,
                kernel_size=1
            )
         )
        head_layers.add_module('reg_pred3',
            # reg_pred3
            nn.Conv2d(
                in_channels=channels_list[3],
                out_channels=4 * num_anchors,
                kernel_size=1
            )
        )

    return head_layers
