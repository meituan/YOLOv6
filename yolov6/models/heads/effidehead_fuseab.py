import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from yolov6.layers.common import *
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox


class Detect(nn.Module):
    '''Efficient Decoupled Head for fusing anchor-base branches.
    '''
    def __init__(self, num_classes=80, anchors=None, num_layers=3, inplace=True, head_layers=None, use_dfl=True, reg_max=16):  # detection layer
        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5 + 10  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32] if num_layers == 3 else [8, 16, 32, 64] # strides computed during build
        self.stride = torch.tensor(stride)
        self.use_dfl = use_dfl
        self.reg_max = reg_max
        self.proj_conv = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0
        self.anchors_init= ((torch.tensor(anchors) / self.stride[:,None])).reshape(self.nl, self.na, 2)

        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.cls_preds_ab = nn.ModuleList()
        self.reg_preds_ab = nn.ModuleList()

        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i*7
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx+1])
            self.reg_convs.append(head_layers[idx+2])
            self.cls_preds.append(head_layers[idx+3])
            self.reg_preds.append(head_layers[idx+4])
            self.cls_preds_ab.append(head_layers[idx+5])
            self.reg_preds_ab.append(head_layers[idx+6])

    def initialize_biases(self):

        for conv in self.cls_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        for conv in self.cls_preds_ab:
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
        
        for conv in self.reg_preds_ab:
            b = conv.bias.view(-1, )
            b.data.fill_(1.0)
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)
        
        self.proj = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.proj_conv.weight = nn.Parameter(self.proj.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)

    def forward(self, x):
        if self.training:
            device = x[0].device
            cls_score_list = []
            reg_dist_list = []
            cls_score_list_ab = []
            reg_dist_list_ab = []

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w

                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]

                cls_feat = self.cls_convs[i](cls_x)
                reg_feat = self.reg_convs[i](reg_x)

                #anchor_base
                cls_output_ab = self.cls_preds_ab[i](cls_feat)
                reg_output_ab = self.reg_preds_ab[i](reg_feat)

                cls_output_ab = torch.sigmoid(cls_output_ab)
                cls_output_ab = cls_output_ab.reshape(b, self.na, -1, h, w).permute(0,1,3,4,2)
                cls_score_list_ab.append(cls_output_ab.flatten(1,3))

                reg_output_ab = reg_output_ab.reshape(b, self.na, -1, h, w).permute(0,1,3,4,2)
                reg_output_ab[..., 2:4] = ((reg_output_ab[..., 2:4].sigmoid() * 2) ** 2 ) * (self.anchors_init[i].reshape(1, self.na, 1, 1, 2).to(device))
                reg_dist_list_ab.append(reg_output_ab.flatten(1,3))

                #anchor_free
                cls_output = self.cls_preds[i](cls_feat)
                reg_output = self.reg_preds[i](reg_feat)

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))
                reg_dist_list.append(reg_output.flatten(2).permute((0, 2, 1)))
                
            
            cls_score_list_ab = torch.cat(cls_score_list_ab, axis=1)
            reg_dist_list_ab = torch.cat(reg_dist_list_ab, axis=1)
            cls_score_list = torch.cat(cls_score_list, axis=1)
            reg_dist_list = torch.cat(reg_dist_list, axis=1)
            
            return x, cls_score_list_ab, reg_dist_list_ab, cls_score_list, reg_dist_list

        else:
            device = x[0].device
            cls_score_list = []
            reg_dist_list = []
            reg_ldmk_list = []

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w

                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]

                cls_feat = self.cls_convs[i](cls_x)
                reg_feat = self.reg_convs[i](reg_x)

                #anchor_free
                cls_output = self.cls_preds[i](cls_feat)
                reg_output = self.reg_preds[i](reg_feat)
                reg_output_box = reg_output[:, :-10, :, :]
                reg_output_ldmk = reg_output[:, -10:, :, :]

                if self.use_dfl:
                    reg_output_box = reg_output_box.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output_box = self.proj_conv(F.softmax(reg_output_box, dim=1))

                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output_box.reshape([b, 4, l]))
                reg_ldmk_list.append(reg_output_ldmk.reshape([b, 10, l]))
                
            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)
            reg_ldmk_list = torch.cat(reg_ldmk_list, axis=-1).permute(0, 2, 1)
            

            #anchor_free
            anchor_points, stride_tensor = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode='af')

            pred_bboxes = dist2bbox(reg_dist_list, anchor_points, box_format='xywh')
            pred_bboxes *= stride_tensor
            
            ldmk1 = (reg_ldmk_list[..., 0:2] + anchor_points) * stride_tensor
            ldmk2 = (reg_ldmk_list[..., 2:4] + anchor_points) * stride_tensor
            ldmk3 = (reg_ldmk_list[..., 4:6] + anchor_points) * stride_tensor
            ldmk4 = (reg_ldmk_list[..., 6:8] + anchor_points) * stride_tensor
            ldmk5 = (reg_ldmk_list[..., 8:10] + anchor_points) * stride_tensor

            return torch.cat(
                [
                    pred_bboxes,
                    ldmk1, ldmk2, ldmk3, ldmk4, ldmk5, 
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list
                ],
                axis=-1)


def EffiDeHead(channels_list, num_anchors, num_classes, reg_max=16, num_layers=3):

    chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]
    
    head_layers = nn.Sequential(
        # stem0
        Conv(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv0
        Conv(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv0
        Conv(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred0_af
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=num_classes,
            kernel_size=1
        ),
        # reg_pred0_af
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=4 * (reg_max + 1) + 10,
            kernel_size=1
        ),
        # cls_pred0_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred0_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=4 * num_anchors,
            kernel_size=1
        ),
        # stem1
        Conv(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv1
        Conv(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv1
        Conv(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred1_af
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=num_classes,
            kernel_size=1
        ),
        # reg_pred1_af
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=4 * (reg_max + 1) + 10,
            kernel_size=1
        ),
        # cls_pred1_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred1_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=4 * num_anchors,
            kernel_size=1
        ),
        # stem2
        Conv(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv2
        Conv(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv2
        Conv(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred2_af
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=num_classes,
            kernel_size=1
        ),
        # reg_pred2_af
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=4 * (reg_max + 1) + 10,
            kernel_size=1
        ),
        # cls_pred2_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred2_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=4 * num_anchors,
            kernel_size=1
        ),
    )

    if num_layers == 4:
        head_layers.add_module('stem3',
            # stem3
            Conv(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=1,
                stride=1
            )
        )
        head_layers.add_module('cls_conv3',
            # cls_conv3
            Conv(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=3,
                stride=1
            )
        )
        head_layers.add_module('reg_conv3',
            # reg_conv3
            Conv(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[3]],
                kernel_size=3,
                stride=1
            )
        )
        head_layers.add_module('cls_pred3',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=num_classes,
                kernel_size=1
            )
        )
        head_layers.add_module('reg_pred3',
            # reg_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=4 * (reg_max + 1) + 10,
                kernel_size=1
            )
        )
        head_layers.add_module('cls_pred3_3ab',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=num_classes * num_anchors,
                kernel_size=1
            )
        )
        head_layers.add_module('reg_pred3_3ab',
            # reg_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[3]],
                out_channels=4 * num_anchors,
                kernel_size=1
            )
        )

    return head_layers


def DeHead(channels_list, num_anchors, num_classes, reg_max=16, num_layers=3):

    chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]

    head_layers = nn.Sequential(
        # stem0
        Conv(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[-2]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv0
        Conv(
            in_channels=channels_list[chx[-2]],
            out_channels=channels_list[chx[-2]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv0
        Conv(
            in_channels=channels_list[chx[-2]],
            out_channels=channels_list[chx[-2]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred0_af
        nn.Conv2d(
            in_channels=channels_list[chx[-2]],
            out_channels=num_classes,
            kernel_size=1
        ),
        # reg_pred0_af
        nn.Conv2d(
            in_channels=channels_list[chx[-2]],
            out_channels=4 * (reg_max + 1) + 10,
            kernel_size=1
        ),
        # cls_pred0_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[-2]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred0_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[-2]],
            out_channels=4 * num_anchors,
            kernel_size=1
        ),
        # stem1
        Conv(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[-2]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv1
        Conv(
            in_channels=channels_list[chx[-2]],
            out_channels=channels_list[chx[-2]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv1
        Conv(
            in_channels=channels_list[chx[-2]],
            out_channels=channels_list[chx[-2]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred1_af
        nn.Conv2d(
            in_channels=channels_list[chx[-2]],
            out_channels=num_classes,
            kernel_size=1
        ),
        # reg_pred1_af
        nn.Conv2d(
            in_channels=channels_list[chx[-2]],
            out_channels=4 * (reg_max + 1) + 10,
            kernel_size=1
        ),
        # cls_pred1_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[-2]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred1_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[-2]],
            out_channels=4 * num_anchors,
            kernel_size=1
        ),
        # stem2
        Conv(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[-2]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv2
        Conv(
            in_channels=channels_list[chx[-2]],
            out_channels=channels_list[chx[-2]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv2
        Conv(
            in_channels=channels_list[chx[-2]],
            out_channels=channels_list[chx[-2]],
            kernel_size=3,
            stride=1
        ),
        # cls_pred2_af
        nn.Conv2d(
            in_channels=channels_list[chx[-2]],
            out_channels=num_classes,
            kernel_size=1
        ),
        # reg_pred2_af
        nn.Conv2d(
            in_channels=channels_list[chx[-2]],
            out_channels=4 * (reg_max + 1) + 10,
            kernel_size=1
        ),
        # cls_pred2_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[-2]],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred2_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[-2]],
            out_channels=4 * num_anchors,
            kernel_size=1
        ),
    )

    if num_layers == 4:
        head_layers.add_module('stem3',
            # stem3
            Conv(
                in_channels=channels_list[chx[3]],
                out_channels=channels_list[chx[-2]],
                kernel_size=1,
                stride=1
            )
        )
        head_layers.add_module('cls_conv3',
            # cls_conv3
            Conv(
                in_channels=channels_list[chx[-2]],
                out_channels=channels_list[chx[-2]],
                kernel_size=3,
                stride=1
            )
        )
        head_layers.add_module('reg_conv3',
            # reg_conv3
            Conv(
                in_channels=channels_list[chx[-2]],
                out_channels=channels_list[chx[-2]],
                kernel_size=3,
                stride=1
            )
        )
        head_layers.add_module('cls_pred3',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[-2]],
                out_channels=num_classes,
                kernel_size=1
            )
        )
        head_layers.add_module('reg_pred3',
            # reg_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[-2]],
                out_channels=4 * (reg_max + 1) + 10,
                kernel_size=1
            )
        )
        head_layers.add_module('cls_pred3_3ab',
            # cls_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[-2]],
                out_channels=num_classes * num_anchors,
                kernel_size=1
            )
        )
        head_layers.add_module('reg_pred3_3ab',
            # reg_pred3
            nn.Conv2d(
                in_channels=channels_list[chx[-2]],
                out_channels=4 * num_anchors,
                kernel_size=1
            )
        )

    return head_layers