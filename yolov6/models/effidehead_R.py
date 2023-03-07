import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from yolov6.layers.common import *
from yolov6.assigners.anchor_generator import generate_anchors_OBB
from yolov6.utils.general import dist2bbox

class Detect_OBB(nn.Module):
    def __init__(self, num_classes=15, anchors=1, num_layers=3, inplace=True, head_layers=None, use_reg_dfl=True, reg_max=16, use_angle_dfl=True,angle_max=90,trt=False,export_onnx=False):  # detection layer
        super().__init__()
        assert head_layers is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors
        self.grid = [torch.zeros(1)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = inplace
        stride = [8, 16, 32]  # strides computed during build
        self.stride = torch.tensor(stride)
        self.use_reg_dfl = use_reg_dfl
        self.reg_max = reg_max
        self.use_angle_dfl = use_angle_dfl
        self.angle_max = angle_max
        self.proj_reg = nn.Conv2d(self.reg_max + 1, 1, 1, bias=False)
        self.proj_angle = nn.Conv2d(self.angle_max + 1, 1, 1, bias=False)
        self.grid_cell_offset = 0.5
        self.grid_cell_size = 5.0 
        self.half_pi = torch.tensor(
            [1.5707963267948966],dtype=torch.float32)
        self.half_pi_bin = self.half_pi/self.angle_max
        # Model Deployment
        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.ori_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.ori_preds = nn.ModuleList()
        # Efficient Decoupled Head (pred conv ori)
        for i in range(num_layers):
            idx = i*(5+2)
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx+1])
            self.reg_convs.append(head_layers[idx+2])
            self.ori_convs.append(head_layers[idx+3])
            self.cls_preds.append(head_layers[idx+4])
            self.reg_preds.append(head_layers[idx+5])
            self.ori_preds.append(head_layers[idx+6])
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
        
        for conv in self.ori_preds:
            b = conv.bias.view(-1, )
            # TODO initial_bias
            import random
            b.data.fill_(random.random())
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        self.reg_para = nn.Parameter(torch.linspace(0, self.reg_max, self.reg_max + 1), requires_grad=False)
        self.angle_para = nn.Parameter(torch.linspace(0, self.angle_max, self.angle_max + 1), requires_grad=False) * self.half_pi_bin
        self.proj_reg.weight = nn.Parameter(self.reg_para.view([1, self.reg_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)
        self.proj_angle.weight = nn.Parameter(self.angle_para.view([1, self.angle_max + 1, 1, 1]).clone().detach(),
                                                   requires_grad=False)
    
    # def obb_decode(self, points, pred_dist, pred_angle, stride_tensor):
    #     b,l = pred_angle.shape[:2]
    #     xy,wh = torch.split(pred_dist, 2, axis = -1)
    #     xy = xy*stride_tensor + points
    #     wh = (F.elu(wh)+1.)*stride_tensor
    #     if self.use_angle_dfl:
    #         angle = F.softmax(pred_angle,l,1,self.angle_max+1).matmul(self.angle_proj)
    #     return torch.concat([xy, wh, angle])
    def forward(self,x):
        if self.training:
            cls_score_list = []
            reg_dist_list = []
            ori_dist_list = []
            for i in range(self.nl):
                x[i] = self.stems[i](x[i])# neck conv+bn+silu 
                cls_x = x[i]  
                reg_x = x[i]
                ori_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)
                ori_feat = self.ori_convs[i](ori_x)
                ori_output = self.ori_preds[i](ori_feat)
                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.flatten(2).permute((0, 2, 1)))               
                reg_dist_list.append(reg_output.flatten(2).permute((0, 2, 1)))
                ori_dist_list.append(ori_output.flatten(2).permute((0, 2, 1)))
            cls_score_list = torch.cat(cls_score_list,axis=1)
            reg_dist_list = torch.cat(reg_dist_list,axis=1)
            ori_dist_list = torch.cat(ori_dist_list,axis=1)
            if(torch.onnx.is_in_onnx_export()):
                return cls_score_list, reg_dist_list, ori_dist_list
            else:
                return x,cls_score_list,reg_dist_list,ori_dist_list
        else:
            anchor_points, stride_tensor  = generate_anchors_OBB(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True)
            cls_score_list = []
            reg_dist_list = []
            ori_dist_list = []
            for i,stride in enumerate(self.stride):
                b, _, h, w = x[i].shape
                l = h * w
                x[i] = self.stems[i](x[i])
                cls_x = x[i]
                reg_x = x[i]
                ori_x = x[i]
                cls_feat = self.cls_convs[i](cls_x)
                cls_output = self.cls_preds[i](cls_feat)
                reg_feat = self.reg_convs[i](reg_x)
                reg_output = self.reg_preds[i](reg_feat)
                ori_feat = self.ori_convs[i](ori_x)
                ori_output = self.ori_preds[i](ori_feat)
                if self.use_reg_dfl:
                    reg_output = reg_output.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output = self.proj_reg(F.softmax(reg_output, dim=1))
                if self.use_angle_dfl:
                    ori_output = ori_output.reshape([-1, 1, self.angle_max + 1,l]).permute(0, 2, 1, 3)
                    ori_output = self.proj_angle(F.softmax(ori_output, dim=1))
                cls_output = torch.sigmoid(cls_output)
                cls_score_list.append(cls_output.reshape([b, self.nc, l]))
                reg_dist_list.append(reg_output.reshape([b, 4, l]))
                ori_dist_list.append(ori_output.reshape([b, 1, l]))
                
            cls_score_list = torch.cat(cls_score_list, axis=-1).permute(0, 2, 1)
            reg_dist_list = torch.cat(reg_dist_list, axis=-1).permute(0, 2, 1)
            ori_dist_list = torch.cat(ori_dist_list, axis=-1).permute(0, 2, 1)
            """
                需要进行解码的操作
            """
            reg_box_list = dist2bbox(reg_dist_list, anchor_points, box_format="xywh")
            reg_box_list =  reg_box_list * stride_tensor
            reg_conf_list = torch.ones((b, reg_box_list.shape[1], 1), device=reg_box_list.device, dtype=reg_box_list.dtype)
            return torch.concat([reg_box_list, ori_dist_list, reg_conf_list,cls_score_list], dim=-1)
        
            

def build_effidehead_layer_OBB(channels_list, num_anchors, num_classes, reg_max=16, angle_max = 16):
    
    head_layers = nn.Sequential(
        # stem0
        Conv(
            in_channels=channels_list[6],
            out_channels=channels_list[6],
            kernel_size=1,
            stride=1
        ),
        # cls_conv0
        Conv(
            in_channels=channels_list[6],
            out_channels=channels_list[6],
            kernel_size=3,
            stride=1
        ),
        # reg_conv0
        Conv(
            in_channels=channels_list[6],
            out_channels=channels_list[6],
            kernel_size=3,
            stride=1
        ),
        # ori_conv0
        Conv(
            in_channels=channels_list[6],
            out_channels=channels_list[6],
            kernel_size=3,
            stride=1
        ),
        # cls_pred0
        nn.Conv2d(
            in_channels=channels_list[6],
            out_channels= num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred0
        nn.Conv2d(
            in_channels=channels_list[6],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # ori_pred0
        nn.Conv2d(
            in_channels=channels_list[6],
            out_channels=1 * (num_anchors + angle_max),
            kernel_size=1
        ),
        # stem1
        Conv(
            in_channels=channels_list[8],
            out_channels=channels_list[8],
            kernel_size=1,
            stride=1
        ),
        # cls_conv1
        Conv(
            in_channels=channels_list[8],
            out_channels=channels_list[8],
            kernel_size=3,
            stride=1
        ),
        # reg_conv1
        Conv(
            in_channels=channels_list[8],
            out_channels=channels_list[8],
            kernel_size=3,
            stride=1
        ),
        # ori_conv1
        Conv(
            in_channels=channels_list[8],
            out_channels=channels_list[8],
            kernel_size=3,
            stride=1
        ),
        # cls_pred1
        nn.Conv2d(
            in_channels=channels_list[8],
            out_channels= num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred1
        nn.Conv2d(
            in_channels=channels_list[8],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # ori_pred1
        nn.Conv2d(
            in_channels=channels_list[8],
            out_channels=1 * (angle_max + num_anchors),
            kernel_size=1
        ),
        # stem2
        Conv(
            in_channels=channels_list[10],
            out_channels=channels_list[10],
            kernel_size=1,
            stride=1
        ),
        # cls_conv2
        Conv(
            in_channels=channels_list[10],
            out_channels=channels_list[10],
            kernel_size=3,
            stride=1
        ),
        # reg_conv2
        Conv(
            in_channels=channels_list[10],
            out_channels=channels_list[10],
            kernel_size=3,
            stride=1
        ),
        # ori_conv2
        Conv(
            in_channels=channels_list[10],
            out_channels=channels_list[10],
            kernel_size=3,
            stride=1
        ),
        # cls_pred2
        nn.Conv2d(
            in_channels=channels_list[10],
            out_channels=num_classes * num_anchors,
            kernel_size=1
        ),
        # reg_pred2
        nn.Conv2d(
            in_channels=channels_list[10],
            out_channels=4 * (reg_max + num_anchors),
            kernel_size=1
        ),
        # ori_pred2
        nn.Conv2d(
            in_channels=channels_list[10],
            out_channels=1 * (angle_max + num_anchors),
            kernel_size=1
        )
    )
    
    return head_layers


def dist2bbox(distance, anchor_points, box_format='xyxy'):
    '''Transform distance(ltrb) to box(xywh or xyxy).'''
    lt, rb = torch.split(distance, 2, -1)
    x1y1 = anchor_points - lt
    x2y2 = anchor_points + rb
    if box_format == 'xyxy':
        bbox = torch.cat([x1y1, x2y2], -1)
    elif box_format == 'xywh':
        c_xy = (x1y1 + x2y2) / 2
        wh = x2y2 - x1y1
        bbox = torch.cat([c_xy, wh], -1)
    return bbox


def bbox2dist(anchor_points, bbox, reg_max):
    '''Transform bbox(xyxy) to dist(ltrb).'''
    x1y1, x2y2 = torch.split(bbox, 2, -1)
    lt = anchor_points - x1y1
    rb = x2y2 - anchor_points
    dist = torch.cat([lt, rb], -1).clip(0, reg_max - 0.01)
    return dist
