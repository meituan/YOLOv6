import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from yolov6.layers.common import *
from yolov6.assigners.anchor_generator import generate_anchors
from yolov6.utils.general import dist2bbox


class Detect(nn.Module):
    export = False
    '''Efficient Decoupled Head for fusing anchor-base branches.
    '''
    def __init__(self, num_classes=80, anchors=None, num_layers=3, inplace=True, head_layers=None, reg_mask=None, use_dfl=True, reg_max=16, nm=32, fuse_ab=False):  # detection layer
        super().__init__()
        assert head_layers is not None
        assert reg_mask is not None
        self.nc = num_classes  # number of classes
        self.no = num_classes + 5 + nm  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        self.nm = nm # number of masks
        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.grid = [torch.zeros(1)] * num_layers
        self.fuse_ab = fuse_ab
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
        self.reg_mask = reg_mask

        # Init decouple head
        self.stems = nn.ModuleList()
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        self.seg_convs = nn.ModuleList()
        self.cls_preds = nn.ModuleList()
        self.reg_preds = nn.ModuleList()
        self.seg_preds = nn.ModuleList()
        self.cls_preds_ab = nn.ModuleList()
        self.reg_preds_ab = nn.ModuleList()
        self.seg_preds_ab = nn.ModuleList()
        self.seg_proto = nn.ModuleList()
        self.seg_proto.append(reg_mask[0])
        self.seg_proto.append(reg_mask[1])
        self.seg_proto.append(reg_mask[2])


        # Efficient decoupled head layers
        for i in range(num_layers):
            idx = i*10
            self.stems.append(head_layers[idx])
            self.cls_convs.append(head_layers[idx+1])
            self.reg_convs.append(head_layers[idx+2])
            self.seg_convs.append(head_layers[idx+3])
            self.cls_preds.append(head_layers[idx+4])
            self.reg_preds.append(head_layers[idx+5])
            self.seg_preds.append(head_layers[idx+6])
            if self.fuse_ab:
                self.cls_preds_ab.append(head_layers[idx+7])
                self.reg_preds_ab.append(head_layers[idx+8])
                self.seg_preds_ab.append(head_layers[idx+9])
            

    def initialize_biases(self):

        for conv in self.cls_preds:
            b = conv.bias.view(-1, )
            b.data.fill_(-math.log((1 - self.prior_prob) / self.prior_prob))
            conv.bias = torch.nn.Parameter(b.view(-1), requires_grad=True)
            w = conv.weight
            w.data.fill_(0.)
            conv.weight = torch.nn.Parameter(w, requires_grad=True)

        if self.fuse_ab:
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

        if self.fuse_ab:
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


    def handleseg_ab(self, sgot_lst, sg_msk_lst):
        '''
        sg_msk_lst --> lst sg_msk: segmask: Shape(bs, 32, w, h)
        sgot_lst --> lst sgot: seg_output_conf: Shape(bs, num_of_anchors, h, w, 32)
        sgot.flatten(1, 3) -> Shape(bs, n*num_of_anchors, 32)
        for j in range(bs) -> ((n*num_of_anchor, 32)@(32, w0, h0) = (n*num_of_anchor, 32)@(32, w0, h0))
        '''
        mask_res = []
        for i in range(len(sgot_lst)):
            sgot = sgot_lst[i]
            sg_msk = sg_msk_lst[i]
            s_shape = sgot.shape[1:4]
            sgot = sgot.flatten(1, 3)
            t_mask_res = []
            for j in range(sgot.shape[0]):
                sgot_t = sgot[j] # (n, 32)
                sg_msk_t = sg_msk[j] # (32, w, h)
                m_t = (sgot_t@sg_msk_t.reshape(self.nm, -1)).reshape(-1, *sg_msk_t.shape[1:])
                m_t = m_t.unsqueeze(0)
                t_mask_res.append(m_t)
            mask_res.append(torch.cat(t_mask_res, 0).flatten(0,1))
        return mask_res
            


            

    def forward(self, x):
        if self.training:
            device = x[0].device
            cls_score_list_af = []
            reg_dist_list_af = []
            cls_score_list_ab = []
            reg_dist_list_ab = []
            seg_conf_list_af = []
            seg_conf_list_ab = []
            seg_list = []
            af_seg_list = []
            ab_seg_list = []

            s1 = self.seg_proto[0](x[0])
            s2 = self.seg_proto[1](x[1])
            s3 = self.seg_proto[2](x[2])
            seg_mask = s1 + s2 + s3
            seg_list.append(seg_mask)



            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w
                
                
                x[i] = self.stems[i](x[i])
                
                
                cls_x = x[i]
                reg_x = x[i]
                seg_x = x[i]

                cls_feat = self.cls_convs[i](cls_x)
                reg_feat = self.reg_convs[i](reg_x)
                seg_feat = self.seg_convs[i](seg_x)

                #anchor_base
                if self.fuse_ab:
                    cls_output_ab = self.cls_preds_ab[i](cls_feat)
                    reg_output_ab = self.reg_preds_ab[i](reg_feat)
                    seg_output_ab = self.seg_preds_ab[i](seg_feat)

                    cls_output_ab = torch.sigmoid(cls_output_ab)
                    seg_output_ab = torch.sigmoid(seg_output_ab)
                    if self.fuse_ab:
                        seg_conf_list_ab.append(seg_output_ab.reshape(b, self.na, -1, h, w).permute(0,1,3,4,2))
                    cls_output_ab = cls_output_ab.reshape(b, self.na, -1, h, w).permute(0,1,3,4,2)
                    cls_score_list_ab.append(cls_output_ab.flatten(1,3))
                    

                    reg_output_ab = reg_output_ab.reshape(b, self.na, -1, h, w).permute(0,1,3,4,2)
                    reg_output_ab[..., 2:4] = ((reg_output_ab[..., 2:4].sigmoid() * 2) ** 2 ) * (self.anchors_init[i].reshape(1, self.na, 1, 1, 2).to(device))
                    reg_dist_list_ab.append(reg_output_ab.flatten(1,3))

                #anchor_free
                cls_output_af = self.cls_preds[i](cls_feat)
                reg_output_af = self.reg_preds[i](reg_feat)
                seg_output_af = self.seg_preds[i](seg_feat)

                cls_output_af = torch.sigmoid(cls_output_af)
                # seg_output_af = torch.sigmoid(seg_output_af)
                seg_conf_list_af.append(seg_output_af.flatten(2).permute((0, 2, 1)))

                cls_score_list_af.append(cls_output_af.flatten(2).permute((0, 2, 1)))
                reg_dist_list_af.append(reg_output_af.flatten(2).permute((0, 2, 1)))

            #Not support fuseab now
            if False:
                ab_seg_list = self.handleseg_ab(seg_conf_list_ab, seg_list) if self.fuse_ab else []
                cls_score_list_ab = torch.cat(cls_score_list_ab, axis=1)
                reg_dist_list_ab = torch.cat(reg_dist_list_ab, axis=1)
            cls_score_list_af = torch.cat(cls_score_list_af, axis=1)
            reg_dist_list_af = torch.cat(reg_dist_list_af, axis=1)

            return x, cls_score_list_ab, reg_dist_list_ab, cls_score_list_af, reg_dist_list_af, [seg_conf_list_af, seg_list], ab_seg_list

        else:
            device = x[0].device
            cls_score_list_af = []
            reg_dist_list_af = []
            seg_list = []
            seg_conf_list_af = []
            s1 = self.seg_proto[0](x[0])
            s2 = self.seg_proto[1](x[1])
            s3 = self.seg_proto[2](x[2])
            seg_mask = s1 + s2 + s3
            seg_list.append(seg_mask)

            for i in range(self.nl):
                b, _, h, w = x[i].shape
                l = h * w

                
                x[i] = self.stems[i](x[i])
                
                cls_x = x[i]
                reg_x = x[i]
                seg_x = x[i]

                cls_feat = self.cls_convs[i](cls_x)
                reg_feat = self.reg_convs[i](reg_x)
                seg_feat = self.seg_convs[i](seg_x)

                #anchor_free
                cls_output_af = self.cls_preds[i](cls_feat)
                reg_output_af = self.reg_preds[i](reg_feat)
                seg_output_af = self.seg_preds[i](seg_feat)

                if self.use_dfl:
                    reg_output_af = reg_output_af.reshape([-1, 4, self.reg_max + 1, l]).permute(0, 2, 1, 3)
                    reg_output_af = self.proj_conv(F.softmax(reg_output_af, dim=1))

                cls_output_af = torch.sigmoid(cls_output_af)
                proto_no = (torch.ones(b, 1, l) * i).cuda()
                

                if self.export:
                    cls_score_list_af.append(cls_output_af)
                    reg_dist_list_af.append(reg_output_af)
                    seg_conf_list_af.append(seg_output_af)
                else:
                    cls_score_list_af.append(cls_output_af.reshape([b, self.nc, l]))
                    reg_dist_list_af.append(reg_output_af.reshape([b, 4, l]))
                    seg_conf_list_af.append(torch.cat([proto_no, seg_output_af.reshape([b, 67, l])], axis = 1)) #[which_proto, (32...)]

            if self.export:
                return tuple(torch.cat([cls, reg, seg], 1) for cls, reg, seg in zip(cls_score_list_af, reg_dist_list_af, seg_conf_list_af)), seg_list[0]

            cls_score_list_af = torch.cat(cls_score_list_af, axis=-1).permute(0, 2, 1)
            reg_dist_list_af = torch.cat(reg_dist_list_af, axis=-1).permute(0, 2, 1)
            seg_conf_list_af = torch.cat(seg_conf_list_af, axis=-1).permute(0, 2, 1)



            #anchor_free
            anchor_points_af, stride_tensor_af = generate_anchors(
                x, self.stride, self.grid_cell_size, self.grid_cell_offset, device=x[0].device, is_eval=True, mode='af')

            pred_bboxes_af = dist2bbox(reg_dist_list_af, anchor_points_af, box_format='xywh')
            pred_bboxes_af *= stride_tensor_af

            pred_bboxes = pred_bboxes_af
            cls_score_list = cls_score_list_af

            return torch.cat(
                [
                    pred_bboxes,
                    torch.ones((b, pred_bboxes.shape[1], 1), device=pred_bboxes.device, dtype=pred_bboxes.dtype),
                    cls_score_list
                ],
                axis=-1), seg_list, seg_conf_list_af

class Proto(nn.Module):
    # Borrowed from YOLOv5
    def __init__(self, num_layers, channels_list, pos, c_=256, c2=64, scale_factor=2):  # ch_in, number of protos, number of masks
        super().__init__()
        chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]
        c1 = channels_list[chx[pos]]
        self.cv1 = Conv(c1, c_, k=3)
        self.upsample = nn.Upsample(scale_factor=scale_factor, mode='nearest')
        self.cv2 = Conv(c_, c_, k=3)
        self.cv3 = Conv(c_, c2)

    def forward(self, x):
        return self.cv3(self.cv2(self.upsample(self.cv1(x))))
    
def autopad(k, p=None, d=1):  # kernel, padding, dilation
    # Pad to 'same' shape outputs
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]  # actual kernel-size
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]  # auto-pad
    return p


class Conv(nn.Module):
    # Standard convolution with args(ch_in, ch_out, kernel, stride, padding, groups, dilation, activation)
    default_act = nn.SiLU()  # default activation

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True):
        super().__init__()
        self.conv = nn.Conv2d(c1, c2, k, s, autopad(k, p, d), groups=g, dilation=d, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))


def build_effidehead_layer(channels_list, num_anchors, num_classes, reg_max=16, num_layers=3, num_masks=67, fuse_ab=False):

    chx = [6, 8, 10] if num_layers == 3 else [8, 9, 10, 11]

    head_layers = nn.Sequential(
        # stem0
        ConvBNSiLU(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv0
        ConvBNSiLU(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv0
        ConvBNSiLU(
            in_channels=channels_list[chx[0]],
            out_channels=channels_list[chx[0]],
            kernel_size=3,
            stride=1
        ),
        # seg_conv0
        ConvBNSiLU(
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
            out_channels=4 * (reg_max + 1),
            kernel_size=1
        ),
        # seg_pred0_af
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=num_masks,
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
        # seg_pred0_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[0]],
            out_channels=num_masks * num_anchors,
            kernel_size=1
        ),
        # stem1
        ConvBNSiLU(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv1
        ConvBNSiLU(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv1
        ConvBNSiLU(
            in_channels=channels_list[chx[1]],
            out_channels=channels_list[chx[1]],
            kernel_size=3,
            stride=1
        ),
        # seg_conv1
        ConvBNSiLU(
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
            out_channels=4 * (reg_max + 1),
            kernel_size=1
        ),
        # seg_pred1_af
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=num_masks,
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
        # seg_pred1_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[1]],
            out_channels=num_masks * num_anchors,
            kernel_size=1
        ),
        # stem2
        ConvBNSiLU(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=1,
            stride=1
        ),
        # cls_conv2
        ConvBNSiLU(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # reg_conv2
        ConvBNSiLU(
            in_channels=channels_list[chx[2]],
            out_channels=channels_list[chx[2]],
            kernel_size=3,
            stride=1
        ),
        # seg_conv2
        ConvBNSiLU(
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
            out_channels=4 * (reg_max + 1),
            kernel_size=1
        ),
        # seg_pred2_af
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=num_masks,
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
        # seg_pred2_3ab
        nn.Conv2d(
            in_channels=channels_list[chx[2]],
            out_channels=num_masks * num_anchors,
            kernel_size=1
        ),
    )

    return head_layers







