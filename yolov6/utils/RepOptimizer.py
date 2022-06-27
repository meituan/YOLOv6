import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from ..layers.common import RealVGGBlock, LinearAddBlock

def extract_blocks_into_list(model):
    stages = [model.stage0, model.stage1, model.stage2, model.stage3, model.stage4]
    blocks = []
    for stage in stages:
        if isinstance(stage, RealVGGBlock) or isinstance(stage, LinearAddBlock):
            blocks.append(stage)
        else:
            assert isinstance(stage, nn.Sequential)
            for block in stage.children():
                assert isinstance(block, RealVGGBlock) or isinstance(block, LinearAddBlock)
                blocks.append(block)
    return blocks

def extract_scales(model):
    blocks = extract_blocks_into_list(model)
    scales = []
    for b in blocks:
        assert isinstance(b, LinearAddBlock)
        if hasattr(b, 'scale_identity'):
            scales.append((b.scale_identity.weight.detach(), b.scale_1x1.weight.detach(), b.scale_conv.weight.detach()))
        else:
            scales.append((b.scale_1x1.weight.detach(), b.scale_conv.weight.detach()))
        print('extract scales: ', scales[-1][-2].mean(), scales[-1][-1].mean())
    return scales

def check_keywords_in_name(name, keywords=()):
    isin = False
    for keyword in keywords:
        if keyword in name:
            isin = True
    return isin

def set_weight_decay(model, skip_list=(), skip_keywords=(), echo=False):
    has_decay = []
    no_decay = []

    for name, param in model.named_parameters():
        if not param.requires_grad:
            continue  # frozen weights
        if 'identity.weight' in name:
            has_decay.append(param)
            if echo:
                print(f"{name} USE weight decay")
        elif len(param.shape) == 1 or name.endswith(".bias") or (name in skip_list) or \
            check_keywords_in_name(name, skip_keywords):
            no_decay.append(param)
            if echo:
                print(f"{name} has no weight decay")
        else:
            has_decay.append(param)
            if echo:
                print(f"{name} USE weight decay")

    return [{'params': has_decay},
            {'params': no_decay, 'weight_decay': 0.}]


class RepVGGOptimizer(SGD):
    #   scales is a list, scales[i] is a triple (scale_identity.weight, scale_1x1.weight, scale_conv.weight) or a two-tuple (scale_1x1.weight, scale_conv.weight) (if the block has no scale_identity)
    def __init__(self, model, scales, num_blocks, width_multiplier,
                 lr, momentum=0, dampening=0,
                 weight_decay=0, nesterov=False,
                 reinit=True, use_identity_scales_for_reinit=True,
                 cpu_mode=False):
        defaults = dict(lr=lr, momentum=momentum, dampening=dampening, weight_decay=weight_decay, nesterov=nesterov)
        if nesterov and (momentum <= 0 or dampening != 0):
            raise ValueError("Nesterov momentum requires a momentum and zero dampening")
        parameters = set_weight_decay(model)
        super(SGD, self).__init__(parameters, defaults)
        self.num_blocks = num_blocks
        self.width_multiplier = width_multiplier
        self.num_layers = len(scales)

        blocks = extract_blocks_into_list(model)
        convs = [b.conv for b in blocks]

        if reinit:
            for m in model.modules():
                if isinstance(m, nn.BatchNorm2d):
                    gamma_init = m.weight.mean()
                    if gamma_init == 1.0:
                        print('Checked. This is training from scratch.')
                    else:
                        raise Warning('========================== Warning! Is this really training from scratch ? =================')
            print('##################### Re-initialize #############')
            self.reinitialize(scales, convs, use_identity_scales_for_reinit)

        self.generate_gradient_masks(scales, convs, cpu_mode)

    def reinitialize(self, scales_by_idx, conv3x3_by_idx, use_identity_scales):
        for scales, conv3x3 in zip(scales_by_idx, conv3x3_by_idx):
            in_channels = conv3x3.in_channels
            out_channels = conv3x3.out_channels
            kernel_1x1 = nn.Conv2d(in_channels, out_channels, 1)
            if len(scales) == 2:
                conv3x3.weight.data = conv3x3.weight * scales[1].view(-1, 1, 1, 1) \
                                      + F.pad(kernel_1x1.weight, [1, 1, 1, 1]) * scales[0].view(-1, 1, 1, 1)
            else:
                assert len(scales) == 3
                assert in_channels == out_channels
                identity = torch.from_numpy(np.eye(out_channels, dtype=np.float32).reshape(out_channels, out_channels, 1, 1))
                conv3x3.weight.data = conv3x3.weight * scales[2].view(-1, 1, 1, 1) + F.pad(kernel_1x1.weight, [1, 1, 1, 1]) * scales[1].view(-1, 1, 1, 1)
                if use_identity_scales:     # You may initialize the imaginary CSLA block with the trained identity_scale values. Makes almost no difference.
                    identity_scale_weight = scales[0]
                    conv3x3.weight.data += F.pad(identity * identity_scale_weight.view(-1, 1, 1, 1), [1, 1, 1, 1])
                else:
                    conv3x3.weight.data += F.pad(identity, [1, 1, 1, 1])


    def generate_gradient_masks(self, scales_by_idx, conv3x3_by_idx, cpu_mode=False):
        self.grad_mask_map = {}
        for scales, conv3x3 in zip(scales_by_idx, conv3x3_by_idx):
            para = conv3x3.weight
            if len(scales) == 2:
                mask = torch.ones_like(para) * (scales[1] ** 2).view(-1, 1, 1, 1)
                mask[:, :, 1:2, 1:2] += torch.ones(para.shape[0], para.shape[1], 1, 1) * (scales[0] ** 2).view(-1, 1, 1, 1)
            else:
                mask = torch.ones_like(para) * (scales[2] ** 2).view(-1, 1, 1, 1)
                mask[:, :, 1:2, 1:2] += torch.ones(para.shape[0], para.shape[1], 1, 1) * (scales[1] ** 2).view(-1, 1, 1, 1)
                ids = np.arange(para.shape[1])
                assert para.shape[1] == para.shape[0]
                mask[ids, ids, 1:2, 1:2] += 1.0
            if cpu_mode:
                self.grad_mask_map[para] = mask
            else:
                self.grad_mask_map[para] = mask.cuda()

    def __setstate__(self, state):
        super(SGD, self).__setstate__(state)
        for group in self.param_groups:
            group.setdefault('nesterov', False)

    def step(self, closure=None):

        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            dampening = group['dampening']
            nesterov = group['nesterov']

            for p in group['params']:
                if p.grad is None:
                    continue

                if p in self.grad_mask_map:
                    d_p = p.grad.data * self.grad_mask_map[p]  # Note: multiply the mask here
                else:
                    d_p = p.grad.data

                if weight_decay != 0:
                    d_p.add_(weight_decay, p.data)
                if momentum != 0:
                    param_state = self.state[p]
                    if 'momentum_buffer' not in param_state:
                        buf = param_state['momentum_buffer'] = torch.clone(d_p).detach()
                    else:
                        buf = param_state['momentum_buffer']
                        buf.mul_(momentum).add_(1 - dampening, d_p)
                    if nesterov:
                        d_p = d_p.add(momentum, buf)
                    else:
                        d_p = buf

                p.data.add_(-group['lr'], d_p)

        return loss