#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math

import numpy as np
import torch
import torch.nn as nn

from yolov6.utils.events import LOGGER


def build_optimizer(cfg, model):
    """ Build optimizer from cfg file."""
    g_bnw, g_w, g_b = [], [], []
    for v in model.modules():
        if hasattr(v, 'bias') and isinstance(v.bias, nn.Parameter):
            g_b.append(v.bias)
        if isinstance(v, nn.BatchNorm2d):
            g_bnw.append(v.weight)
        elif hasattr(v, 'weight') and isinstance(v.weight, nn.Parameter):
            g_w.append(v.weight)

    # assert cfg.solver.optim == 'SGD' or 'Adam', 'ERROR: unknown optimizer, use SGD defaulted'
    if cfg.solver.optim == 'SGD':
        optimizer = torch.optim.SGD(g_bnw, lr=cfg.solver.lr0, momentum=cfg.solver.momentum, nesterov=True)
    elif cfg.solver.optim == 'Adam':
        optimizer = torch.optim.Adam(g_bnw, lr=cfg.solver.lr0, betas=(cfg.solver.momentum, 0.999))
    elif cfg.solver.optim == 'AdamW':
        optimizer = torch.optim.AdamW(g_bnw, lr=cfg.solver.lr0, eps=1e-3, weight_decay=0.05)
        # TODO ???? 有影响?
        # optimizer = torch.optim.AdamW(g_bnw, lr=cfg.solver.lr0, weight_decay=0.05)

    optimizer.add_param_group({'params': g_w, 'weight_decay': cfg.solver.weight_decay})
    optimizer.add_param_group({'params': g_b})

    del g_bnw, g_w, g_b
    return optimizer


def build_lr_scheduler(cfg, optimizer, epochs, num_batches):
    """Build learning rate scheduler from cfg file."""
    # NOTE 怎么计算的?
    if cfg.solver.lr_scheduler == 'Cosine':
        lf = lambda x: ((1 - math.cos(x * math.pi / epochs)) / 2) * (cfg.solver.lrf - 1) + 1
    elif cfg.solver.lr_scheduler == 'Constant':
        lf = lambda x: 1.0
    else:
        LOGGER.error('unknown lr scheduler, use Cosine defaulted')

    scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=lf)

    # scheduler = CosineDecayLR(
    #     optimizer,
    #     T_max=epochs * num_batches,
    #     lr_init=cfg.solver.lr0,
    #     lr_min=cfg.solver.lr0 * cfg.solver.lrf,
    #     warmup=cfg.solver.warmup_epochs * num_batches,
    # )

    return scheduler, lf


class CosineDecayLR(object):
    def __init__(self, optimizer, T_max, lr_init, lr_min=0., warmup=0):
        """
        a cosine decay scheduler about steps, not epochs.
        :param optimizer: ex. optim.SGD
        :param T_max:  max steps, and steps=epochs * batches
        :param lr_max: lr_max is init lr.
        :param warmup: in the training begin, the lr is smoothly increase from 0 to lr_init, which means "warmup",
                        this means warmup steps, if 0 that means don't use lr warmup.
        """
        super(CosineDecayLR, self).__init__()
        self.__optimizer = optimizer
        self.__T_max = T_max
        self.__lr_min = lr_min
        self.__lr_max = lr_init
        self.__warmup = warmup


    def step(self, t):
        if self.__warmup and t < self.__warmup:
            lr = self.__lr_max / self.__warmup * t
        else:
            T_max = self.__T_max - self.__warmup
            t = t - self.__warmup
            lr = self.__lr_min + 0.5 * (self.__lr_max - self.__lr_min) * (1 + np.cos(t/T_max * np.pi))
        for param_group in self.__optimizer.param_groups:
            param_group["lr"] = lr
