"""CUDA AMP entry points across the supported PyTorch versions."""

from functools import partial

import torch


_amp = getattr(torch, 'amp', None)
autocast = (
    partial(_amp.autocast, 'cuda')
    if hasattr(_amp, 'autocast') else torch.cuda.amp.autocast
)
GradScaler = (
    partial(_amp.GradScaler, 'cuda')
    if hasattr(_amp, 'GradScaler') else torch.cuda.amp.GradScaler
)
