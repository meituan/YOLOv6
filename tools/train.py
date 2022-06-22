#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import os.path as osp
import torch
import torch.distributed as dist
import sys

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.core.engine import Trainer
from yolov6.utils.config import Config
from yolov6.utils.events import LOGGER, save_yaml
from yolov6.utils.envs import get_envs, select_device, set_random_seed


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Training', add_help=add_help)
    parser.add_argument('--data-path', default='./data/coco.yaml', type=str, help='dataset path')
    parser.add_argument('--conf-file', default='./configs/yolov6s.py', type=str, help='experiment description file')
    parser.add_argument('--img-size', type=int, default=640, help='train, val image size (pixels)')
    parser.add_argument('--batch-size', default=32, type=int, help='total batch size for all GPUs')
    parser.add_argument('--epochs', default=400, type=int, help='number of total epochs to run')
    parser.add_argument('--workers', default=8, type=int, help='number of data loading workers (default: 8)')
    parser.add_argument('--device', default='0', type=str, help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--noval', action='store_true', help='only evaluate in final epoch')
    parser.add_argument('--check-images', action='store_true', help='check images when initializing datasets')
    parser.add_argument('--check-labels', action='store_true', help='check label files when initializing datasets')
    parser.add_argument('--output-dir', default='./runs/train', type=str, help='path to save outputs')
    parser.add_argument('--name', default='exp', type=str, help='experiment name, save to output_dir/name')
    parser.add_argument('--dist_url', type=str, default="tcp://127.0.0.1:8888")
    parser.add_argument('--gpu_count', type=int, default=0)
    parser.add_argument('--local_rank', type=int, default=-1, help='DDP parameter, do not modify')

    return parser


def check_and_init(args):
    '''check config files and device, and initialize '''

    # check files
    args.save_dir = osp.join(args.output_dir, args.name)
    os.makedirs(args.save_dir, exist_ok=True)
    cfg = Config.fromfile(args.conf_file)

    # check device
    device = select_device(args.device)

    # set random seed
    set_random_seed(1+args.rank, deterministic=(args.rank == -1))

    # save args
    save_yaml(vars(args), osp.join(args.save_dir, 'args.yaml'))

    return cfg, device


def main(args):
    '''main function of training'''
    # Setup
    args.rank, args.local_rank, args.world_size = get_envs()
    LOGGER.info(f'training args are: {args}\n')
    cfg, device = check_and_init(args)

    if args.local_rank != -1: # if DDP mode
        torch.cuda.set_device(args.local_rank)
        device = torch.device('cuda', args.local_rank)
        LOGGER.info('Initializing process group... ')
        dist.init_process_group(backend="nccl" if dist.is_nccl_available() else "gloo", \
                init_method=args.dist_url, rank=args.local_rank, world_size=args.world_size)

    # Start
    trainer = Trainer(args, cfg, device)
    trainer.train()

    # End
    if args.world_size > 1 and args.rank == 0:
        LOGGER.info('Destroying process group... ')
        dist.destroy_process_group()


if __name__ == '__main__':
    args = get_args_parser().parse_args()
    main(args)
