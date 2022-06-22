#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import time
from copy import deepcopy
import os.path as osp

from tqdm import tqdm

import numpy as np
import torch
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter

import tools.eval as eval
from yolov6.data.data_load import create_dataloader
from yolov6.models.yolo import build_model
from yolov6.models.loss import ComputeLoss
from yolov6.utils.events import LOGGER, NCOLS, load_yaml, write_tblog
from yolov6.utils.ema import ModelEMA, de_parallel
from yolov6.utils.checkpoint import load_state_dict, save_checkpoint, strip_optimizer
from yolov6.solver.build import build_optimizer, build_lr_scheduler


class Trainer:
    def __init__(self, args, cfg, device):
        self.args = args
        self.cfg = cfg
        self.device = device

        self.rank = args.rank
        self.local_rank = args.local_rank
        self.world_size = args.world_size
        self.main_process = self.rank in [-1, 0]
        self.save_dir = args.save_dir
        # get data loader
        self.data_dict = load_yaml(args.data_path)
        self.num_classes = self.data_dict['nc']
        self.train_loader, self.val_loader = self.get_data_loader(args, cfg, self.data_dict)
        # get model and optimizer
        model = self.get_model(args, cfg, self.num_classes, device)
        self.optimizer = self.get_optimizer(args, cfg, model)
        self.scheduler, self.lf = self.get_lr_scheduler(args, cfg, self.optimizer)
        self.ema = ModelEMA(model) if self.main_process else None
        self.model = self.parallel_model(args, model, device)
        self.model.nc, self.model.names = self.data_dict['nc'], self.data_dict['names']
        # tensorboard
        self.tblogger = SummaryWriter(self.save_dir) if self.main_process else None

        self.start_epoch = 0
        self.max_epoch = args.epochs
        self.max_stepnum = len(self.train_loader)
        self.batch_size = args.batch_size
        self.img_size = args.img_size

    # Training Process
    def train(self):
        try:
            self.train_before_loop()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.train_in_loop()

        except Exception as _:
            LOGGER.error('ERROR in training loop or eval/save model.')
            raise
        finally:
            self.train_after_loop()

    # Training loop for each epoch
    def train_in_loop(self):
        try:
            self.prepare_for_steps()
            for self.step, self.batch_data in self.pbar:
                self.train_in_steps()
                self.print_details()
        except Exception as _:
            LOGGER.error('ERROR in training steps.')
            raise
        try:
            self.eval_and_save()
        except Exception as _:
            LOGGER.error('ERROR in evaluate and save model.')
            raise

    # Training loop for batchdata
    def train_in_steps(self):
        images, targets = self.prepro_data(self.batch_data, self.device)
        # forward
        with amp.autocast(enabled=self.device != 'cpu'):
            preds = self.model(images)
            total_loss, loss_items = self.compute_loss(preds, targets)
            if self.rank != -1:
                total_loss *= self.world_size
        # backward
        self.scaler.scale(total_loss).backward()
        self.loss_items = loss_items
        self.update_optimizer()

    def eval_and_save(self):
        epoch_sub = self.max_epoch - self.epoch
        val_period = 20 if epoch_sub > 100 else 1 # to fasten training time, evaluate in every 20 epochs for the early stage.
        is_val_epoch = (not self.args.noval or (epoch_sub == 1)) and (self.epoch % val_period == 0)
        if self.main_process:
            self.ema.update_attr(self.model, include=['nc', 'names', 'stride']) # update attributes for ema model
            if is_val_epoch:
                self.eval_model()
                self.ap = self.evaluate_results[0] * 0.1 + self.evaluate_results[1] * 0.9
                self.best_ap = max(self.ap, self.best_ap)
            # save ckpt
            ckpt = {
                    'model': deepcopy(de_parallel(self.model)).half(),
                    'ema': deepcopy(self.ema.ema).half(),
                    'updates': self.ema.updates,
                    'optimizer': self.optimizer.state_dict(),
                    'epoch': self.epoch,
                    }

            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            save_checkpoint(ckpt, (is_val_epoch) and (self.ap == self.best_ap), save_ckpt_dir, model_name='last_ckpt')
            del ckpt
            # log for tensorboard
            write_tblog(self.tblogger, self.epoch, self.evaluate_results, self.mean_loss)

    def eval_model(self):
        results = eval.run(self.data_dict,
                           batch_size=self.batch_size // self.world_size * 2,
                           img_size=self.img_size,
                           model=self.ema.ema,
                           dataloader=self.val_loader,
                           save_dir=self.save_dir,
                           task='train')

        LOGGER.info(f"Epoch: {self.epoch} | mAP@0.5: {results[0]} | mAP@0.50:0.95: {results[1]}")
        self.evaluate_results = results[:2]

    def train_before_loop(self):
        LOGGER.info('Training start...')
        self.start_time = time.time()
        self.warmup_stepnum = max(round(self.cfg.solver.warmup_epochs * self.max_stepnum), 1000)
        self.scheduler.last_epoch = self.start_epoch - 1
        self.last_opt_step = -1
        self.scaler = amp.GradScaler(enabled=self.device != 'cpu')

        self.best_ap, self.ap = 0.0, 0.0
        self.evaluate_results = (0, 0) # AP50, AP50_95
        self.compute_loss = ComputeLoss(iou_type=self.cfg.model.head.iou_type)

    def prepare_for_steps(self):
        if self.epoch > self.start_epoch:
            self.scheduler.step()
        self.model.train()
        if self.rank != -1:
            self.train_loader.sampler.set_epoch(self.epoch)
        self.mean_loss = torch.zeros(4, device=self.device)
        self.optimizer.zero_grad()

        LOGGER.info(('\n' + '%10s' * 5) % ('Epoch', 'iou_loss', 'l1_loss', 'obj_loss', 'cls_loss'))
        self.pbar = enumerate(self.train_loader)
        if self.main_process:
            self.pbar = tqdm(self.pbar, total=self.max_stepnum, ncols=NCOLS, bar_format='{l_bar}{bar:10}{r_bar}{bar:-10b}')

    # Print loss after each steps
    def print_details(self):
        if self.main_process:
            self.mean_loss = (self.mean_loss * self.step + self.loss_items) / (self.step + 1)
            self.pbar.set_description(('%10s' + '%10.4g' * 4) % (f'{self.epoch}/{self.max_epoch - 1}', \
                                                                *(self.mean_loss)))

    # Empty cache if training finished
    def train_after_loop(self):
        if self.main_process:
            LOGGER.info(f'\nTraining completed in {(time.time() - self.start_time) / 3600:.3f} hours.')
            save_ckpt_dir = osp.join(self.save_dir, 'weights')
            strip_optimizer(save_ckpt_dir)  # strip optimizers for saved pt model
        if self.device != 'cpu':
            torch.cuda.empty_cache()

    def update_optimizer(self):
        curr_step = self.step + self.max_stepnum * self.epoch
        self.accumulate = max(1, round(64 / self.batch_size))
        if curr_step <= self.warmup_stepnum:
            self.accumulate = max(1, np.interp(curr_step, [0, self.warmup_stepnum], [1, 64 / self.batch_size]).round())
            for k, param in enumerate(self.optimizer.param_groups):
                warmup_bias_lr = self.cfg.solver.warmup_bias_lr if k == 2 else 0.0
                param['lr'] = np.interp(curr_step, [0, self.warmup_stepnum], [warmup_bias_lr, param['initial_lr'] * self.lf(self.epoch)])
                if 'momentum' in param:
                    param['momentum'] = np.interp(curr_step, [0, self.warmup_stepnum], [self.cfg.solver.warmup_momentum, self.cfg.solver.momentum])
        if curr_step - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
            self.last_opt_step = curr_step

    @staticmethod
    def get_data_loader(args, cfg, data_dict):
        train_path, val_path = data_dict['train'], data_dict['val']
        # check data
        nc = int(data_dict['nc'])
        class_names = data_dict['names']
        assert len(class_names) == nc, f'the length of class names does not match the number of classes defined'
        grid_size = max(int(max(cfg.model.head.strides)), 32)
        # create train dataloader
        train_loader = create_dataloader(train_path, args.img_size, args.batch_size // args.world_size, grid_size,
                                         hyp=dict(cfg.data_aug), augment=True, rect=False, rank=args.local_rank,
                                         workers=args.workers, shuffle=True, check_images=args.check_images,
                                         check_labels=args.check_labels, class_names=class_names, task='train')[0]
        # create val dataloader
        val_loader = None
        if args.rank in [-1, 0]:
            val_loader = create_dataloader(val_path, args.img_size, args.batch_size // args.world_size * 2, grid_size,
                                           hyp=dict(cfg.data_aug), rect=True, rank=-1, pad=0.5,
                                           workers=args.workers, check_images=args.check_images,
                                           check_labels=args.check_labels, class_names=class_names, task='val')[0]

        return train_loader, val_loader

    @staticmethod
    def prepro_data(batch_data, device):
        images = batch_data[0].to(device, non_blocking=True).float() / 255
        targets = batch_data[1].to(device)
        return images, targets

    @staticmethod
    def get_model(args, cfg, nc, device):
        model = build_model(cfg, nc, device)
        weights = cfg.model.pretrained
        if weights:  # finetune if pretrained model is set
            LOGGER.info(f'Loading state_dict from {weights} for fine-tuning...')
            model = load_state_dict(weights, model, map_location=device)
        LOGGER.info('Model: {}'.format(model))
        return model

    @staticmethod
    def parallel_model(args, model, device):
        # If DP mode
        dp_mode = device.type != 'cpu' and args.rank == -1
        if dp_mode and torch.cuda.device_count() > 1:
            LOGGER.warning('WARNING: DP not recommended, use DDP instead.\n')
            model = torch.nn.DataParallel(model)

        # If DDP mode
        ddp_mode = device.type != 'cpu' and args.rank != -1
        if ddp_mode:
            model = DDP(model, device_ids=[args.local_rank], output_device=args.local_rank)

        return model

    @staticmethod
    def get_optimizer(args, cfg, model):
        accumulate = max(1, round(64 / args.batch_size))
        cfg.solver.weight_decay *= args.batch_size * accumulate / 64
        optimizer = build_optimizer(cfg, model)
        return optimizer

    @staticmethod
    def get_lr_scheduler(args, cfg, optimizer):
        epochs = args.epochs
        lr_scheduler, lf = build_lr_scheduler(cfg, optimizer, epochs)
        return lr_scheduler, lf
