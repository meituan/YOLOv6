#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import math
import os
import os.path as osp
import time
from ast import Pass
from copy import deepcopy

import cv2
import numpy as np
import torch
from rich.progress import (BarColumn, Progress, SpinnerColumn, TextColumn,
                           TimeElapsedColumn, TimeRemainingColumn)
from torch.cuda import amp
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

import tools.eval_R as eval
from yolov6.data.data_load_R import create_dataloader
from yolov6.models.losses.loss_distill_R import \
    ComputeLoss as ComputeLoss_distill
from yolov6.models.losses.loss_fuseab import ComputeLoss as ComputeLoss_ab
from yolov6.models.losses.loss_R import ComputeLoss as ComputeLoss
from yolov6.models.yolo_R import build_model
from yolov6.solver.build import build_lr_scheduler, build_optimizer
from yolov6.utils.checkpoint import (load_state_dict, save_checkpoint,
                                     strip_optimizer)
from yolov6.utils.ema import ModelEMA, de_parallel
from yolov6.utils.events_R import (LOGGER, NCOLS, load_yaml, write_tbimg,
                                   write_tblog)
from yolov6.utils.general import download_ckpt
from yolov6.utils.nms_R import xywh2xyxy, xyxy2xywh
from yolov6.utils.RepOptimizer import RepVGGOptimizer, extract_scales

# from yolov6.models.losses.loss_distill_ns_R import \
    # ComputeLoss as ComputeLoss_distill_ns


class Trainer:
    def __init__(self, args, cfg, device):
        self.args = args
        self.cfg = cfg
        self.device = device

        if args.resume:
            self.ckpt = torch.load(args.resume, map_location="cpu")

        self.rank = args.rank
        self.local_rank = args.local_rank
        self.world_size = args.world_size
        self.main_process = self.rank in [-1, 0]
        self.save_dir = args.save_dir
        # get data loader
        self.data_dict = load_yaml(args.data_path)
        self.num_classes = self.data_dict["nc"]
        # NOTE data loader
        self.train_loader, self.val_loader = self.get_data_loader(
            args, cfg, self.data_dict
        )
        # get model and optimizer
        # NOTE YOLOv6n 和 YOLOV6s 都是默认蒸馏配置, 需要将这个解耦开了

        self.distill_ns = True if self.args.distill and args.distill_ns else False
        # NOTE change model
        model = self.get_model(args, cfg, self.num_classes, device)
        if self.args.distill:
            if self.args.fuse_ab:
                LOGGER.error("ERROR in: Distill models should turn off the fuse_ab.\n")
                exit()
            self.teacher_model = self.get_teacher_model(
                args, cfg, self.num_classes, device
            )
        if self.args.quant:
            self.quant_setup(model, cfg, device)
        if cfg.training_mode == "repopt":
            scales = self.load_scale_from_pretrained_models(cfg, device)
            reinit = False if cfg.model.pretrained is not None else True
            self.optimizer = RepVGGOptimizer(model, scales, args, cfg, reinit=reinit)
        else:
            self.optimizer = self.get_optimizer(args, cfg, model)
        self.scheduler, self.lf = self.get_lr_scheduler(args, cfg, self.optimizer)
        self.ema = ModelEMA(model) if self.main_process else None
        # tensorboard
        self.tblogger = SummaryWriter(self.save_dir) if self.main_process else None
        self.start_epoch = 0
        # resume
        if hasattr(self, "ckpt"):
            resume_state_dict = (
                self.ckpt["model"].float().state_dict()
            )  # checkpoint state_dict as FP32
            model.load_state_dict(resume_state_dict, strict=True)  # load
            self.start_epoch = self.ckpt["epoch"] + 1
            self.optimizer.load_state_dict(self.ckpt["optimizer"])
            if self.main_process:
                self.ema.ema.load_state_dict(self.ckpt["ema"].float().state_dict())
                self.ema.updates = self.ckpt["updates"]
        self.model = self.parallel_model(args, model, device)  # NOTE dp/ddp
        self.model.nc, self.model.names = self.data_dict["nc"], self.data_dict["names"]

        self.max_epoch = args.epochs
        self.max_stepnum = len(self.train_loader)
        self.batch_size = args.batch_size
        self.img_size = args.img_size
        self.vis_imgs_list = []
        self.write_trainbatch_tb = args.write_trainbatch_tb
        # set color for classnames
        self.color = [
            tuple(np.random.choice(range(256), size=3)) for _ in range(self.model.nc)
        ]
        # REVIEW loss_num and info
        self.loss_num = 4
        self.loss_info = ["Epoch", "iou_loss", "dfl_loss", "cls_loss", "ang_loss"]

        self.progress = Progress(
            SpinnerColumn(),
            TextColumn("[bold blue]{task.fields[epoch_name]}"),
            "•",
            TextColumn("[red]{task.fields[iou_loss]}"),
            "•",
            TextColumn("[red]{task.fields[dfl_loss]}"),
            "•",
            TextColumn("[red]{task.fields[cls_loss]}"),
            "•",
            TextColumn("[red]{task.fields[ang_loss]}"),
            BarColumn(
                bar_width=None,
                style="white",
                complete_style="blue",
                finished_style="green",
            ),
            TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
            "•",
            TimeElapsedColumn(),
            "•",
            TimeRemainingColumn(),
            transient=True,
            expand=True,
        )
        if self.distill_ns:
            self.loss_num = 6
            self.loss_info = [
                "Epoch",
                "iou_loss",
                "dfl_loss",
                "class",
                "ang_cls_loss",
                "angle_reg_loss",
                "cwd",
            ]
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.fields[epoch_name]}"),
                "•",
                TextColumn("[red]{task.fields[iou_loss]}"),
                "•",
                TextColumn("[red]{task.fields[dfl_loss]}"),
                "•",
                TextColumn("[red]{task.fields[cls_loss]}"),
                "•",
                TextColumn("[red]{task.fields[ang_cls_loss]}"),
                "•",
                TextColumn("[red]{task.fields[ang_reg_loss]}"),
                "•",
                TextColumn("[red]{task.fields[cwd_loss]}"),
                BarColumn(
                    bar_width=None,
                    style="white",
                    complete_style="blue",
                    finished_style="green",
                ),
                TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                "•",
                TimeElapsedColumn(),
                "•",
                TimeRemainingColumn(),
                transient=True,
                expand=True,
            )
        if self.args.distill:
            self.loss_num += 1
            self.loss_info += ["cwd_loss"]
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[bold blue]{task.fields[epoch_name]}"),
                "•",
                TextColumn("[red]{task.fields[iou_loss]}"),
                "•",
                TextColumn("[red]{task.fields[dfl_loss]}"),
                "•",
                TextColumn("[red]{task.fields[cls_loss]}"),
                "•",
                TextColumn("[red]{task.fields[ang_loss]}"),
                "•",
                TextColumn("[red]{task.fields[cwd_loss]}"),
                BarColumn(
                    bar_width=None,
                    style="white",
                    complete_style="blue",
                    finished_style="green",
                ),
                TextColumn("[progress.percentage]{task.percentage:>3.1f}%"),
                "•",
                TimeElapsedColumn(),
                "•",
                TimeRemainingColumn(),
                transient=True,
                expand=True,
            )

    # Training Process
    def train(self):
        try:
            self.train_before_loop()
            for self.epoch in range(self.start_epoch, self.max_epoch):
                self.train_in_loop(self.epoch)
            self.strip_model()

        except Exception as _:
            LOGGER.error("ERROR in training loop or eval/save model.")
            raise
        finally:
            self.train_after_loop()

    # Training loop for each epoch
    def train_in_loop(self, epoch_num):
        try:
            self.prepare_for_steps()
            for self.step, self.batch_data in self.pbar:
                if self.main_process:
                    self.progress.advance(self.task)
                self.train_in_steps(epoch_num, self.step)
                self.print_details()
            if self.main_process:
                self.progress.remove_task(self.task)
                self.progress.stop()
            LOGGER.info(("\n" + "%10s" * (self.loss_num + 1)) % (*self.loss_info,))
            LOGGER.info(
                ("\n" + "%10g" * (self.loss_num + 1))
                % (
                    self.epoch,
                    *self.mean_loss,
                )
            )
        except Exception as _:
            LOGGER.error("ERROR in training steps.")
            raise
        try:
            self.eval_and_save()
        except Exception as _:
            LOGGER.error("ERROR in evaluate and save model.")
            raise

    # Training loop for batchdata
    def train_in_steps(self, epoch_num, step_num):
        # NOTE images and targets
        # NOTE Targets: torch [num_labels_all_batchs, 7] [bs_id, class_id, x, y, w, h, angle] 相对值
        images, targets = self.prepro_data(self.batch_data, self.device)
        # plot train_batch and save to tensorboard once an epoch
        if self.write_trainbatch_tb and self.main_process and self.step == 0:
            # TODO
            self.plot_train_batch(images, targets)
            write_tbimg(
                self.tblogger,
                self.vis_train_batch,
                self.step + self.max_stepnum * self.epoch,
                type="train",
            )

        # forward
        with amp.autocast(enabled=self.device != "cpu"):
            preds, s_featmaps = self.model(images)
            if self.args.distill:
                with torch.no_grad():
                    t_preds, t_featmaps = self.teacher_model(images)
                temperature = self.args.temperature
                total_loss, loss_items = self.compute_loss_distill(
                    preds,
                    t_preds,
                    s_featmaps,
                    t_featmaps,
                    targets,
                    epoch_num,
                    self.max_epoch,
                    temperature,
                    step_num,
                )

            elif self.args.fuse_ab:
                total_loss, loss_items = self.compute_loss(
                    (preds[0], preds[3], preds[4]), targets, epoch_num, step_num
                )  # YOLOv6_af
                total_loss_ab, loss_items_ab = self.compute_loss_ab(
                    preds[:3], targets, epoch_num, step_num
                )  # YOLOv6_ab
                total_loss += total_loss_ab
                loss_items += loss_items_ab
            else:
                total_loss, loss_items = self.compute_loss(
                    preds, targets, epoch_num, step_num
                )  # YOLOv6_af
            if self.rank != -1:
                total_loss *= self.world_size
        # backward
        self.scaler.scale(total_loss).backward()
        self.loss_items = loss_items
        self.update_optimizer()

    def eval_and_save(self):
        remaining_epochs = self.max_epoch - self.epoch
        eval_interval = (
            self.args.eval_interval
            if remaining_epochs > self.args.heavy_eval_range
            else 3
        )
        is_val_epoch = (
            (not self.args.eval_final_only or (remaining_epochs == 1))
            and (self.epoch % eval_interval == 0)
            and (self.epoch != 0)
        )
        if self.main_process:
            self.ema.update_attr(
                self.model, include=["nc", "names", "stride"]
            )  # update attributes for ema model
            if is_val_epoch:
                self.eval_model()
                self.ap = self.evaluate_results[1]
                self.best_ap = max(self.ap, self.best_ap)
            # save ckpt
            ckpt = {
                "model": deepcopy(de_parallel(self.model)).half(),
                "ema": deepcopy(self.ema.ema).half(),
                "updates": self.ema.updates,
                "optimizer": self.optimizer.state_dict(),
                "epoch": self.epoch,
            }

            save_ckpt_dir = osp.join(self.save_dir, "weights")
            save_checkpoint(
                ckpt,
                (is_val_epoch) and (self.ap == self.best_ap),
                save_ckpt_dir,
                model_name="last_ckpt",
            )
            if self.epoch >= self.max_epoch - self.args.save_ckpt_on_last_n_epoch:
                save_checkpoint(
                    ckpt, False, save_ckpt_dir, model_name=f"{self.epoch}_ckpt"
                )

            # default save best ap ckpt in stop strong aug epochs
            if self.epoch >= self.max_epoch - self.args.stop_aug_last_n_epoch:
                if self.best_stop_strong_aug_ap < self.ap:
                    self.best_stop_strong_aug_ap = max(
                        self.ap, self.best_stop_strong_aug_ap
                    )
                    save_checkpoint(
                        ckpt, False, save_ckpt_dir, model_name="best_stop_aug_ckpt"
                    )

            del ckpt
            # log for learning rate
            lr = [x["lr"] for x in self.optimizer.param_groups]
            self.evaluate_results = list(self.evaluate_results) + lr

            # log for tensorboard
            write_tblog(
                self.tblogger, self.epoch, self.evaluate_results, self.mean_loss
            )
            # save validation predictions to tensorboard
            write_tbimg(self.tblogger, self.vis_imgs_list, self.epoch, type="val")

    def eval_model(self):
        if not hasattr(self.cfg, "eval_params"):
            results, vis_outputs, vis_paths = eval.run(
                self.data_dict,
                batch_size=self.batch_size // self.world_size * 2,
                img_size=self.img_size,
                model=self.ema.ema if self.args.calib is False else self.model,
                conf_thres=0.03,
                dataloader=self.val_loader,
                save_dir=self.save_dir,
                task="train",
                angle_max=self.cfg.model.head.angle_max,
                angle_fitting_methods=self.cfg.model.head.angle_fitting_methods,
                ap_method="VOC12",
            )
        else:

            def get_cfg_value(cfg_dict, value_str, default_value):
                if value_str in cfg_dict:
                    if isinstance(cfg_dict[value_str], list):
                        return (
                            cfg_dict[value_str][0]
                            if cfg_dict[value_str][0] is not None
                            else default_value
                        )
                    else:
                        return (
                            cfg_dict[value_str]
                            if cfg_dict[value_str] is not None
                            else default_value
                        )
                else:
                    return default_value

            eval_img_size = get_cfg_value(
                self.cfg.eval_params, "img_size", self.img_size
            )
            results, vis_outputs, vis_paths = eval.run(
                self.data_dict,
                batch_size=get_cfg_value(
                    self.cfg.eval_params,
                    "batch_size",
                    self.batch_size // self.world_size * 2,
                ),
                img_size=eval_img_size,
                model=self.ema.ema if self.args.calib is False else self.model,
                conf_thres=get_cfg_value(self.cfg.eval_params, "conf_thres", 0.03),
                dataloader=self.val_loader,
                save_dir=self.save_dir,
                task="train",
                test_load_size=get_cfg_value(
                    self.cfg.eval_params, "test_load_size", eval_img_size
                ),
                letterbox_return_int=get_cfg_value(
                    self.cfg.eval_params, "letterbox_return_int", False
                ),
                force_no_pad=get_cfg_value(self.cfg.eval_params, "force_no_pad", False),
                not_infer_on_rect=get_cfg_value(
                    self.cfg.eval_params, "not_infer_on_rect", False
                ),
                scale_exact=get_cfg_value(self.cfg.eval_params, "scale_exact", False),
                verbose=get_cfg_value(self.cfg.eval_params, "verbose", False),
                do_coco_metric=get_cfg_value(
                    self.cfg.eval_params, "do_coco_metric", False
                ),
                do_pr_metric=get_cfg_value(self.cfg.eval_params, "do_pr_metric", True),
                plot_curve=get_cfg_value(self.cfg.eval_params, "plot_curve", False),
                plot_confusion_matrix=get_cfg_value(
                    self.cfg.eval_params, "plot_confusion_matrix", False
                ),
                angle_max=self.cfg.model.head.angle_max,
                angle_fitting_methods=self.cfg.model.head.angle_fitting_methods,
                ap_method=get_cfg_value(self.cfg.eval_params, "ap_method", False),
            )

        LOGGER.info(
            f"Epoch: {self.epoch} | mAP@0.5: {results[0]} | mAP@0.50:0.95: {results[1]}"
        )
        self.evaluate_results = results[:2]
        # plot validation predictions
        self.plot_val_pred(vis_outputs, vis_paths)

    def train_before_loop(self):
        LOGGER.info("Training start...")
        self.start_time = time.time()
        self.warmup_stepnum = (
            max(round(self.cfg.solver.warmup_epochs * self.max_stepnum), 1000)
            if self.args.quant is False
            else 0
        )
        self.scheduler.last_epoch = self.start_epoch - 1
        self.last_opt_step = -1
        self.scaler = amp.GradScaler(enabled=self.device != "cpu")

        self.best_ap, self.ap = 0.0, 0.0
        self.best_stop_strong_aug_ap = 0.0
        self.evaluate_results = (0, 0)  # AP50, AP50_95

        # NOTE Loss 计算
        # REVIEW
        self.compute_loss = ComputeLoss(
            num_classes=self.data_dict["nc"],
            ori_img_size=self.img_size,
            warmup_epoch=self.cfg.model.head.atss_warmup_epoch,
            use_dfl=self.cfg.model.head.use_dfl,
            reg_max=self.cfg.model.head.reg_max,
            iou_type=self.cfg.model.head.iou_type,
            fpn_strides=self.cfg.model.head.strides,
            loss_weight=self.cfg.loss.loss_weight,
            angle_max=self.cfg.model.head.angle_max,
            angle_fitting_methods=self.cfg.model.head.angle_fitting_methods,
        )

        if self.args.fuse_ab:
            self.compute_loss_ab = ComputeLoss_ab(
                num_classes=self.data_dict["nc"],
                ori_img_size=self.img_size,
                warmup_epoch=0,
                use_dfl=False,
                reg_max=0,
                iou_type=self.cfg.model.head.iou_type,
                fpn_strides=self.cfg.model.head.strides,
            )
        # NOTE
        if self.args.distill:
            # NOTE n/s 所使用的蒸馏函数不一样，原因在HEAD部分
            if self.distill_ns:
                Loss_distill_func = ComputeLoss_distill_ns()
                self.compute_loss_distill = Loss_distill_func(
                    num_classes=self.data_dict["nc"],
                    ori_img_size=self.img_size,
                    fpn_strides=self.cfg.model.head.strides,
                    warmup_epoch=self.cfg.model.head.atss_warmup_epoch,
                    use_dfl=self.cfg.model.head.use_dfl,
                    reg_max=self.cfg.model.head.reg_max,
                    angle_max=self.cfg.model.head.angle_max,
                    angle_fitting_methods=self.cfg.model.head.angle_fitting_methods,
                    iou_type=self.cfg.model.head.iou_type,
                    distill_weight=self.cfg.model.head.distill_weight,
                    distill_feat=self.args.distill_feat,
                    loss_weight=self.cfg.loss.loss_weight,
                )
            else:
                Loss_distill_func = ComputeLoss_distill
                self.compute_loss_distill = Loss_distill_func(
                    num_classes=self.data_dict["nc"],
                    ori_img_size=self.img_size,
                    fpn_strides=self.cfg.model.head.strides,
                    warmup_epoch=self.cfg.model.head.atss_warmup_epoch,
                    use_dfl=self.cfg.model.head.use_dfl,
                    reg_max=self.cfg.model.head.reg_max,
                    angle_max=self.cfg.model.head.angle_max,
                    angle_fitting_methods=self.cfg.model.head.angle_fitting_methods,
                    iou_type=self.cfg.model.head.iou_type,
                    distill_weight=self.cfg.model.head.distill_weight,
                    distill_feat=self.args.distill_feat,
                    loss_weight=self.cfg.loss.loss_weight,
                )

    def prepare_for_steps(self):
        if self.epoch > self.start_epoch:
            self.scheduler.step()
        # stop strong aug like mosaic and mixup from last n epoch by recreate dataloader
        if self.epoch == self.max_epoch - self.args.stop_aug_last_n_epoch:
            self.cfg.data_aug.mosaic = 0.0
            self.cfg.data_aug.mixup = 0.0
            self.train_loader, self.val_loader = self.get_data_loader(
                self.args, self.cfg, self.data_dict
            )
        self.model.train()
        if self.rank != -1:
            self.train_loader.sampler.set_epoch(self.epoch)
        self.mean_loss = torch.zeros(self.loss_num, device=self.device)
        self.optimizer.zero_grad()

        # TODO
        # LOGGER.info(("\n" + "%10s" * (self.loss_num + 1)) % (*self.loss_info,))

        self.pbar = enumerate(self.train_loader)
        if self.main_process:
            # self.pbar = tqdm(
            #     self.pbar, total=self.max_stepnum, ncols=NCOLS, bar_format="{l_bar}{bar:10}{r_bar}{bar:-10b}"
            # )
            if not self.args.distill:
                self.task = self.progress.add_task(
                    "LOGGER",
                    total=len(self.train_loader),
                    epoch_name=f"Epoch {self.epoch}/{self.max_epoch - 1}",
                    iou_loss=f"iou{self.mean_loss[0]:7.4g}",
                    dfl_loss=f"dfl{self.mean_loss[1]:7.4g}",
                    cls_loss=f"cls{self.mean_loss[2]:7.4g}",
                    ang_loss=f"angle{self.mean_loss[3]:7.4g}",
                )
            else:
                self.task = self.progress.add_task(
                    "LOGGER",
                    total=len(self.train_loader),
                    epoch_name=f"Epoch {self.epoch}/{self.max_epoch - 1}",
                    iou_loss=f"iou{self.mean_loss[0]:7.4g}",
                    dfl_loss=f"dfl{self.mean_loss[1]:7.4g}",
                    cls_loss=f"cls{self.mean_loss[2]:7.4g}",
                    ang_loss=f"angle{self.mean_loss[3]:7.4g}",
                    cwd_loss=f"cwd{self.mean_loss[4]:7.4g}",
                )
            self.progress.start()
            self.progress.start_task(self.task)

    # Print loss after each steps
    def print_details(self):
        if self.main_process:
            self.mean_loss = (self.mean_loss * self.step + self.loss_items) / (
                self.step + 1
            )
            # TODO
            # self.pbar.set_description(
            #     ("%10s" + "%10.4g" * self.loss_num) % (f"{self.epoch}/{self.max_epoch - 1}", *(self.mean_loss))
            # )
            if not self.args.distill:
                self.progress.update(
                    self.task,
                    iou_loss=f"iou{self.mean_loss[0]:7.4g}",
                    dfl_loss=f"dfl{self.mean_loss[1]:7.4g}",
                    cls_loss=f"cls{self.mean_loss[2]:7.4g}",
                    ang_loss=f"angle{self.mean_loss[3]:7.4g}",
                )
            else:
                self.progress.update(
                    self.task,
                    iou_loss=f"iou{self.mean_loss[0]:7.4g}",
                    dfl_loss=f"dfl{self.mean_loss[1]:7.4g}",
                    cls_loss=f"cls{self.mean_loss[2]:7.4g}",
                    ang_loss=f"angle{self.mean_loss[3]:7.4g}",
                    cwd_loss=f"cwd{self.mean_loss[4]:7.4g}",
                )

    def strip_model(self):
        if self.main_process:
            LOGGER.info(
                f"\nTraining completed in {(time.time() - self.start_time) / 3600:.3f} hours."
            )
            save_ckpt_dir = osp.join(self.save_dir, "weights")
            strip_optimizer(
                save_ckpt_dir, self.epoch
            )  # strip optimizers for saved pt model

    # Empty cache if training finished
    def train_after_loop(self):
        if self.device != "cpu":
            torch.cuda.empty_cache()

    def update_optimizer(self):
        curr_step = self.step + self.max_stepnum * self.epoch
        self.accumulate = max(1, round(64 / self.batch_size))
        if curr_step <= self.warmup_stepnum:
            self.accumulate = max(
                1,
                np.interp(
                    curr_step, [0, self.warmup_stepnum], [1, 64 / self.batch_size]
                ).round(),
            )
            for k, param in enumerate(self.optimizer.param_groups):
                warmup_bias_lr = self.cfg.solver.warmup_bias_lr if k == 2 else 0.0
                param["lr"] = np.interp(
                    curr_step,
                    [0, self.warmup_stepnum],
                    [warmup_bias_lr, param["initial_lr"] * self.lf(self.epoch)],
                )
                if "momentum" in param:
                    param["momentum"] = np.interp(
                        curr_step,
                        [0, self.warmup_stepnum],
                        [self.cfg.solver.warmup_momentum, self.cfg.solver.momentum],
                    )
        if curr_step - self.last_opt_step >= self.accumulate:
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.optimizer.zero_grad()
            if self.ema:
                self.ema.update(self.model)
            self.last_opt_step = curr_step

    @staticmethod
    def get_data_loader(args, cfg, data_dict):
        # REVIEW
        train_path, val_path = data_dict["train"], data_dict["val"]
        # check data
        nc = int(data_dict["nc"])
        class_names = data_dict["names"]
        assert (
            len(class_names) == nc
        ), f"the length of class names does not match the number of classes defined"
        grid_size = max(int(max(cfg.model.head.strides)), 32)
        # create train dataloader
        train_loader = create_dataloader(
            train_path,
            args.img_size,
            args.batch_size // args.world_size,
            grid_size,
            hyp=dict(cfg.data_aug),
            augment=True,
            rect=False,
            rank=args.local_rank,
            workers=args.workers,
            shuffle=True,
            check_images=args.check_images,
            check_labels=args.check_labels,
            data_dict=data_dict,
            task="train",
        )[0]
        # create val dataloader
        val_loader = None
        if args.rank in [-1, 0]:
            val_loader = create_dataloader(
                val_path,
                args.img_size,
                args.batch_size // args.world_size * 2,
                grid_size,
                hyp=dict(cfg.data_aug),
                rect=True,
                rank=-1,
                pad=0.5,
                workers=args.workers,
                check_images=args.check_images,
                check_labels=args.check_labels,
                data_dict=data_dict,
                task="val",
            )[0]

        return train_loader, val_loader

    @staticmethod
    def prepro_data(batch_data, device):
        images = batch_data[0].to(device, non_blocking=True).float() / 255
        targets = batch_data[1].to(device)
        return images, targets

    def get_model(self, args, cfg, nc, device):
        model = build_model(
            cfg, nc, device, fuse_ab=self.args.fuse_ab, distill_ns=False
        )
        weights = cfg.model.pretrained
        # NOTE load weights
        # TODO 预训练
        if weights:  # finetune if pretrained model is set
            if not os.path.exists(weights):
                download_ckpt(weights)
            LOGGER.info(f"Loading state_dict from {weights} for fine-tuning...")
            model = load_state_dict(weights, model, map_location=device)

        # NOTE 隐藏
        # LOGGER.info("Model: {}".format(model))
        return model

    def get_teacher_model(self, args, cfg, nc, device):
        teacher_fuse_ab = False if cfg.model.head.num_layers != 3 else True
        # NOTE default fuse ab turn off
        teacher_fuse_ab = False
        model = build_model(cfg, nc, device, fuse_ab=teacher_fuse_ab)
        weights = args.teacher_model_path
        if weights:  # finetune if pretrained model is set
            LOGGER.info(f"Loading state_dict from {weights} for teacher")
            model = load_state_dict(weights, model, map_location=device)
        # NOTE 隐藏
        # LOGGER.info("Model: {}".format(model))
        # Do not update running means and running vars
        for module in model.modules():
            if isinstance(module, torch.nn.BatchNorm2d):
                module.track_running_stats = False
        # NOTE BatchNorm 不更新
        return model

    @staticmethod
    def load_scale_from_pretrained_models(cfg, device):
        weights = cfg.model.scales
        scales = None
        if not weights:
            LOGGER.error("ERROR: No scales provided to init RepOptimizer!")
        else:
            ckpt = torch.load(weights, map_location=device)
            scales = extract_scales(ckpt)
        return scales

    @staticmethod
    def parallel_model(args, model, device):
        # If DP mode
        dp_mode = device.type != "cpu" and args.rank == -1
        if dp_mode and torch.cuda.device_count() > 1:
            LOGGER.warning("WARNING: DP not recommended, use DDP instead.\n")
            model = torch.nn.DataParallel(model)

        # If DDP mode
        ddp_mode = device.type != "cpu" and args.rank != -1
        if ddp_mode:
            model = DDP(
                model, device_ids=[args.local_rank], output_device=args.local_rank
            )

        return model

    def get_optimizer(self, args, cfg, model):
        # NOTE batchsize gpu_cont 影响
        accumulate = max(1, round(64 / args.batch_size))
        cfg.solver.weight_decay *= args.batch_size * accumulate / 64
        cfg.solver.lr0 *= args.batch_size / (
            self.world_size * args.bs_per_gpu
        )  # rescale lr0 related to batchsize
        optimizer = build_optimizer(cfg, model)
        return optimizer

    @staticmethod
    def get_lr_scheduler(args, cfg, optimizer):
        epochs = args.epochs
        lr_scheduler, lf = build_lr_scheduler(cfg, optimizer, epochs)
        return lr_scheduler, lf

    def plot_train_batch(self, images, targets, max_size=1920, max_subplots=16):
        # Plot train_batch with labels
        if isinstance(images, torch.Tensor):
            images = images.cpu().float().numpy()
        if isinstance(targets, torch.Tensor):
            targets = targets.cpu().numpy()
        if np.max(images[0]) <= 1:
            images *= 255  # de-normalise (optional)
        bs, _, h, w = images.shape  # batch size, _, height, width
        bs = min(bs, max_subplots)  # limit plot images
        ns = np.ceil(bs**0.5)  # number of subplots (square)
        paths = self.batch_data[2]  # image paths
        # Build Image
        mosaic = np.full((int(ns * h), int(ns * w), 3), 255, dtype=np.uint8)  # init
        for i, im in enumerate(images):
            if i == max_subplots:  # if last batch has fewer images than we expect
                break
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            im = im.transpose(1, 2, 0)
            mosaic[y : y + h, x : x + w, :] = im
        # Resize (optional)
        scale = max_size / ns / max(h, w)
        if scale < 1:
            h = math.ceil(scale * h)
            w = math.ceil(scale * w)
            mosaic = cv2.resize(mosaic, tuple(int(x * ns) for x in (w, h)))
        for i in range(bs):
            x, y = int(w * (i // ns)), int(h * (i % ns))  # block origin
            cv2.rectangle(
                mosaic, (x, y), (x + w, y + h), (255, 255, 255), thickness=2
            )  # borders
            cv2.putText(
                mosaic,
                f"{os.path.basename(paths[i])[:40]}",
                (x + 5, y + 15),
                cv2.FONT_HERSHEY_COMPLEX,
                0.5,
                color=(220, 220, 220),
                thickness=1,
            )  # filename
            if len(targets) > 0:
                ti = targets[targets[:, 0] == i]  # image targets
                boxes = ti[:, 2:6].T
                angles = ti[:, 6:7]
                classes = ti[:, 1].astype("int")
                labels = ti.shape[1] == 7  # labels if no conf column
                if boxes.shape[1]:
                    if boxes.max() <= 1.01:  # if normalized with tolerance 0.01
                        boxes[[0, 2]] *= w  # scale to pixels
                        boxes[[1, 3]] *= h
                    elif scale < 1:  # absolute coords need scale if image scales
                        boxes *= scale
                boxes[[0]] += x
                boxes[[1]] += y
                # boxes_xywh = xyxy2xywh(boxes.T)
                for j, (box, angle) in enumerate(
                    zip(boxes.T.tolist(), angles.tolist())
                ):
                    box = [int(k) for k in box]
                    cls = classes[j]
                    color = tuple([int(x) for x in self.color[cls]])
                    cls = (
                        self.data_dict["names"][cls] if self.data_dict["names"] else cls
                    )
                    if labels:
                        label = f"{cls}"
                        rect = ((box[0], box[1]), (box[2], box[3]), int(angle[0]))
                        poly = cv2.boxPoints(rect)
                        poly = np.int0(poly)
                        cv2.drawContours(
                            mosaic,
                            contours=[poly],
                            contourIdx=-1,
                            color=color,
                            thickness=2,
                            lineType=cv2.LINE_AA,
                        )
                        cv2.putText(
                            mosaic,
                            label,
                            (int(box[0] - 5), int(box[1] - 5)),
                            cv2.FONT_HERSHEY_COMPLEX,
                            0.5,
                            color,
                            thickness=2,
                        )
        self.vis_train_batch = mosaic.copy()

    def plot_val_pred(self, vis_outputs, vis_paths, vis_conf=0.3, vis_max_box_num=100):
        # plot validation predictions
        # TODO test
        self.vis_imgs_list = []
        for (vis_output, vis_path) in zip(vis_outputs, vis_paths):
            vis_output_array = vis_output.cpu().numpy()  # xyxy
            ori_img = cv2.imread(vis_path)
            for bbox_idx, vis_bbox in enumerate(vis_output_array):
                cx = int(vis_bbox[0])
                cy = int(vis_bbox[1])
                w = int(vis_bbox[2])
                h = int(vis_bbox[3])
                angle = int(vis_bbox[4])
                box_score = vis_bbox[5]
                cls_id = int(vis_bbox[6])
                # draw top n bbox
                if box_score < vis_conf or bbox_idx > vis_max_box_num:
                    break
                rect = ((cx, cy), (w, h), angle)
                poly = cv2.boxPoints(rect)
                poly = np.int0(poly)
                cv2.drawContours(
                    ori_img,
                    contours=[poly],
                    contourIdx=-1,
                    color=tuple([int(x) for x in self.color[cls_id]]),
                    thickness=2,
                    lineType=cv2.LINE_AA,
                )
                cv2.putText(
                    ori_img,
                    f"{self.data_dict['names'][cls_id]}: {box_score:.2f}",
                    (cx - 5, cy - 5),
                    cv2.FONT_HERSHEY_COMPLEX,
                    0.5,
                    tuple([int(x) for x in self.color[cls_id]]),
                    thickness=2,
                )
            self.vis_imgs_list.append(torch.from_numpy(ori_img[:, :, ::-1].copy()))

    # PTQ
    def calibrate(self, cfg):
        def save_calib_model(model, cfg):
            # Save calibrated checkpoint
            output_model_path = os.path.join(
                cfg.ptq.calib_output_path,
                "{}_calib_{}.pt".format(
                    os.path.splitext(os.path.basename(cfg.model.pretrained))[0],
                    cfg.ptq.calib_method,
                ),
            )
            if cfg.ptq.sensitive_layers_skip is True:
                output_model_path = output_model_path.replace(".pt", "_partial.pt")
            LOGGER.info("Saving calibrated model to {}... ".format(output_model_path))
            if not os.path.exists(cfg.ptq.calib_output_path):
                os.mkdir(cfg.ptq.calib_output_path)
            torch.save(
                {"model": deepcopy(de_parallel(model)).half()}, output_model_path
            )

        assert self.args.quant is True and self.args.calib is True
        if self.main_process:
            from tools.qat.qat_utils import ptq_calibrate

            ptq_calibrate(self.model, self.train_loader, cfg)
            self.epoch = 0
            self.eval_model()
            save_calib_model(self.model, cfg)

    # QAT
    def quant_setup(self, model, cfg, device):
        if self.args.quant:
            from tools.qat.qat_utils import (qat_init_model_manu,
                                             skip_sensitive_layers)

            qat_init_model_manu(model, cfg, self.args)
            # workaround
            model.neck.upsample_enable_quant(cfg.ptq.num_bits, cfg.ptq.calib_method)
            # if self.main_process:
            #     print(model)
            # QAT
            if self.args.calib is False:
                if cfg.qat.sensitive_layers_skip:
                    skip_sensitive_layers(model, cfg.qat.sensitive_layers_list)
                # QAT flow load calibrated model
                assert cfg.qat.calib_pt is not None, "Please provide calibrated model"
                model.load_state_dict(
                    torch.load(cfg.qat.calib_pt)["model"].float().state_dict()
                )
            model.to(device)
