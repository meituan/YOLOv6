#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os
import yaml
import logging
import shutil
import cv2
from rich.logging import RichHandler
class OBRichHandler(RichHandler):
    # Note KEYWORDS highlight use regex
    KEYWORDS = [
        "mAP",
        "Epoch",
        "iou_loss",
        "cls_loss",
        "dfl_loss",
        "ang_loss",
        "distill_loss",
    ]


class RichStreamLogger(object):
    """stream logger based rich.logging.RichHandler [__call__]

    Args:
        logger_name (str, optional): [description]. Defaults to 'OB_rich'.
        log_level (str, optional): [description]. Defaults to 'DEBUG'.

    Returns:
        logging.Logger: [description]
    """

    def __call__(self, logger_name: str = "OB_rich", log_level: str = "DEBUG") -> logging.Logger:
        # * create logging
        logger = logging.getLogger(logger_name)
        logger.setLevel(log_level)

        # * StreamHandler -> rich
        console_handler = OBRichHandler(
            rich_tracebacks=True, tracebacks_show_locals=True, log_time_format="[%Y-%m-%d %H:%M:%S:%S]"
        )

        rich_formatter = logging.Formatter("%(message)s")
        console_handler.setFormatter(rich_formatter)
        logger.addHandler(console_handler)
        # Note logger.propagate 设置为True在多进程情况下shell会输出两次
        logger.propagate = False

        return logger


class RichStreamAndFileLogger(object):
    """stream and file logger [__call__]
    Args:
        log_file_name (str, optional): [description]. Defaults to './demo.log'.
        logger_name (str, optional): [description]. Defaults to 'OB_rich'.
        log_level (str, optional): [description]. Defaults to 'DEBUG'.

    Returns:
        logging.Logger: [new looger]
    """

    def __init__(self) -> None:
        self.logger_name = None
        self.log_file_name = None

    def __call__(
        self, log_file_name: str = "./demo.log", logger_name: str = "OB_rich", log_level: str = "DEBUG"
    ) -> logging.Logger:
        # * create logging
        self.log_file_name = log_file_name
        self.logger_name = logger_name
        logger = logging.getLogger(self.logger_name)
        logger.setLevel(log_level)

        # * FileHandler -> File
        # * StreamHandler -> rich
        file_handler = logging.FileHandler(self.log_file_name, encoding="utf-8")
        console_handler = OBRichHandler(
            rich_tracebacks=True, tracebacks_show_locals=True, log_time_format="[%m-%d %H:%M]"
        )

        file_formatter = logging.Formatter(
            "%(levelname)s %(asctime)s [%(filename)s:%(funcName)s:%(lineno)d] %(message)s"
        )
        rich_formatter = logging.Formatter("%(message)s")

        file_handler.setFormatter(file_formatter)
        console_handler.setFormatter(rich_formatter)

        logger.addHandler(file_handler)
        logger.addHandler(console_handler)

        # Note logger.propagate 设置为True在多进程情况下shell会输出两次
        logger.propagate = False

        return logger


def set_logging(name=None):
    rank = int(os.getenv("RANK", -1))
    logging.basicConfig(format="%(message)s", level=logging.INFO if (rank in (-1, 0)) else logging.WARNING)
    return logging.getLogger(name)


# LOGGER = set_logging(__name__)
LOGGER = RichStreamAndFileLogger()()
NCOLS = min(100, shutil.get_terminal_size().columns)


def load_yaml(file_path):
    """Load data from yaml file."""
    if isinstance(file_path, str):
        with open(file_path, errors="ignore") as f:
            data_dict = yaml.safe_load(f)
    return data_dict


def save_yaml(data_dict, save_path):
    """Save data to yaml file"""
    with open(save_path, "w") as f:
        yaml.safe_dump(data_dict, f, sort_keys=False)


def write_tblog(tblogger, epoch, results, losses):
    """Display mAP and loss information to log."""
    tblogger.add_scalar("val/mAP@0.5", results[0], epoch + 1)
    tblogger.add_scalar("val/mAP@0.50:0.95", results[1], epoch + 1)

    tblogger.add_scalar("train/iou_loss", losses[0], epoch + 1)
    tblogger.add_scalar("train/dist_focalloss", losses[1], epoch + 1)
    tblogger.add_scalar("train/cls_loss", losses[2], epoch + 1)
    tblogger.add_scalar("train/angle_loss", losses[3], epoch + 1)

    tblogger.add_scalar("x/lr0", results[2], epoch + 1)
    tblogger.add_scalar("x/lr1", results[3], epoch + 1)
    tblogger.add_scalar("x/lr2", results[4], epoch + 1)


def write_tbimg(tblogger, imgs, step, type="train"):
    """Display train_batch and validation predictions to tensorboard."""
    if type == "train":
        tblogger.add_image(f"train_batch", imgs, step + 1, dataformats="HWC")
        # TODO

    elif type == "val":
        for idx, img in enumerate(imgs):
            tblogger.add_image(f"val_img_{idx + 1}", img, step + 1, dataformats="HWC")
            # TODO

    else:
        LOGGER.warning("WARNING: Unknown image type to visualize.\n")
