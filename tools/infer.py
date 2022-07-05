#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import sys
import wget
import wandb
import urllib
import os.path as osp
from pathlib import Path
from typing import Optional

import torch

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.utils.events import LOGGER
from yolov6.core.inferer import Inferer
from yolov6.logger.wandb_inference_logger import WandbInferenceLogger


def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(
        description="YOLOv6 PyTorch Inference.", add_help=add_help
    )
    parser.add_argument(
        "--weights",
        type=str,
        default="weights/yolov6s.pt",
        help="model path(s) for inference.",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/images",
        help="the source path, e.g. image-file/dir.",
    )
    parser.add_argument(
        "--yaml", type=str, default="data/coco.yaml", help="data yaml file."
    )
    parser.add_argument(
        "--img-size",
        type=int,
        default=640,
        help="the image-size(h,w) in inference size.",
    )
    parser.add_argument(
        "--conf-thres",
        type=float,
        default=0.25,
        help="confidence threshold for inference.",
    )
    parser.add_argument(
        "--iou-thres", type=float, default=0.45, help="NMS IoU threshold for inference."
    )
    parser.add_argument(
        "--max-det", type=int, default=1000, help="maximal inferences per image."
    )
    parser.add_argument(
        "--device",
        default="0",
        help="device to run our model i.e. 0 or 0,1,2,3 or cpu.",
    )
    parser.add_argument(
        "--save-txt", action="store_true", help="save results to *.txt."
    )
    parser.add_argument(
        "--save-img", action="store_false", help="save visuallized inference results."
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        type=int,
        help="filter by classes, e.g. --classes 0, or --classes 0 2 3.",
    )
    parser.add_argument(
        "--agnostic-nms", action="store_true", help="class-agnostic NMS."
    )
    parser.add_argument(
        "--project",
        default="runs/inference",
        help="save inference results to project/name.",
    )
    parser.add_argument(
        "--name",
        type=lambda x: None if x == "None" else str(x),
        nargs="?",
        default=None,
        help="save inference results to project/name.",
    )
    parser.add_argument(
        "--hide-labels", default=False, action="store_true", help="hide labels."
    )
    parser.add_argument(
        "--hide-conf", default=False, action="store_true", help="hide confidences."
    )
    parser.add_argument(
        "--half",
        action="store_true",
        help="whether to use FP16 half-precision inference.",
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default=None,
        help="Name of Weights & Biases Project.",
    )
    parser.add_argument(
        "--wandb_entity",
        type=str,
        default=None,
        help="Name of Weights & Biases Entity.",
    )

    args = parser.parse_args()
    LOGGER.info(args)
    return args


@torch.no_grad()
def run(
    weights=osp.join(ROOT, "yolov6s.pt"),
    source=osp.join(ROOT, "data/images"),
    yaml=None,
    img_size=640,
    conf_thres=0.25,
    iou_thres=0.45,
    max_det=1000,
    device="",
    save_txt=False,
    save_img=True,
    classes=None,
    agnostic_nms=False,
    project=osp.join(ROOT, "runs/inference"),
    name="exp",
    hide_labels=False,
    hide_conf=False,
    half=False,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
):
    """Inference process

    This function is the main process of inference, supporting image files or dirs containing images.

    Args:
        weights: The path of model.pt, e.g. yolov6s.pt
        source: Source path, supporting image files or dirs containing images.
        yaml: Data yaml file, .
        img_size: Inference image-size, e.g. 640
        conf_thres: Confidence threshold in inference, e.g. 0.25
        iou_thres: NMS IOU threshold in inference, e.g. 0.45
        max_det: Maximal detections per image, e.g. 1000
        device: Cuda device, e.e. 0, or 0,1,2,3 or cpu
        save_txt: Save results to *.txt
        save_img: Save visualized inference results
        classes: Filter by class: --class 0, or --class 0 2 3
        agnostic_nms: Class-agnostic NMS
        project: Save results to project/name
        name: Save results to project/name, e.g. 'exp'
        line_thickness: Bounding box thickness (pixels), e.g. 3
        hide_labels: Hide labels, e.g. False
        hide_conf: Hide confidences
        half: Use FP16 half-precision inference, e.g. False
    """

    if not osp.isfile(weights) and not osp.isfile(weights + ".pt"):
        try:
            print("Downloading weights...")
            weights = wget.download(
                f"https://github.com/meituan/YOLOv6/releases/download/0.1.0/{weights}.pt"
            )
            print("\nDone.")
        except urllib.error.HTTPError:
            print("Unable to download model.")
    elif osp.isfile(weights + ".pt"):
        weights += ".pt"

    if not osp.isfile(source) and not osp.isdir(source):
        try:
            print("Downloading image...")
            source = wget.download(source)
            print("\nDone.")
        except urllib.error.HTTPError:
            print("Unable to download image.")

    # Initialize Weights & Biases
    if wandb_project is not None:
        wandb.init(
            project=wandb_project,
            name=name if name is not None else None,
            entity=wandb_entity,
        )

        if name is None:
            name = wandb.run.name

        config = wandb.config
        config.weights = Path(weights).name
        config.source = source
        config.yaml = yaml
        config.img_size = img_size
        config.conf_thres = conf_thres
        config.iou_thres = iou_thres
        config.max_det = max_det
        config.device = device
        config.save_txt = save_txt
        config.save_img = save_img
        config.classes = classes
        config.agnostic_nms = agnostic_nms
        config.hide_labels = hide_labels
        config.hide_conf = hide_conf
        config.half = half
    elif name is None:
        raise ValueError(
            "`name` must be explicitly specified in case `wandb_project` is None."
        )

    # create save dir
    save_dir = osp.join(project, name)
    if (save_img or save_txt) and not osp.exists(save_dir):
        os.makedirs(save_dir)
    else:
        LOGGER.warning("Save directory already existed")
    if save_txt:
        os.mkdir(osp.join(save_dir, "labels"))

    # Inference
    inferer = Inferer(
        source,
        weights,
        device,
        yaml,
        img_size,
        half,
        inference_logger=WandbInferenceLogger() if wandb.run is not None else None,
    )
    inferer.infer(
        conf_thres,
        iou_thres,
        classes,
        agnostic_nms,
        max_det,
        save_dir,
        save_txt,
        save_img,
        hide_labels,
        hide_conf,
    )

    if save_txt or save_img:
        LOGGER.info(f"Results saved to {save_dir}")

    if wandb.run is not None:
        wandb.finish()


def main(args):
    run(**vars(args))


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
