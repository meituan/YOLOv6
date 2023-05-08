#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import argparse
import os
import os.path as osp
import sys
import torch

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.core.evaler import Evaler
from yolov6.utils.events import LOGGER
from yolov6.utils.general import increment_name, check_img_size
from yolov6.utils.config import Config

def boolean_string(s):
    if s not in {'False', 'True'}:
        raise ValueError('Not a valid boolean string')
    return s == 'True'

def get_args_parser(add_help=True):
    parser = argparse.ArgumentParser(description='YOLOv6 PyTorch Evalating', add_help=add_help)
    parser.add_argument('--data', type=str, default='./data/coco.yaml', help='dataset.yaml path')
    parser.add_argument('--weights', type=str, default='./weights/yolov6s.pt', help='model.pt path(s)')
    parser.add_argument('--batch-size', type=int, default=32, help='batch size')
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.03, help='confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.65, help='NMS IoU threshold')
    parser.add_argument('--task', default='val', help='val, test, or speed')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--half', default=False, action='store_true', help='whether to use fp16 infer')
    parser.add_argument('--save_dir', type=str, default='runs/val/', help='evaluation save dir')
    parser.add_argument('--name', type=str, default='exp', help='save evaluation results to save_dir/name')
    parser.add_argument('--shrink_size', type=int, default=0, help='load img resize when test')
    parser.add_argument('--infer_on_rect', default=True, type=boolean_string, help='default to run with rectangle image to boost speed.')
    parser.add_argument('--reproduce_640_eval', default=False, action='store_true', help='whether to reproduce 640 infer result, overwrite some config')
    parser.add_argument('--eval_config_file', type=str, default='./configs/experiment/eval_640_repro.py', help='config file for repro 640 infer result')
    parser.add_argument('--do_coco_metric', default=True, type=boolean_string, help='whether to use pycocotool to metric, set False to close')
    parser.add_argument('--do_pr_metric', default=False, type=boolean_string, help='whether to calculate precision, recall and F1, n, set False to close')
    parser.add_argument('--plot_curve', default=True, type=boolean_string, help='whether to save plots in savedir when do pr metric, set False to close')
    parser.add_argument('--plot_confusion_matrix', default=False, action='store_true', help='whether to save confusion matrix plots when do pr metric, might cause no harm warning print')
    parser.add_argument('--verbose', default=False, action='store_true', help='whether to print metric on each class')
    parser.add_argument('--config-file', default='', type=str, help='experiments description file, lower priority than reproduce_640_eval')
    parser.add_argument('--specific-shape', action='store_true', help='rectangular training')
    parser.add_argument('--height', type=int, default=None, help='image height of model input')
    parser.add_argument('--width', type=int, default=None, help='image width of model input')
    args = parser.parse_args()

    if args.config_file:
        assert os.path.exists(args.config_file), print("Config file {} does not exist".format(args.config_file))
        cfg = Config.fromfile(args.config_file)
        if not hasattr(cfg, 'eval_params'):
            LOGGER.info("Config file doesn't has eval params config.")
        else:
            eval_params=cfg.eval_params
            for key, value in eval_params.items():
                if key not in args.__dict__:
                    LOGGER.info(f"Unrecognized config {key}, continue")
                    continue
                if isinstance(value, list):
                    if value[1] is not None:
                        args.__dict__[key] = value[1]
                else:
                    if value is not None:
                        args.__dict__[key] = value

    # load params for reproduce 640 eval result
    if args.reproduce_640_eval:
        assert os.path.exists(args.eval_config_file), print("Reproduce config file {} does not exist".format(args.eval_config_file))
        eval_params = Config.fromfile(args.eval_config_file).eval_params
        eval_model_name = os.path.splitext(os.path.basename(args.weights))[0]
        if eval_model_name not in eval_params:
            eval_model_name = "default"
        args.shrink_size = eval_params[eval_model_name]["shrink_size"]
        args.infer_on_rect = eval_params[eval_model_name]["infer_on_rect"]
        #force params
        #args.img_size = 640
        args.conf_thres = 0.03
        args.iou_thres = 0.65
        args.task = "val"
        args.do_coco_metric = True

    LOGGER.info(args)
    return args


@torch.no_grad()
def run(data,
        weights=None,
        batch_size=32,
        img_size=640,
        conf_thres=0.03,
        iou_thres=0.65,
        task='val',
        device='',
        half=False,
        model=None,
        dataloader=None,
        save_dir='',
        name = '',
        shrink_size=640,
        letterbox_return_int=False,
        infer_on_rect=False,
        reproduce_640_eval=False,
        eval_config_file='./configs/experiment/eval_640_repro.py',
        verbose=False,
        do_coco_metric=True,
        do_pr_metric=False,
        plot_curve=False,
        plot_confusion_matrix=False,
        config_file=None,
        specific_shape=False,
        height=640,
        width=640
        ):
    """ Run the evaluation process

    This function is the main process of evaluation, supporting image file and dir containing images.
    It has tasks of 'val', 'train' and 'speed'. Task 'train' processes the evaluation during training phase.
    Task 'val' processes the evaluation purely and return the mAP of model.pt. Task 'speed' processes the
    evaluation of inference speed of model.pt.

    """

     # task
    Evaler.check_task(task)
    if task == 'train':
        save_dir = save_dir
    else:
        save_dir = str(increment_name(osp.join(save_dir, name)))
        os.makedirs(save_dir, exist_ok=True)

    # check the threshold value, reload device/half/data according task
    Evaler.check_thres(conf_thres, iou_thres, task)
    device = Evaler.reload_device(device, model, task)
    half = device.type != 'cpu' and half
    data = Evaler.reload_dataset(data, task) if isinstance(data, str) else data

    # # verify imgsz is gs-multiple
    if specific_shape:
        height = check_img_size(height, 32, floor=256)
        width = check_img_size(width, 32, floor=256)
    else:
        img_size = check_img_size(img_size, 32, floor=256)
    val = Evaler(data, batch_size, img_size, conf_thres, \
                iou_thres, device, half, save_dir, \
                shrink_size, infer_on_rect,
                verbose, do_coco_metric, do_pr_metric,
                plot_curve, plot_confusion_matrix,
                specific_shape=specific_shape,height=height, width=width)
    model = val.init_model(model, weights, task)
    dataloader = val.init_data(dataloader, task)

    # eval
    model.eval()
    pred_result, vis_outputs, vis_paths = val.predict_model(model, dataloader, task)
    eval_result = val.eval_model(pred_result, model, dataloader, task)
    return eval_result, vis_outputs, vis_paths


def main(args):
    run(**vars(args))


if __name__ == "__main__":
    args = get_args_parser()
    main(args)
