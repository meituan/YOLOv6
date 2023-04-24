"""visualize.py

This script is for visualization of YOLO models.
"""
import os
import sys
import json
import argparse
import math

import cv2
import torch
from tensorrt_processor import Processor
from tqdm import tqdm


def parse_args():
    """Parse input arguments."""
    desc = 'Visualization of YOLO TRT model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--imgs-dir', type=str, default='./coco_images/',
        help='directory of to be visualized images ./coco_images/')
    parser.add_argument(
        '--visual-dir', type=str, default='./visual_out',
        help='directory of visualized images ./visual_out')
    parser.add_argument('--batch-size', type=int,
                        default=1, help='batch size for training: default 64')
    parser.add_argument(
        '-c', '--category-num', type=int, default=80,
        help='number of object categories [80]')
    parser.add_argument(
        '--img-size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument(
        '-m', '--model', type=str, default='./weights/yolov5s-simple.trt',
        help=('trt model path'))
    parser.add_argument(
        '--conf-thres', type=float, default=0.03,
        help='object confidence threshold')
    parser.add_argument(
        '--iou-thres', type=float, default=0.65,
        help='IOU threshold for NMS')
    parser.add_argument('--shrink_size', type=int, default=6, help='load img with size (img_size - shrink_size), for better performace.')
    args = parser.parse_args()
    return args


def check_args(args):
    """Check and make sure command-line arguments are valid."""
    if not os.path.isdir(args.imgs_dir):
        sys.exit('%s is not a valid directory' % args.imgs_dir)
    if not os.path.exists(args.visual_dir):
        print("Directory {} does not exist, create it".format(args.visual_dir))
        os.makedirs(args.visual_dir)


def generate_results(processor, imgs_dir, visual_dir, jpgs, conf_thres, iou_thres,
                     batch_size=1, img_size=[640,640], shrink_size=0):
    """Run detection on each jpg and write results to file."""
    results = []
    # pbar = tqdm(jpgs, desc="TRT-Model test in val datasets.")
    pbar = tqdm(range(math.ceil(len(jpgs) / batch_size)), desc="TRT-Model test in val datasets.")
    idx = 0
    num_visualized = 0
    for _ in pbar:
        imgs = torch.randn((batch_size, 3, 640, 640), dtype=torch.float32, device=torch.device('cuda:0'))
        source_imgs = []
        image_names = []
        shapes = []
        for i in range(batch_size):
            if (idx == len(jpgs)): break
            img = cv2.imread(os.path.join(imgs_dir, jpgs[idx]))
            img_src = img.copy()
            # shapes.append(img.shape)
            h0, w0 = img.shape[:2]
            r = (max(img_size) - shrink_size) / max(h0, w0)
            if r != 1:
                img = cv2.resize(
                    img,
                    (int(w0 * r), int(h0 * r)),
                    interpolation=cv2.INTER_AREA
                    if r < 1 else cv2.INTER_LINEAR,
                )
            h, w = img.shape[:2]
            imgs[i], pad = processor.pre_process(img)
            source_imgs.append(img_src)
            shape = (h0, w0), ((h / h0, w / w0), pad)
            shapes.append(shape)
            image_names.append(jpgs[idx])
            idx += 1
        output = processor.inference(imgs)

        for j in range(len(shapes)):
            pred = processor.post_process(output[j].unsqueeze(0), shapes[j], conf_thres=conf_thres, iou_thres=iou_thres)
            image = source_imgs[j]
            for p in pred:
                x = float(p[0])
                y = float(p[1])
                w = float(p[2] - p[0])
                h = float(p[3] - p[1])
                s = float(p[4])

                cv2.rectangle(image, (int(x), int(y)), (int(x + w), int(y + h)), (255, 0, 0), 1)

            # print("saving to {}".format(os.path.join(visual_dir, image_names[j])))
            cv2.imwrite("{}".format(os.path.join(visual_dir, image_names[j])), image)

def main():
    args = parse_args()
    check_args(args)

    assert args.model.endswith('.trt'), "Only support trt engine test"

    # setup processor
    processor = Processor(model=args.model)
    jpgs = [j for j in os.listdir(args.imgs_dir) if j.endswith('.jpg')]
    generate_results(processor, args.imgs_dir, args.visual_dir, jpgs, args.conf_thres, args.iou_thres,
                    batch_size=args.batch_size, img_size = args.img_size, shrink_size=args.shrink_size)


if __name__ == '__main__':
    main()
