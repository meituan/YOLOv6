"""eval_yolo.py

This script is for evaluating mAP (accuracy) of YOLO models.
"""
import os
import sys
import json
import argparse
import math

import cv2
import torch
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
from Processor import Processor
from tqdm import tqdm


coco91class = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
     21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
     41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
     59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
     80, 81, 82, 84, 85, 86, 87, 88, 89, 90]


def parse_args():
    """Parse input arguments."""
    desc = 'Evaluate mAP of YOLO TRT model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        '--imgs-dir', type=str, default='../coco/images/val2017',
        help='directory of validation images ../coco/images/val2017')
    parser.add_argument(
        '--annotations', type=str, default='../coco/annotations/instances_val2017.json',
        help='groundtruth annotations ../coco/annotations/instances_val2017.json')
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
    parser.add_argument('--test_load_size', type=int, default=634, help='load img resize when test')
    parser.add_argument('--letterbox_return_int', type=bool, default=True, help='return int offset for letterbox')
    parser.add_argument('--scale_exact', type=bool, default=True, help='use exact scale size to scale coords')
    parser.add_argument('--force_no_pad', type=bool, default=True, help='for no extra pad in letterbox')
    parser.add_argument('-v', '--visualize', action="store_true", default=False, help='visualize demo')
    parser.add_argument('--num_imgs_to_visualize', type=int, default=10, help='number of images to visualize')
    args = parser.parse_args()
    return args


def check_args(args):
    """Check and make sure command-line arguments are valid."""
    if not os.path.isdir(args.imgs_dir):
        sys.exit('%s is not a valid directory' % args.imgs_dir)
    if not os.path.isfile(args.annotations):
        sys.exit('%s is not a valid file' % args.annotations)


def generate_results(processor, imgs_dir, jpgs, results_file, conf_thres, iou_thres, non_coco,
                    batch_size=1, test_load_size=640, visualize=False,  num_imgs_to_visualize=0):
    """Run detection on each jpg and write results to file."""
    results = []
    # pbar = tqdm(jpgs, desc="TRT-Model test in val datasets.")
    pbar = tqdm(range(math.ceil(len(jpgs)/batch_size)), desc="TRT-Model test in val datasets.")
    idx = 0
    num_visualized = 0
    for _ in pbar:
        imgs = torch.randn((batch_size,3,640,640), dtype=torch.float32, device=torch.device('cuda:0'))
        source_imgs = []
        image_ids = []
        shapes = []
        for i in range(batch_size):
            if (idx == len(jpgs)): break
            img = cv2.imread(os.path.join(imgs_dir, jpgs[idx]))
            img_src = img.copy()
            # shapes.append(img.shape)
            h0, w0 = img.shape[:2]
            r = test_load_size / max(h0, w0)
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
            image_ids.append(int(jpgs[idx].split('.')[0].split('_')[-1]))
            idx += 1
        output = processor.inference(imgs)
        
        for j in range(len(shapes)):
            pred = processor.post_process(output[j].unsqueeze(0), shapes[j], conf_thres=conf_thres, iou_thres=iou_thres)
            if visualize and num_visualized < num_imgs_to_visualize:
                image = source_imgs[i]
            for p in pred:
                x = float(p[0])
                y = float(p[1])
                w = float(p[2] - p[0])
                h = float(p[3] - p[1])
                s = float(p[4])
                results.append({'image_id': image_ids[j],
                                'category_id': coco91class[int(p[5])] if not non_coco else int(p[5]),
                                'bbox': [round(x, 3) for x in [x, y, w, h]],
                                'score': round(s, 5)})

                if visualize and num_visualized < num_imgs_to_visualize:
                    cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 1)

            if visualize and num_visualized < num_imgs_to_visualize:
                print("saving to %d.jpg" % (num_visualized))
                err_code = cv2.imwrite("./%d.jpg"%num_visualized, image)
                num_visualized += 1

    with open(results_file, 'w') as f:
        f.write(json.dumps(results, indent=4))


def main():
    args = parse_args()
    check_args(args)

    if args.model.endswith('.onnx'):
        from onnx_to_trt import build_engine_from_onnx
        engine = build_engine_from_onnx(args.model, 'fp32', False)
        args.model = args.model.replace('.onnx', '.trt')

        with open(args.model, 'wb') as f:
            f.write(engine.serialize())
        print('Serialized the TensorRT engine to file: %s' % args.model)

    model_prefix = args.model.replace('.trt', '').split('/')[-1]
    results_file = 'results_{}.json'.format(model_prefix)

    # setup processor
    processor = Processor(model=args.model, scale_exact=args.scale_exact, return_int=args.letterbox_return_int, force_no_pad=args.force_no_pad)
    jpgs = [j for j in os.listdir(args.imgs_dir) if j.endswith('.jpg')]
    generate_results(processor, args.imgs_dir, jpgs, results_file, args.conf_thres, args.iou_thres,
                     non_coco=False, batch_size=args.batch_size, test_load_size=args.test_load_size,
                     visualize=args.visualize, num_imgs_to_visualize=args.num_imgs_to_visualize)


    # Run COCO mAP evaluation
    # Reference: https://github.com/cocodataset/cocoapi/blob/master/PythonAPI/pycocoEvalDemo.ipynb
    cocoGt = COCO(args.annotations)
    cocoDt = cocoGt.loadRes(results_file)
    imgIds = sorted(cocoGt.getImgIds())
    cocoEval = COCOeval(cocoGt, cocoDt, 'bbox')
    cocoEval.params.imgIds = imgIds
    cocoEval.evaluate()
    cocoEval.accumulate()
    cocoEval.summarize()


if __name__ == '__main__':
    main()
