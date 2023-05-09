"""
This script is used for evaluating the performance of YOLOv6 TensorRT models.
"""
import os
import sys
import json
import argparse
import math
import cv2
import torch
import numpy as np
from tqdm import tqdm
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from tensorrt_processor import Processor

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from yolov6.utils.events import LOGGER

IMG_FORMATS = ["bmp", "jpg", "jpeg", "png", "tif", "tiff", "dng", "webp", "mpo"]
IMG_FORMATS.extend([f.upper() for f in IMG_FORMATS])

def parse_args():
    """Parse input arguments."""
    desc = 'Evaluate mAP of YOLOv6 TensorRT model'
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument('--imgs_dir', type=str, default='../coco/images/val2017',
        help='directory of validation dataset images.')
    parser.add_argument('--labels_dir', type=str, default='../coco/labels/val2017',
        help='directory of validation dataset labels.')
    parser.add_argument('--annotations', type=str, default='../coco/annotations/instances_val2017.json',
        help='coco format annotations of validation dataset.')
    parser.add_argument('--batch_size', type=int,
        default=1, help='batch size of evaluation.')
    parser.add_argument('--img_size', nargs='+', type=int, default=[640, 640], help='image size')
    parser.add_argument('--model', '-m', type=str, default='./weights/yolov5s.trt',
        help=('trt model path'))
    parser.add_argument('--conf_thres', type=float, default=0.03,
        help='confidence threshold')
    parser.add_argument('--iou_thres', type=float, default=0.65,
        help='IOU threshold for NMS')
    parser.add_argument('--class_num', type=int, default=3, help='class list for general datasets that must be specified')
    parser.add_argument('--is_coco', action='store_true', help='whether the validation dataset is coco, default is False.')
    parser.add_argument('--shrink_size', type=int, default=4, help='load img with size (img_size - shrink_size), for better performace.')
    parser.add_argument('--visualize', '-v', action="store_true", default=False, help='visualize demo')
    parser.add_argument('--num_imgs_to_visualize', type=int, default=10, help='number of images to visualize')
    parser.add_argument('--do_pr_metric', action='store_true', help='use pr_metric to evaluate models')
    parser.add_argument('--plot_curve', type=bool, default=True, help='plot curve for pr_metric')
    parser.add_argument('--plot_confusion_matrix', action='store_true', help='plot confusion matrix ')
    parser.add_argument('--verbose', action='store_true', help='report mAP by class')
    parser.add_argument('--save_dir',default='', help='whether use pr_metric')
    parser.add_argument('--is_end2end', action='store_true', help='whether the model is end2end (build with NMS)')

    args = parser.parse_args()
    return args


def scale_coords(img1_shape, coords, img0_shape, ratio_pad=None):
    '''Rescale coords (xyxy) from img1_shape to img0_shape.'''

    gain = ratio_pad[0]
    pad = ratio_pad[1]

    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [0, 2]] /= gain[0]  # raw x gain
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, [1, 3]] /= gain[0]  # y gain

    if isinstance(coords, torch.Tensor):  # faster individually
        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
    else:  # np.array (faster grouped)
        coords[:, [0, 2]] = coords[:, [0, 2]].clip(0, img0_shape[1])  # x1, x2
        coords[:, [1, 3]] = coords[:, [1, 3]].clip(0, img0_shape[0])  # y1, y2
    return coords


def check_args(args):
    """Check and make sure command-line arguments are valid."""
    if not os.path.isdir(args.imgs_dir):
        sys.exit('%s is not a valid directory' % args.imgs_dir)
    if not os.path.isfile(args.annotations):
        sys.exit('%s is not a valid file' % args.annotations)


def generate_results(data_class,
                      model_names,
                      do_pr_metric,
                      plot_confusion_matrix,
                      processor,
                      imgs_dir,
                      labels_dir,
                      valid_images,
                      results_file,
                      conf_thres,
                      iou_thres,
                      is_coco,
                      batch_size=1,
                      img_size=[640, 640],
                      shrink_size=0,
                      visualize=False,
                      num_imgs_to_visualize=0,
                      imgname2id={}):
    """Run detection on each jpg and write results to file."""
    results = []
    pbar = tqdm(range(math.ceil(len(valid_images)/batch_size)), desc="TRT-Model test in val datasets.")
    idx = 0
    num_visualized = 0
    stats= []
    seen = 0
    if do_pr_metric:
            iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
            niou = iouv.numel()
            if plot_confusion_matrix:
                from yolov6.utils.metrics import ConfusionMatrix
                confusion_matrix = ConfusionMatrix(nc=len(model_names))
    for _ in pbar:
        preprocessed_imgs = []
        source_imgs = []
        image_ids = []
        shapes = []
        targets = []

        for i in range(batch_size):
            if (idx == len(valid_images)): break
            img = cv2.imread(os.path.join(imgs_dir, valid_images[idx]))
            imgs_name = os.path.splitext(valid_images[idx])[0]
            label_path = os.path.join(labels_dir, imgs_name+ '.txt')
            with open(label_path, "r") as f:
                    target = [
                        x.split() for x in f.read().strip().splitlines() if len(x)
                    ]
                    target = np.array(target ,dtype=np.float32)
                    targets.append(target)

            img_src = img.copy()
            h0, w0 = img.shape[:2]
            r = (max(img_size) - shrink_size) / max(h0, w0)
            if r != 1:
                img = cv2.resize(
                    img,
                    (int(w0 * r), int(h0 * r)),
                    interpolation = cv2.INTER_AREA
                    if r < 1 else cv2.INTER_LINEAR,
                )
            h, w = img.shape[:2]
            preprocessed_img, pad = processor.pre_process(img)
            preprocessed_imgs.append(preprocessed_img)
            source_imgs.append(img_src)
            shape = (h0, w0), ((h / h0, w / w0), pad)
            shapes.append(shape)
            assert valid_images[idx] in imgname2id.keys(), f'valid_images[idx] not in annotations you provided.'
            image_ids.append(imgname2id[valid_images[idx]])
            idx += 1
        output = processor.inference(torch.stack(preprocessed_imgs, axis=0))
        for j in range(len(shapes)):
            pred = processor.post_process(output[j].unsqueeze(0), shapes[j], conf_thres = conf_thres, iou_thres = iou_thres)

            if visualize and num_visualized < num_imgs_to_visualize:
                image = source_imgs[i]

            for p in pred:
                x = float(p[0])
                y = float(p[1])
                w = float(p[2] - p[0])
                h = float(p[3] - p[1])
                s = float(p[4])
                # Warning, some dataset, the category id is start from 1, so that the category id must add 1.
                # For example, change the line bellow to: 'category_id': data_class[int(p[5])] if is_coco else int(p[5]) + 1,
                results.append({'image_id': image_ids[j],
                                'category_id': data_class[int(p[5])] if is_coco else int(p[5]),
                                'bbox': [round(x, 3) for x in [x, y, w, h]],
                                'score': round(s, 5)})

                if visualize and num_visualized < num_imgs_to_visualize:
                    cv2.rectangle(image, (int(x), int(y)), (int(x+w), int(y+h)), (255, 0, 0), 1)

            if do_pr_metric:
                import copy
                target = targets[j]
                labels = target.copy()
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                if nl:
                    from yolov6.utils.nms import xywh2xyxy
                    # target boxes
                    tbox = xywh2xyxy(labels[:,1:5])
                    tbox[:, [0, 2]] *= shapes[j][0][1]
                    tbox[:, [1, 3]] *= shapes[j][0][0]

                    labelsn = torch.cat((torch.from_numpy(labels[:,0:1]).cpu(), torch.from_numpy(tbox).cpu()), 1)  # native-space labels

                    from yolov6.utils.metrics import process_batch

                    correct = process_batch(predn.cpu(), labelsn.cpu(), iouv)
                    if plot_confusion_matrix:
                        confusion_matrix.process_batch(predn, labelsn)
                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

            if visualize and num_visualized < num_imgs_to_visualize:
                print("saving to %d.jpg" % (num_visualized))
                err_code = cv2.imwrite("./%d.jpg"%num_visualized, image)
                num_visualized += 1

    with open(results_file, 'w') as f:
        LOGGER.info(f'saving coco format detection resuslt to {results_file}')
        f.write(json.dumps(results, indent=4))
    return stats, seen


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

    if args.is_coco:
        data_class = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
                21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
                41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
                59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
                80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        model_names = ['person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
                     'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign', 'parking meter',
                     'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra',
                     'giraffe', 'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee', 'skis',
                     'snowboard', 'sports ball', 'kite', 'baseball bat', 'baseball glove', 'skateboard',
                     'surfboard', 'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
                     'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
                     'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'dining table', 'toilet', 'tv',
                     'laptop', 'mouse', 'remote', 'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
                     'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush']
    else:
        data_class = list(range(0, args.class_num))
        model_names = list(range(0, args.class_num))

    # setup processor
    processor = Processor(model=args.model, is_end2end=args.is_end2end)
    image_names = [p for p in os.listdir(args.imgs_dir) if p.split(".")[-1].lower() in IMG_FORMATS]
    # Eliminate data with missing labels.
    with open(args.annotations) as f:
        coco_format_annotation = json.load(f)
    # Get image names from coco format annotations.
    coco_format_imgs = [x['file_name'] for x in coco_format_annotation['images']]
    # make a projection of image names and ids.
    imgname2id = {}
    for item in coco_format_annotation['images']:
        imgname2id[item['file_name']] = item['id']
    valid_images = []
    for img_name in image_names:
        img_name_wo_ext = os.path.splitext(img_name)[0]
        label_path = os.path.join(args.labels_dir, img_name_wo_ext + '.txt')
        if os.path.exists(label_path) and img_name in coco_format_imgs:
            valid_images.append(img_name)
        else:
            continue
    assert len(valid_images) > 0, 'No valid images are found. Please check you image format or whether annotation file is match.'
    #targets=[j for j in os.listdir(args.labels_dir) if j.endswith('.txt')]
    stats, seen = generate_results(data_class,
                                    model_names,
                                    args.do_pr_metric,
                                    args.plot_confusion_matrix,
                                    processor,
                                    args.imgs_dir,
                                    args.labels_dir,
                                    valid_images,
                                    results_file,
                                    args.conf_thres,
                                    args.iou_thres,
                                    args.is_coco,
                                    batch_size=args.batch_size,
                                    img_size = args.img_size,
                                    shrink_size=args.shrink_size,
                                    visualize=args.visualize,
                                    num_imgs_to_visualize=args.num_imgs_to_visualize,
                                    imgname2id=imgname2id)

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

    # Run PR_metric evaluation
    if args.do_pr_metric:
        # Compute statistics
        stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
        if len(stats) and stats[0].any():
            from yolov6.utils.metrics import ap_per_class
            p, r, ap, f1, ap_class = ap_per_class(*stats, plot=args.plot_curve, save_dir=args.save_dir, names=model_names)
            AP50_F1_max_idx = len(f1.mean(0)) - f1.mean(0)[::-1].argmax() -1
            LOGGER.info(f"IOU 50 best mF1 thershold near {AP50_F1_max_idx/1000.0}.")
            ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
            mp, mr, map50, map = p[:, AP50_F1_max_idx].mean(), r[:, AP50_F1_max_idx].mean(), ap50.mean(), ap.mean()
            nt = np.bincount(stats[3].astype(np.int64), minlength=len(model_names))  # number of targets per class

            # Print results
            s = ('%-16s' + '%12s' * 7) % ('Class', 'Images', 'Labels', 'P@.5iou', 'R@.5iou', 'F1@.5iou', 'mAP@.5', 'mAP@.5:.95')
            LOGGER.info(s)
            pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format
            LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, f1.mean(0)[AP50_F1_max_idx], map50, map))

            pr_metric_result = (map50, map)
            print("pr_metric results:", pr_metric_result)

            # Print results per class
            if args.verbose and len(model_names) > 1:
                for i, c in enumerate(ap_class):
                    LOGGER.info(pf % (model_names[c], seen, nt[c], p[i, AP50_F1_max_idx], r[i, AP50_F1_max_idx],
                                        f1[i, AP50_F1_max_idx], ap50[i], ap[i]))

            if args.plot_confusion_matrix:
                confusion_matrix.plot(save_dir=args.save_dir, names=list(model_names))
        else:
            LOGGER.info("Calculate metric failed, might check dataset.")
            pr_metric_result = (0.0, 0.0)


if __name__ == '__main__':
    main()
