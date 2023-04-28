import numpy as np
import cv2
import argparse
from numpy import ndarray
from typing import List
import math
import ncnn
import sys
import os

ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
MAJOR, MINOR = map(int, cv2.__version__.split('.')[:2])
assert MAJOR == 4


def softmax(x: ndarray, axis: int = -1) -> ndarray:
    e_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
    y = e_x / e_x.sum(axis=axis, keepdims=True)
    return y


def sigmoid(x: ndarray) -> ndarray:
    return 1. / (1. + np.exp(-x))


CLASS_NAMES = ('person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
               'train', 'truck', 'boat', 'traffic light', 'fire hydrant',
               'stop sign', 'parking meter', 'bench', 'bird', 'cat', 'dog',
               'horse', 'sheep', 'cow', 'elephant', 'bear', 'zebra', 'giraffe',
               'backpack', 'umbrella', 'handbag', 'tie', 'suitcase', 'frisbee',
               'skis', 'snowboard', 'sports ball', 'kite', 'baseball bat',
               'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
               'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
               'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
               'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch',
               'potted plant', 'bed', 'dining table', 'toilet', 'tv', 'laptop',
               'mouse', 'remote', 'keyboard', 'cell phone', 'microwave',
               'oven', 'toaster', 'sink', 'refrigerator', 'book', 'clock',
               'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush')

CLASS_COLORS = [(220, 20, 60), (119, 11, 32), (0, 0, 142), (0, 0, 230),
                (106, 0, 228), (0, 60, 100), (0, 80, 100), (0, 0, 70),
                (0, 0, 192), (250, 170, 30), (100, 170, 30), (220, 220, 0),
                (175, 116, 175), (250, 0, 30), (165, 42, 42), (255, 77, 255),
                (0, 226, 252), (182, 182, 255), (0, 82, 0), (120, 166, 157),
                (110, 76, 0), (174, 57, 255), (199, 100, 0), (72, 0, 118),
                (255, 179, 240), (0, 125, 92), (209, 0, 151), (188, 208, 182),
                (0, 220, 176), (255, 99, 164), (92, 0, 73), (133, 129, 255),
                (78, 180, 255), (0, 228, 0), (174, 255, 243), (45, 89, 255),
                (134, 134, 103), (145, 148, 174), (255, 208, 186),
                (197, 226, 255), (171, 134, 1), (109, 63, 54), (207, 138, 255),
                (151, 0, 95), (9, 80, 61), (84, 105, 51), (74, 65, 105),
                (166, 196, 102), (208, 195, 210), (255, 109, 65),
                (0, 143, 149), (179, 0, 194), (209, 99, 106), (5, 121, 0),
                (227, 255, 205), (147, 186, 208), (153, 69, 1), (3, 95, 161),
                (163, 255, 0), (119, 0, 170), (0, 182, 199), (0, 165, 120),
                (183, 130, 88), (95, 32, 0), (130, 114, 135), (110, 129, 133),
                (166, 74, 118), (219, 142, 185), (79, 210, 114), (178, 90, 62),
                (65, 70, 15), (127, 167, 115), (59, 105, 106), (142, 108, 45),
                (196, 172, 0), (95, 54, 80), (128, 76, 255), (201, 57, 1),
                (246, 0, 122), (191, 162, 208)]

MASK_COLORS = np.array([(255, 56, 56), (255, 157, 151), (255, 112, 31),
                        (255, 178, 29), (207, 210, 49), (72, 249, 10),
                        (146, 204, 23), (61, 219, 134), (26, 147, 52),
                        (0, 212, 187), (44, 153, 168), (0, 194, 255),
                        (52, 69, 147), (100, 115, 255), (0, 24, 236),
                        (132, 56, 255), (82, 0, 133), (203, 56, 255),
                        (255, 149, 200), (255, 55, 199)],
                       dtype=np.uint8)

CONF_THRES = 0.45
IOU_THRES = 0.65


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('img', help='Image files')
    parser.add_argument('param', help='NCNN param file')
    parser.add_argument('bin', help='NCNN bin file')
    parser.add_argument('--show', action='store_true', help='Show image result')
    parser.add_argument(
        '--out-dir', default='./output', help='Path to output file')
    parser.add_argument(
        '--img-size',
        nargs='+',
        type=int,
        default=[320, 320],
        help='Image size of height and width')
    parser.add_argument(
        '--max-stride',
        type=int,
        default=64,
        help='Max stride of yolov6 model')
    args = parser.parse_args()
    assert args.max_stride in (32, 64)
    return args


def yolov6_decode(feats: List[ndarray],
                  conf_thres: float,
                  iou_thres: float,
                  num_labels: int = 80,
                  **kwargs):
    proposal_boxes: List[ndarray] = []
    proposal_scores: List[float] = []
    proposal_labels: List[int] = []
    for i, feat in enumerate(feats):
        feat = np.ascontiguousarray(feat.transpose((1, 2, 0)))
        stride = 8 << i
        score_feat, box_feat = np.split(feat, [
            num_labels,
        ], -1)
        _argmax = score_feat.argmax(-1)
        _max = score_feat.max(-1)
        indices = np.where(_max > conf_thres)
        hIdx, wIdx = indices
        num_proposal = hIdx.size
        if not num_proposal:
            continue

        scores = _max[hIdx, wIdx]
        boxes = box_feat[hIdx, wIdx]
        labels = _argmax[hIdx, wIdx]

        for k in range(num_proposal):
            score = scores[k]
            label = labels[k]

            x0, y0, x1, y1 = boxes[k]

            x0 = (wIdx[k] + 0.5 - x0) * stride
            y0 = (hIdx[k] + 0.5 - y0) * stride
            x1 = (wIdx[k] + 0.5 + x1) * stride
            y1 = (hIdx[k] + 0.5 + y1) * stride

            w = x1 - x0
            h = y1 - y0

            proposal_scores.append(float(score))
            proposal_boxes.append(
                np.array([x0, y0, w, h], dtype=np.float32))
            proposal_labels.append(int(label))

    if MINOR >= 7:
        indices = cv2.dnn.NMSBoxesBatched(proposal_boxes, proposal_scores, proposal_labels, conf_thres,
                                          iou_thres)
    elif MINOR == 6:
        indices = cv2.dnn.NMSBoxes(proposal_boxes, proposal_scores, conf_thres, iou_thres)
    else:
        indices = cv2.dnn.NMSBoxes(proposal_boxes, proposal_scores, conf_thres, iou_thres).flatten()

    if not len(indices):
        return [], [], []

    nmsd_boxes: List[ndarray] = []
    nmsd_scores: List[float] = []
    nmsd_labels: List[int] = []
    for idx in indices:
        box = proposal_boxes[idx]
        box[2:] = box[:2] + box[2:]
        score = proposal_scores[idx]
        label = proposal_labels[idx]
        nmsd_boxes.append(box)
        nmsd_scores.append(score)
        nmsd_labels.append(label)
    return nmsd_boxes, nmsd_scores, nmsd_labels


def main(args: argparse.Namespace):
    image_path = args.img
    net_h, net_w = args.img_size

    if not args.show and not os.path.exists(args.out_dir):
        os.makedirs(args.out_dir, exist_ok=True)

    net = ncnn.Net()
    # use gpu or not
    net.opt.use_vulkan_compute = False
    net.opt.num_threads = 4
    net.load_param(args.param)
    net.load_model(args.bin)

    ex = net.create_extractor()
    img = cv2.imread(image_path)
    draw_img = img.copy()
    img_w = img.shape[1]
    img_h = img.shape[0]

    w = img_w
    h = img_h
    scale = 1.0
    if w > h:
        scale = float(net_w) / w
        w = net_w
        h = int(h * scale)
    else:
        scale = float(net_h) / h
        h = net_h
        w = int(w * scale)

    mat_in = ncnn.Mat.from_pixels_resize(
        img, ncnn.Mat.PixelType.PIXEL_BGR2RGB, img_w, img_h, w, h
    )

    wpad = (w + args.max_stride - 1) // args.max_stride * args.max_stride - w
    hpad = (h + args.max_stride - 1) // args.max_stride * args.max_stride - h

    mat_in_pad = ncnn.copy_make_border(
        mat_in,
        hpad // 2,
        hpad - hpad // 2,
        wpad // 2,
        wpad - wpad // 2,
        ncnn.BorderType.BORDER_CONSTANT,
        114.0,
    )

    mat_in_pad.substract_mean_normalize([0, 0, 0], [1 / 225, 1 / 225, 1 / 225])

    ex.input('in0', mat_in_pad)

    ret1, mat_out1 = ex.extract("out0")  # stride 8
    ret2, mat_out2 = ex.extract("out1")  # stride 16
    ret3, mat_out3 = ex.extract("out2")  # stride 32
    if args.max_stride == 64:
        ret4, mat_out4 = ex.extract("out3")  # stride 64

    outputs = [np.array(mat_out1), np.array(mat_out2), np.array(mat_out3)]
    if args.max_stride == 64:
        outputs.append(np.array(mat_out4))

    nmsd_boxes, nmsd_scores, nmsd_labels = yolov6_decode(outputs, CONF_THRES, IOU_THRES)

    for box, score, label in zip(nmsd_boxes, nmsd_scores, nmsd_labels):
        x0, y0, x1, y1 = box
        x0 = x0 - (wpad / 2)
        y0 = y0 - (hpad / 2)
        x1 = x1 - (wpad / 2)
        y1 = y1 - (hpad / 2)
        name = CLASS_NAMES[label]
        box_color = CLASS_COLORS[label]

        x0 = math.floor(min(max(x0 / scale, 1), img_w - 1))
        y0 = math.floor(min(max(y0 / scale, 1), img_h - 1))
        x1 = math.ceil(min(max(x1 / scale, 1), img_w - 1))
        y1 = math.ceil(min(max(y1 / scale, 1), img_h - 1))
        cv2.rectangle(draw_img, (x0, y0), (x1, y1), box_color, 2)
        cv2.putText(draw_img, f'{name}: {score:.2f}',
                    (x0, max(y0 - 5, 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (0, 255, 255), 2)
    if args.show:
        cv2.imshow('res', draw_img)
        cv2.waitKey(0)
    else:
        cv2.imwrite(os.path.join(args.out_dir, os.path.basename(image_path)), draw_img)


if __name__ == '__main__':
    main(parse_args())
