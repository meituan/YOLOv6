import warnings
warnings.filterwarnings("ignore")
import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
import pkg_resources as pkg
import time
import cv2

import paddle
import onnx

sys.path.append("../")
from post_process import YOLOPostProcess, coco_metric
from dataset import COCOValDataset
import trt_backend


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--onnx_model_file',
        type=str,
        default='yolov6s_quant_onnx/quant_model.onnx',
        help="onnx model file path.")
    parser.add_argument(
        '--calibration_file',
        type=str,
        default='yolov6s_quant_onnx/calibration.cache',
        help="quant onnx model calibration cache file.")
    parser.add_argument(
        '--image_file',
        type=str,
        default=None,
        help="image path, if set image_file, it will not eval coco.")
    parser.add_argument(
        '--dataset_dir',
        type=str,
        default='dataset/coco',
        help="COCO dataset dir.")
    parser.add_argument(
        '--val_image_dir',
        type=str,
        default='val2017',
        help="COCO dataset val image dir.")
    parser.add_argument(
        '--val_anno_path',
        type=str,
        default='annotations/instances_val2017.json',
        help="COCO dataset anno path.")
    parser.add_argument(
        '--precision_mode',
        type=str,
        default='fp32',
        help="support fp32/fp16/int8.")
    parser.add_argument(
        '--batch_size', type=int, default=1, help="Batch size of model input.")

    return parser


# load coco labels
CLASS_LABEL = [
    "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train",
    "truck", "boat", "traffic light", "fire hydrant", "stop sign",
    "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
    "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag",
    "tie", "suitcase", "frisbee", "skis", "snowboard", "sports ball", "kite",
    "baseball bat", "baseball glove", "skateboard", "surfboard",
    "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon",
    "bowl", "banana", "apple", "sandwich", "orange", "broccoli", "carrot",
    "hot dog", "pizza", "donut", "cake", "chair", "couch", "potted plant",
    "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote",
    "keyboard", "cell phone", "microwave", "oven", "toaster", "sink",
    "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
    "hair drier", "toothbrush"
]


def preprocess(image, input_size, mean=None, std=None, swap=(2, 0, 1)):
    if len(image.shape) == 3:
        padded_img = np.ones((input_size[0], input_size[1], 3)) * 114.0
    else:
        padded_img = np.ones(input_size) * 114.0
    img = np.array(image)
    r = min(input_size[0] / img.shape[0], input_size[1] / img.shape[1])
    resized_img = cv2.resize(
        img,
        (int(img.shape[1] * r), int(img.shape[0] * r)),
        interpolation=cv2.INTER_LINEAR, ).astype(np.float32)
    padded_img[:int(img.shape[0] * r), :int(img.shape[1] * r)] = resized_img

    padded_img = padded_img[:, :, ::-1]
    padded_img /= 255.0
    if mean is not None:
        padded_img -= mean
    if std is not None:
        padded_img /= std
    padded_img = padded_img.transpose(swap)
    padded_img = np.ascontiguousarray(padded_img, dtype=np.float32)
    return padded_img, r


def get_color_map_list(num_classes):
    color_map = num_classes * [0, 0, 0]
    for i in range(0, num_classes):
        j = 0
        lab = i
        while lab:
            color_map[i * 3] |= (((lab >> 0) & 1) << (7 - j))
            color_map[i * 3 + 1] |= (((lab >> 1) & 1) << (7 - j))
            color_map[i * 3 + 2] |= (((lab >> 2) & 1) << (7 - j))
            j += 1
            lab >>= 3
    color_map = [color_map[i:i + 3] for i in range(0, len(color_map), 3)]
    return color_map


def draw_box(img, boxes, scores, cls_ids, conf=0.5, class_names=None):
    color_list = get_color_map_list(len(class_names))
    for i in range(len(boxes)):
        box = boxes[i]
        cls_id = int(cls_ids[i])
        color = tuple(color_list[cls_id])
        score = scores[i]
        if score < conf:
            continue
        x0 = int(box[0])
        y0 = int(box[1])
        x1 = int(box[2])
        y1 = int(box[3])

        text = '{}:{:.1f}%'.format(class_names[cls_id], score * 100)
        font = cv2.FONT_HERSHEY_SIMPLEX

        txt_size = cv2.getTextSize(text, font, 0.4, 1)[0]
        cv2.rectangle(img, (x0, y0), (x1, y1), color, 2)
        cv2.rectangle(img, (x0, y0 + 1),
                      (x0 + txt_size[0] + 1, y0 + int(1.5 * txt_size[1])),
                      color, -1)
        cv2.putText(
            img,
            text, (x0, y0 + txt_size[1]),
            font,
            0.8, (0, 255, 0),
            thickness=2)

    return img


def get_current_memory_mb():
    """
    It is used to Obtain the memory usage of the CPU and GPU during the running of the program.
    And this function Current program is time-consuming.
    """
    try:
        pkg.require('pynvml')
    except:
        from pip._internal import main
        main(['install', 'pynvml'])
    try:
        pkg.require('psutil')
    except:
        from pip._internal import main
        main(['install', 'psutil'])
    try:
        pkg.require('GPUtil')
    except:
        from pip._internal import main
        main(['install', 'GPUtil'])
    import pynvml
    import psutil
    import GPUtil
    gpu_id = int(os.environ.get('CUDA_VISIBLE_DEVICES', 0))

    pid = os.getpid()
    p = psutil.Process(pid)
    info = p.memory_full_info()
    cpu_mem = info.uss / 1024. / 1024.
    gpu_mem = 0
    gpu_percent = 0
    gpus = GPUtil.getGPUs()
    if gpu_id is not None and len(gpus) > 0:
        gpu_percent = gpus[gpu_id].load
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        meminfo = pynvml.nvmlDeviceGetMemoryInfo(handle)
        gpu_mem = meminfo.used / 1024. / 1024.
    return round(cpu_mem, 4), round(gpu_mem, 4)


def load_trt_engine():
    model = onnx.load(FLAGS.onnx_model_file)
    model_name = os.path.split(FLAGS.onnx_model_file)[-1].rstrip('.onnx')
    if FLAGS.precision_mode == "int8":
        engine_file = "{}_quant_model.trt".format(model_name)
        assert os.path.exists(FLAGS.calibration_file)
        trt_engine = trt_backend.TrtEngine(
            model,
            max_batch_size=1,
            precision_mode=FLAGS.precision_mode,
            engine_file_path=engine_file,
            calibration_cache_file=FLAGS.calibration_file)
    else:
        engine_file = "{}_{}_model.trt".format(model_name, FLAGS.precision_mode)
        trt_engine = trt_backend.TrtEngine(
            model,
            max_batch_size=1,
            precision_mode=FLAGS.precision_mode,
            engine_file_path=engine_file)
    return trt_engine


def eval():
    trt_engine = load_trt_engine()
    bboxes_list, bbox_nums_list, image_id_list = [], [], []
    cpu_mems, gpu_mems = 0, 0
    sample_nums = len(val_loader)
    with tqdm(
            total=sample_nums,
            bar_format='Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
            ncols=80) as t:
        for data in val_loader:
            data_all = {k: np.array(v) for k, v in data.items()}
            outs = trt_engine.infer([data_all['image']])
            outs = np.array(outs).reshape(1, -1, 85)
            postprocess = YOLOPostProcess(
                score_threshold=0.001, nms_threshold=0.65, multi_label=True)
            res = postprocess(np.array(outs), data_all['scale_factor'])
            bboxes_list.append(res['bbox'])
            bbox_nums_list.append(res['bbox_num'])
            image_id_list.append(np.array(data_all['im_id']))
            cpu_mem, gpu_mem = get_current_memory_mb()
            cpu_mems += cpu_mem
            gpu_mems += gpu_mem
            t.update()
    print('Avg cpu_mem:{} MB, avg gpu_mem: {} MB'.format(
        cpu_mems / sample_nums, gpu_mems / sample_nums))

    coco_metric(anno_file, bboxes_list, bbox_nums_list, image_id_list)


def infer():
    origin_img = cv2.imread(FLAGS.image_file)
    input_shape = [640, 640]
    input_image, scale_factor = preprocess(origin_img, input_shape)
    input_image = np.expand_dims(input_image, axis=0)
    scale_factor = np.array([[scale_factor, scale_factor]])
    trt_engine = load_trt_engine()

    repeat = 100
    cpu_mems, gpu_mems = 0, 0
    for _ in range(0, repeat):
        outs = trt_engine.infer(input_image)
        cpu_mem, gpu_mem = get_current_memory_mb()
        cpu_mems += cpu_mem
        gpu_mems += gpu_mem
    print('Avg cpu_mem:{} MB, avg gpu_mem: {} MB'.format(cpu_mems / repeat,
                                                         gpu_mems / repeat))
    # Do postprocess
    outs = np.array(outs).reshape(1, -1, 85)
    postprocess = YOLOPostProcess(
        score_threshold=0.1, nms_threshold=0.45, multi_label=False)
    res = postprocess(np.array(outs), scale_factor)
    # Draw rectangles and labels on the original image
    dets = res['bbox']
    if dets is not None:
        final_boxes, final_scores, final_class = dets[:, 2:], dets[:,
                                                                   1], dets[:,
                                                                            0]
        origin_img = draw_box(
            origin_img,
            final_boxes,
            final_scores,
            final_class,
            conf=0.5,
            class_names=CLASS_LABEL)
    cv2.imwrite('output.jpg', origin_img)
    print('The prediction results are saved in output.jpg.')


def main():
    if FLAGS.image_file:
        infer()
    else:
        global val_loader
        dataset = COCOValDataset(
            dataset_dir=FLAGS.dataset_dir,
            image_dir=FLAGS.val_image_dir,
            anno_path=FLAGS.val_anno_path)
        global anno_file
        anno_file = dataset.ann_file
        val_loader = paddle.io.DataLoader(
            dataset, batch_size=FLAGS.batch_size, drop_last=True)
        eval()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    paddle.set_device('cpu')

    main()
