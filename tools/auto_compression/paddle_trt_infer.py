import os
import cv2
import numpy as np
import argparse
import time

from paddle.inference import Config
from paddle.inference import create_predictor

from post_process import YOLOPostProcess

CLASS_LABEL = [
    'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus', 'train',
    'truck', 'boat', 'traffic light', 'fire hydrant', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'backpack', 'umbrella', 'handbag',
    'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball', 'kite',
    'baseball bat', 'baseball glove', 'skateboard', 'surfboard',
    'tennis racket', 'bottle', 'wine glass', 'cup', 'fork', 'knife', 'spoon',
    'bowl', 'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot',
    'hot dog', 'pizza', 'donut', 'cake', 'chair', 'couch', 'potted plant',
    'bed', 'dining table', 'toilet', 'tv', 'laptop', 'mouse', 'remote',
    'keyboard', 'cell phone', 'microwave', 'oven', 'toaster', 'sink',
    'refrigerator', 'book', 'clock', 'vase', 'scissors', 'teddy bear',
    'hair drier', 'toothbrush'
]


def generate_scale(im, target_shape, keep_ratio=True):
    """
        Args:
            im (np.ndarray): image (np.ndarray)
        Returns:
            im_scale_x: the resize ratio of X
            im_scale_y: the resize ratio of Y
        """
    origin_shape = im.shape[:2]
    if keep_ratio:
        im_size_min = np.min(origin_shape)
        im_size_max = np.max(origin_shape)
        target_size_min = np.min(target_shape)
        target_size_max = np.max(target_shape)
        im_scale = float(target_size_min) / float(im_size_min)
        if np.round(im_scale * im_size_max) > target_size_max:
            im_scale = float(target_size_max) / float(im_size_max)
        im_scale_x = im_scale
        im_scale_y = im_scale
    else:
        resize_h, resize_w = target_shape
        im_scale_y = resize_h / float(origin_shape[0])
        im_scale_x = resize_w / float(origin_shape[1])
    return im_scale_y, im_scale_x


def image_preprocess(img_path, target_shape):
    img = cv2.imread(img_path)
    # Resize
    im_scale_y, im_scale_x = generate_scale(img, target_shape)
    img = cv2.resize(
        img,
        None,
        None,
        fx=im_scale_x,
        fy=im_scale_y,
        interpolation=cv2.INTER_LINEAR)
    # Pad
    im_h, im_w = img.shape[:2]
    h, w = target_shape[:]
    if h != im_h or w != im_w:
        canvas = np.ones((h, w, 3), dtype=np.float32)
        canvas *= np.array([114.0, 114.0, 114.0], dtype=np.float32)
        canvas[0:im_h, 0:im_w, :] = img.astype(np.float32)
        img = canvas
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = np.transpose(img, [2, 0, 1]) / 255
    img = np.expand_dims(img, 0)
    scale_factor = np.array([[im_scale_y, im_scale_x]])
    return img.astype(np.float32), scale_factor


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


def draw_box(image_file, results, class_label, threshold=0.5):
    srcimg = cv2.imread(image_file, 1)
    for i in range(len(results)):
        color_list = get_color_map_list(len(class_label))
        clsid2color = {}
        classid, conf = int(results[i, 0]), results[i, 1]
        if conf < threshold:
            continue
        xmin, ymin, xmax, ymax = int(results[i, 2]), int(results[i, 3]), int(
            results[i, 4]), int(results[i, 5])

        if classid not in clsid2color:
            clsid2color[classid] = color_list[classid]
        color = tuple(clsid2color[classid])

        cv2.rectangle(srcimg, (xmin, ymin), (xmax, ymax), color, thickness=2)
        print(class_label[classid] + ': ' + str(round(conf, 3)))
        cv2.putText(
            srcimg,
            class_label[classid] + ':' + str(round(conf, 3)), (xmin, ymin - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.8, (0, 255, 0),
            thickness=2)
    return srcimg


def load_predictor(model_dir,
                   run_mode='paddle',
                   batch_size=1,
                   device='CPU',
                   min_subgraph_size=3,
                   use_dynamic_shape=False,
                   trt_min_shape=1,
                   trt_max_shape=1280,
                   trt_opt_shape=640,
                   trt_calib_mode=False,
                   cpu_threads=1,
                   enable_mkldnn=False,
                   enable_mkldnn_bfloat16=False,
                   delete_shuffle_pass=False):
    """set AnalysisConfig, generate AnalysisPredictor
    Args:
        model_dir (str): root path of __model__ and __params__
        device (str): Choose the device you want to run, it can be: CPU/GPU/XPU, default is CPU
        run_mode (str): mode of running(paddle/trt_fp32/trt_fp16/trt_int8)
        use_dynamic_shape (bool): use dynamic shape or not
        trt_min_shape (int): min shape for dynamic shape in trt
        trt_max_shape (int): max shape for dynamic shape in trt
        trt_opt_shape (int): opt shape for dynamic shape in trt
        trt_calib_mode (bool): If the model is produced by TRT offline quantitative
            calibration, trt_calib_mode need to set True
        delete_shuffle_pass (bool): whether to remove shuffle_channel_detect_pass in TensorRT.
                                    Used by action model.
    Returns:
        predictor (PaddlePredictor): AnalysisPredictor
    Raises:
        ValueError: predict by TensorRT need device == 'GPU'.
    """
    rerun_flag = False
    if device != 'GPU' and run_mode != 'paddle':
        raise ValueError(
            "Predict by TensorRT mode: {}, expect device=='GPU', but device == {}"
            .format(run_mode, device))
    config = Config(
        os.path.join(model_dir, 'model.pdmodel'),
        os.path.join(model_dir, 'model.pdiparams'))
    if device == 'GPU':
        # initial GPU memory(M), device ID
        config.enable_use_gpu(200, 0)
        # optimize graph and fuse op
        config.switch_ir_optim(True)
    elif device == 'XPU':
        config.enable_lite_engine()
        config.enable_xpu(10 * 1024 * 1024)
    else:
        config.disable_gpu()
        config.set_cpu_math_library_num_threads(cpu_threads)
        if enable_mkldnn:
            try:
                # cache 10 different shapes for mkldnn to avoid memory leak
                config.set_mkldnn_cache_capacity(10)
                config.enable_mkldnn()
                if enable_mkldnn_bfloat16:
                    config.enable_mkldnn_bfloat16()
            except Exception as e:
                print(
                    "The current environment does not support `mkldnn`, so disable mkldnn."
                )
                pass

    precision_map = {
        'trt_int8': Config.Precision.Int8,
        'trt_fp32': Config.Precision.Float32,
        'trt_fp16': Config.Precision.Half
    }
    if run_mode in precision_map.keys():
        config.enable_tensorrt_engine(
            workspace_size=(1 << 25) * batch_size,
            max_batch_size=batch_size,
            min_subgraph_size=min_subgraph_size,
            precision_mode=precision_map[run_mode],
            use_static=False,
            use_calib_mode=trt_calib_mode)

        if use_dynamic_shape:
            dynamic_shape_file = os.path.join(args.model_path,
                                              'dynamic_shape.txt')
            if os.path.exists(dynamic_shape_file):
                config.enable_tuned_tensorrt_dynamic_shape(dynamic_shape_file,
                                                           True)
                print('trt set dynamic shape done!')
            else:
                config.collect_shape_range_info(dynamic_shape_file)
                print('Start collect dynamic shape...')
                rerun_flag = True

    # disable print log when predict
    config.disable_glog_info()
    # enable shared memory
    config.enable_memory_optim()
    # disable feed, fetch OP, needed by zero_copy_run
    config.switch_use_feed_fetch_ops(False)
    if delete_shuffle_pass:
        config.delete_pass("shuffle_channel_detect_pass")
    predictor = create_predictor(config)
    return predictor, rerun_flag


def predict_image(predictor,
                  image_file,
                  image_shape=[640, 640],
                  warmup=1,
                  repeats=1,
                  threshold=0.5,
                  arch='YOLOv5'):
    img, scale_factor = image_preprocess(image_file, image_shape)
    inputs = {}
    if arch == 'YOLOv6':
        inputs['x2paddle_image_arrays'] = img
    else:
        inputs['x2paddle_images'] = img
    input_names = predictor.get_input_names()
    for i in range(len(input_names)):
        input_tensor = predictor.get_input_handle(input_names[i])
        input_tensor.copy_from_cpu(inputs[input_names[i]])

    for i in range(warmup):
        predictor.run()

    np_boxes = None
    predict_time = 0.
    time_min = float("inf")
    time_max = float('-inf')
    for i in range(repeats):
        start_time = time.time()
        predictor.run()
        output_names = predictor.get_output_names()
        boxes_tensor = predictor.get_output_handle(output_names[0])
        np_boxes = boxes_tensor.copy_to_cpu()
        end_time = time.time()
        timed = end_time - start_time
        time_min = min(time_min, timed)
        time_max = max(time_max, timed)
        predict_time += timed

    time_avg = predict_time / repeats
    print('Inference time(ms): min={}, max={}, avg={}'.format(
        round(time_min * 1000, 2),
        round(time_max * 1000, 1), round(time_avg * 1000, 1)))
    postprocess = YOLOPostProcess(
        score_threshold=0.001, nms_threshold=0.65, multi_label=True)
    res = postprocess(np_boxes, scale_factor)
    res_img = draw_box(
        image_file, res['bbox'], CLASS_LABEL, threshold=threshold)
    cv2.imwrite('result.jpg', res_img)


if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--image_file', type=str, default=None, help="image path")
    parser.add_argument(
        '--model_path', type=str, help="inference model filepath")
    parser.add_argument(
        '--benchmark',
        type=bool,
        default=False,
        help="Whether run benchmark or not.")
    parser.add_argument(
        '--use_dynamic_shape',
        type=bool,
        default=True,
        help="Whether use dynamic shape or not.")
    parser.add_argument(
        '--run_mode',
        type=str,
        default='paddle',
        help="mode of running(paddle/trt_fp32/trt_fp16/trt_int8)")
    parser.add_argument(
        '--device',
        type=str,
        default='GPU',
        help="Choose the device you want to run, it can be: CPU/GPU/XPU, default is GPU"
    )
    parser.add_argument(
        '--arch', type=str, default='YOLOv5', help="architectures name.")
    parser.add_argument('--img_shape', type=int, default=640, help="input_size")
    args = parser.parse_args()

    warmup, repeats = 1, 1
    if args.benchmark:
        warmup, repeats = 50, 100

    predictor, rerun_flag = load_predictor(
        args.model_path,
        run_mode=args.run_mode,
        device=args.device,
        use_dynamic_shape=args.use_dynamic_shape)
    predict_image(
        predictor,
        args.image_file,
        image_shape=[args.img_shape, args.img_shape],
        warmup=warmup,
        repeats=repeats,
        arch=args.arch)

    if rerun_flag:
        print(
            "***** Collect dynamic shape done, Please rerun the program to get correct results. *****"
        )
