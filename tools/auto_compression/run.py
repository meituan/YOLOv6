import os
import sys
import numpy as np
import argparse
from tqdm import tqdm
import paddle
from paddleslim.common import load_config
from paddleslim.auto_compression import AutoCompression
from dataset import COCOValDataset, COCOTrainDataset, yolo_image_preprocess
from post_process import YOLOPostProcess, coco_metric


def argsparser():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--config_path',
        type=str,
        default=None,
        help="path of compression strategy config.",
        required=True)
    parser.add_argument(
        '--save_dir',
        type=str,
        default='output',
        help="directory to save compressed model.")
    parser.add_argument(
        '--devices',
        type=str,
        default='gpu',
        help="which device used to compress.")

    return parser


def reader_wrapper(reader, input_name='x2paddle_images'):
    def gen():
        for data in reader:
            yield {input_name: data[0]}

    return gen


def eval_function(exe, compiled_test_program, test_feed_names, test_fetch_list):
    bboxes_list, bbox_nums_list, image_id_list = [], [], []
    with tqdm(
            total=len(val_loader),
            bar_format='Evaluation stage, Run batch:|{bar}| {n_fmt}/{total_fmt}',
            ncols=80) as t:
        for data in val_loader:
            data_all = {k: np.array(v) for k, v in data.items()}
            outs = exe.run(compiled_test_program,
                           feed={test_feed_names[0]: data_all['image']},
                           fetch_list=test_fetch_list,
                           return_numpy=False)
            res = {}
            postprocess = YOLOPostProcess(
                score_threshold=0.001, nms_threshold=0.65, multi_label=True)
            res = postprocess(np.array(outs[0]), data_all['scale_factor'])
            bboxes_list.append(res['bbox'])
            bbox_nums_list.append(res['bbox_num'])
            image_id_list.append(np.array(data_all['im_id']))
            t.update()
    map_res = coco_metric(anno_file, bboxes_list, bbox_nums_list, image_id_list)
    return map_res[0]


def main():
    global global_config
    all_config = load_config(FLAGS.config_path)
    assert "Global" in all_config, f"Key 'Global' not found in config file. \n{all_config}"
    global_config = all_config["Global"]
    input_name = 'x2paddle_image_arrays' if global_config[
        'arch'] == 'YOLOv6' else 'x2paddle_images'

    if global_config['image_path'] != 'None':
        assert os.path.exists(global_config['image_path'])
        paddle.vision.image.set_image_backend('cv2')
        train_dataset = paddle.vision.datasets.ImageFolder(
            global_config['image_path'], transform=yolo_image_preprocess)
        train_loader = paddle.io.DataLoader(
            train_dataset,
            batch_size=1,
            shuffle=True,
            drop_last=True,
            num_workers=0)
        train_loader = reader_wrapper(train_loader, input_name=input_name)
        eval_func = None
    else:
        dataset = COCOTrainDataset(
            dataset_dir=global_config['coco_dataset_dir'],
            image_dir=global_config['coco_train_image_dir'],
            anno_path=global_config['coco_train_anno_path'],
            input_name=input_name)
        train_loader = paddle.io.DataLoader(
            dataset, batch_size=1, shuffle=True, drop_last=True, num_workers=0)
        if paddle.distributed.get_rank() == 0:
            eval_func = eval_function
            global val_loader
            dataset = COCOValDataset(
                dataset_dir=global_config['coco_dataset_dir'],
                image_dir=global_config['coco_val_image_dir'],
                anno_path=global_config['coco_val_anno_path'])
            global anno_file
            anno_file = dataset.ann_file
            val_loader = paddle.io.DataLoader(
                dataset,
                batch_size=1,
                shuffle=False,
                drop_last=False,
                num_workers=0)
        else:
            eval_func = None

    ac = AutoCompression(
        model_dir=global_config["model_dir"],
        train_dataloader=train_loader,
        save_dir=FLAGS.save_dir,
        config=all_config,
        eval_callback=eval_func)
    ac.compress()
    ac.export_onnx()


if __name__ == '__main__':
    paddle.enable_static()
    parser = argsparser()
    FLAGS = parser.parse_args()

    assert FLAGS.devices in ['cpu', 'gpu', 'xpu', 'npu']
    paddle.set_device(FLAGS.devices)

    main()
