#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
from tqdm import tqdm
import numpy as np
import json
import torch
import yaml
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from yolov6.data.data_load import create_dataloader
from yolov6.utils.events import LOGGER, NCOLS
from yolov6.utils.nms import non_max_suppression
from yolov6.utils.checkpoint import load_checkpoint
from yolov6.utils.torch_utils import time_sync, get_model_info

'''
python tools/eval.py --task 'train'/'val'/'speed'
'''


class Evaler:
    def __init__(self,
                 data,
                 batch_size=32,
                 img_size=640,
                 conf_thres=0.03,
                 iou_thres=0.65,
                 device='',
                 half=True,
                 save_dir='',
                 test_load_size=640,
                 letterbox_return_int=False,
                 force_no_pad=False,
                 not_infer_on_rect=False,
                 scale_exact=False):
        self.data = data
        self.batch_size = batch_size
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.half = half
        self.save_dir = save_dir
        self.test_load_size = test_load_size
        self.letterbox_return_int = letterbox_return_int
        self.force_no_pad = force_no_pad
        self.not_infer_on_rect = not_infer_on_rect
        self.scale_exact = scale_exact

    def init_model(self, model, weights, task):
        if task != 'train':
            model = load_checkpoint(weights, map_location=self.device)
            self.stride = int(model.stride.max())
            if self.device.type != 'cpu':
                model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(model.parameters())))
            # switch to deploy
            from yolov6.layers.common import RepVGGBlock
            for layer in model.modules():
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
            LOGGER.info("Switch model to deploy modality.")
            LOGGER.info("Model Summary: {}".format(get_model_info(model, self.img_size)))
        model.half() if self.half else model.float()
        return model

    def init_data(self, dataloader, task):
        '''Initialize dataloader.
        Returns a dataloader for task val or speed.
        '''
        self.is_coco = self.data.get("is_coco", False)
        self.ids = self.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        if task != 'train':
            pad = 0.0 if task == 'speed' else 0.5
            eval_hyp = {
                "test_load_size":self.test_load_size,
                "letterbox_return_int":self.letterbox_return_int,
            }
            if self.force_no_pad:
                pad = 0.0
            rect = not self.not_infer_on_rect
            dataloader = create_dataloader(self.data[task if task in ('train', 'val', 'test') else 'val'],
                                           self.img_size, self.batch_size, self.stride, hyp=eval_hyp, check_labels=True, pad=pad, rect=rect,
                                           data_dict=self.data, task=task)[0]
        return dataloader

    def predict_model(self, model, dataloader, task):
        '''Model prediction
        Predicts the whole dataset and gets the prediced results and inference time.
        '''
        self.speed_result = torch.zeros(4, device=self.device)
        pred_results = []
        pbar = tqdm(dataloader, desc="Inferencing model in val datasets.", ncols=NCOLS)

        # whether to compute metric and plot PR curve and P、R、F1 curve under iou50 match rule
        metric_and_plot = True if task == "val" else False
        if metric_and_plot:
            stats, ap = [], []
            seen = 0
            iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
            niou = iouv.numel()

        for i, (imgs, targets, paths, shapes) in enumerate(pbar):

            # pre-process
            t1 = time_sync()
            imgs = imgs.to(self.device, non_blocking=True)
            imgs = imgs.half() if self.half else imgs.float()
            imgs /= 255
            self.speed_result[1] += time_sync() - t1  # pre-process time

            # Inference
            t2 = time_sync()
            outputs, _ = model(imgs)
            self.speed_result[2] += time_sync() - t2  # inference time

            # post-process
            t3 = time_sync()
            outputs = non_max_suppression(outputs, self.conf_thres, self.iou_thres, multi_label=True)
            self.speed_result[3] += time_sync() - t3  # post-process time
            self.speed_result[0] += len(outputs)

            if metric_and_plot:
                import copy
                eval_outputs = copy.deepcopy([x.detach().cpu() for x in outputs])

            # save result
            pred_results.extend(self.convert_to_coco_format(outputs, imgs, paths, shapes, self.ids))

            # for tensorboard visualization, maximum images to show: 8
            if i == 0:
                vis_num = min(len(imgs), 8)
                vis_outputs = outputs[:vis_num]
                vis_paths = paths[:vis_num]
            
            if not metric_and_plot:
                continue

            # Statistics per image 
            # code mainly based on https://github.com/WongKinYiu/yolov7/blob/main/test.py
            for si, pred in enumerate(eval_outputs):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1

                if len(pred) == 0:
                    if nl:
                        stats.append((torch.zeros(0, niou, dtype=torch.bool), torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Predictions
                predn = pred.clone()
                self.scale_coords(imgs[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                # Assign all predictions as incorrect
                correct = torch.zeros(pred.shape[0], niou, dtype=torch.bool)
                if nl:
                    detected = []  # target indices
                    tcls_tensor = labels[:, 0]

                    # target boxes
                    from yolov6.utils.nms import xywh2xyxy

                    tbox = xywh2xyxy(labels[:, 1:5])
                    tbox[:, [0, 2]] *= imgs[si].shape[1:][0]
                    tbox[:, [1, 3]] *= imgs[si].shape[1:][1]

                    self.scale_coords(imgs[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                    # Per target class
                    for cls in torch.unique(tcls_tensor):
                        ti = (cls == tcls_tensor).nonzero(as_tuple=False).view(-1)  # target indices
                        pi = (cls == pred[:, 5]).nonzero(as_tuple=False).view(-1)  # prediction indices

                        # Search for detections
                        if pi.shape[0] and ti.shape[0]:
                            # Prediction to target ious
                            from yolov6.utils.general import box_iou
                            ious, ious_i = box_iou(predn[pi, :4], tbox[ti]).max(1)  # best ious, indices

                            # Append detections
                            detected_set = set()
                            for j in (ious > iouv[0]).nonzero(as_tuple=False):
                                d = ti[ious_i[j]]  # detected target
                                if d.item() not in detected_set:
                                    detected_set.add(d.item())
                                    detected.append(d)
                                    correct[pi[j]] = ious[j] > iouv  # iou_thres is 1xn
                                    # if len(detected) == nl:  # all targets already located in image
                                    if len(detected_set) == ti.shape[0]:  # all targets already located in image
                                        break

                # Append statistics (correct, conf, pcls, tcls)
                stats.append((correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))
        
        if metric_and_plot:
            # Compute statistics
            stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
            if len(stats) and stats[0].any():

                from yolov6.utils.metrics import ap_per_class
                p, r, ap, f1, ap_class = ap_per_class(*stats, plot=metric_and_plot, save_dir=self.save_dir, names=model.names)
                AP50_F1_max_idx = f1.mean(0).argmax()
                LOGGER.info(f"IOU 50 best mF1 thershold near {AP50_F1_max_idx/1000.0}.")
                ap50, ap = ap[:, 0], ap.mean(1)  # AP@0.5, AP@0.5:0.95
                mp, mr, map50, map = p[:, AP50_F1_max_idx].mean(), r[:, AP50_F1_max_idx].mean(), ap50.mean(), ap.mean()
                nt = np.bincount(stats[3].astype(np.int64), minlength=model.nc)  # number of targets per class
            else:
                nt = torch.zeros(1)

            # Print results
            s = ('%10s' + '%12s' * 6) % ('Class', 'Images', 'Labels', 'P', 'R', 'mAP@.5', 'mAP@.5:.95')
            LOGGER.info(s)
            pf = '%10s' + '%12i' * 2 + '%12.3g' * 4  # print format
            LOGGER.info(pf % ('all', seen, nt.sum(), mp, mr, map50, map))
        return pred_results, vis_outputs, vis_paths
        

    def eval_model(self, pred_results, model, dataloader, task):
        '''Evaluate models
        For task speed, this function only evaluates the speed of model and outputs inference time.
        For task val, this function evaluates the speed and mAP by pycocotools, and returns
        inference time and mAP value.
        '''
        LOGGER.info(f'\nEvaluating speed.')
        self.eval_speed(task)

        LOGGER.info(f'\nEvaluating mAP by pycocotools.')
        if task != 'speed' and len(pred_results):
            if 'anno_path' in self.data:
                anno_json = self.data['anno_path']
            else:
                # generated coco format labels in dataset initialization
                dataset_root = os.path.dirname(os.path.dirname(self.data['val']))
                base_name = os.path.basename(self.data['val'])
                anno_json = os.path.join(dataset_root, 'annotations', f'instances_{base_name}.json')
            pred_json = os.path.join(self.save_dir, "predictions.json")
            LOGGER.info(f'Saving {pred_json}...')
            with open(pred_json, 'w') as f:
                json.dump(pred_results, f)

            anno = COCO(anno_json)
            pred = anno.loadRes(pred_json)
            cocoEval = COCOeval(anno, pred, 'bbox')
            if self.is_coco:
                imgIds = [int(os.path.basename(x).split(".")[0])
                            for x in dataloader.dataset.img_paths]
                cocoEval.params.imgIds = imgIds
            cocoEval.evaluate()
            cocoEval.accumulate()
            cocoEval.summarize()
            map, map50 = cocoEval.stats[:2]  # update results (mAP@0.5:0.95, mAP@0.5)
            # Return results
            model.float()  # for training
            if task != 'train':
                LOGGER.info(f"Results saved to {self.save_dir}")
            return (map50, map)
        return (0.0, 0.0)

    def eval_speed(self, task):
        '''Evaluate model inference speed.'''
        if task != 'train':
            n_samples = self.speed_result[0].item()
            pre_time, inf_time, nms_time = 1000 * self.speed_result[1:].cpu().numpy() / n_samples
            for n, v in zip(["pre-process", "inference", "NMS"],[pre_time, inf_time, nms_time]):
                LOGGER.info("Average {} time: {:.2f} ms".format(n, v))

    def box_convert(self, x):
        '''Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right.'''
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        '''Rescale coords (xyxy) from img1_shape to img0_shape.'''
        if ratio_pad is None:  # calculate from img0_shape
            gain = [min(img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1])]  # gain  = old / new
            if self.scale_exact:
                gain = [img1_shape[0] / img0_shape[0], img1_shape[1] / img0_shape[1]]
            pad = (img1_shape[1] - img0_shape[1] * gain) / 2, (img1_shape[0] - img0_shape[0] * gain) / 2  # wh padding
        else:
            gain = ratio_pad[0]
            pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        if self.scale_exact:
            coords[:, [0, 2]] /= gain[1]  # x gain
        else:
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

    def convert_to_coco_format(self, outputs, imgs, paths, shapes, ids):
        pred_results = []
        for i, pred in enumerate(outputs):
            if len(pred) == 0:
                continue
            path, shape = Path(paths[i]), shapes[i][0]
            self.scale_coords(imgs[i].shape[1:], pred[:, :4], shape, shapes[i][1])
            image_id = int(path.stem) if self.is_coco else path.stem
            bboxes = self.box_convert(pred[:, 0:4])
            bboxes[:, :2] -= bboxes[:, 2:] / 2
            cls = pred[:, 5]
            scores = pred[:, 4]
            for ind in range(pred.shape[0]):
                category_id = ids[int(cls[ind])]
                bbox = [round(x, 3) for x in bboxes[ind].tolist()]
                score = round(scores[ind].item(), 5)
                pred_data = {
                    "image_id": image_id,
                    "category_id": category_id,
                    "bbox": bbox,
                    "score": score
                }
                pred_results.append(pred_data)
        return pred_results

    @staticmethod
    def check_task(task):
        if task not in ['train', 'val', 'speed']:
            raise Exception("task argument error: only support 'train' / 'val' / 'speed' task.")

    @staticmethod
    def check_thres(conf_thres, iou_thres, task):
        '''Check whether confidence and iou threshold are best for task val/speed'''
        if task != 'train':
            if task == 'val':
                if conf_thres > 0.03:
                    LOGGER.warning(f'The best conf_thresh when evaluate the model is less than 0.03, while you set it to: {conf_thres}')
                if iou_thres != 0.65:
                    LOGGER.warning(f'The best iou_thresh when evaluate the model is 0.65, while you set it to: {iou_thres}')
            if task == 'speed' and conf_thres < 0.4:
                LOGGER.warning(f'The best conf_thresh when test the speed of the model is larger than 0.4, while you set it to: {conf_thres}')

    @staticmethod
    def reload_device(device, model, task):
        # device = 'cpu' or '0' or '0,1,2,3'
        if task == 'train':
            device = next(model.parameters()).device
        else:
            if device == 'cpu':
                os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
            elif device:
                os.environ['CUDA_VISIBLE_DEVICES'] = device
                assert torch.cuda.is_available()
            cuda = device != 'cpu' and torch.cuda.is_available()
            device = torch.device('cuda:0' if cuda else 'cpu')
        return device

    @staticmethod
    def reload_dataset(data):
        with open(data, errors='ignore') as yaml_file:
            data = yaml.safe_load(yaml_file)
        val = data.get('val')
        if not os.path.exists(val):
            raise Exception('Dataset not found.')
        return data

    @staticmethod
    def coco80_to_coco91_class():  # converts 80-index (val2014) to 91-index (paper)
    # https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
        x = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20,
            21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40,
            41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58,
            59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79,
            80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
        return x

    def eval_trt(self, engine, stride=32):
        self.stride = stride
        def init_engine(engine):
            import tensorrt as trt
            from collections import namedtuple,OrderedDict
            Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
            logger = trt.Logger(trt.Logger.ERROR)
            trt.init_libnvinfer_plugins(logger, namespace="")
            with open(engine, 'rb') as f, trt.Runtime(logger) as runtime:
                model = runtime.deserialize_cuda_engine(f.read())
            bindings = OrderedDict()
            for index in range(model.num_bindings):
                name = model.get_binding_name(index)
                dtype = trt.nptype(model.get_binding_dtype(index))
                shape = tuple(model.get_binding_shape(index))
                data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(self.device)
                bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))
            binding_addrs = OrderedDict((n, d.ptr) for n, d in bindings.items())
            context = model.create_execution_context()
            return context, bindings, binding_addrs, model.get_binding_shape(0)[0]

        def init_data(dataloader, task):
            self.is_coco = self.data.get("is_coco", False)
            self.ids = self.coco80_to_coco91_class() if self.is_coco else list(range(1000))
            pad = 0.0 if task == 'speed' else 0.5
            dataloader = create_dataloader(self.data[task if task in ('train', 'val', 'test') else 'val'],
                                           self.img_size, self.batch_size, self.stride, check_labels=True, pad=pad, rect=False,
                                           data_dict=self.data, task=task)[0]
            return dataloader

        def convert_to_coco_format_trt(nums, boxes, scores, classes, paths, shapes, ids):
            pred_results = []
            for i, (num, detbox, detscore, detcls) in enumerate(zip(nums, boxes, scores, classes)):
                n = int(num[0])
                if n == 0:
                    continue
                path, shape = Path(paths[i]), shapes[i][0]
                gain = shapes[i][1][0][0]
                pad = torch.tensor(shapes[i][1][1]*2).to(self.device)
                detbox = detbox[:n, :]
                detbox -= pad
                detbox /= gain
                detbox[:, 0].clamp_(0, shape[1])
                detbox[:, 1].clamp_(0, shape[0])
                detbox[:, 2].clamp_(0, shape[1])
                detbox[:, 3].clamp_(0, shape[0])
                detbox[:,2:] = detbox[:,2:] - detbox[:,:2]
                detscore = detscore[:n]
                detcls = detcls[:n]

                image_id = int(path.stem) if path.stem.isnumeric() else path.stem

                for ind in range(n):
                    category_id = ids[int(detcls[ind])]
                    bbox = [round(x, 3) for x in detbox[ind].tolist()]
                    score = round(detscore[ind].item(), 5)
                    pred_data = {
                        "image_id": image_id,
                        "category_id": category_id,
                        "bbox": bbox,
                        "score": score
                    }
                    pred_results.append(pred_data)
            return pred_results

        context, bindings, binding_addrs, trt_batch_size = init_engine(engine)
        assert trt_batch_size >= self.batch_size, f'The batch size you set is {self.batch_size}, it must <= tensorrt binding batch size {trt_batch_size}.'
        tmp = torch.randn(self.batch_size, 3, self.img_size, self.img_size).to(self.device)
        # warm up for 10 times
        for _ in range(10):
            binding_addrs['images'] = int(tmp.data_ptr())
            context.execute_v2(list(binding_addrs.values()))
        dataloader = init_data(None,'val')
        self.speed_result = torch.zeros(4, device=self.device)
        pred_results = []
        pbar = tqdm(dataloader, desc="Inferencing model in validation dataset.", ncols=NCOLS)
        for imgs, targets, paths, shapes in pbar:
            nb_img = imgs.shape[0]
            if nb_img != self.batch_size:
                # pad to tensorrt model setted batch size
                zeros = torch.zeros(self.batch_size - nb_img, 3, *imgs.shape[2:])
                imgs = torch.cat([imgs, zeros],0)
            t1 = time_sync()
            imgs = imgs.to(self.device, non_blocking=True)
            # preprocess
            imgs = imgs.float()
            imgs /= 255

            self.speed_result[1] += time_sync() - t1  # pre-process time

            # inference
            t2 = time_sync()
            binding_addrs['images'] = int(imgs.data_ptr())
            context.execute_v2(list(binding_addrs.values()))
            # in the last batch, the nb_img may less than the batch size, so we need to fetch the valid detect results by [:nb_img]
            nums = bindings['num_dets'].data[:nb_img]
            boxes = bindings['det_boxes'].data[:nb_img]
            scores = bindings['det_scores'].data[:nb_img]
            classes = bindings['det_classes'].data[:nb_img]
            self.speed_result[2] += time_sync() - t2  # inference time

            self.speed_result[3] += 0
            pred_results.extend(convert_to_coco_format_trt(nums, boxes, scores, classes, paths, shapes, self.ids))
            self.speed_result[0] += self.batch_size
        return dataloader, pred_results
