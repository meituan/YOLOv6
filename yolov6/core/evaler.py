#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
from tqdm import tqdm
import numpy as np
import json
import torch
import yaml
from pathlib import Path
import cv2
from multiprocessing.pool import ThreadPool



from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

import torch.nn.functional as F

from yolov6.data.data_load import create_dataloader
from yolov6.utils.events import LOGGER, NCOLS
from yolov6.utils.nms import non_max_suppression_seg, non_max_suppression_seg_solo
from yolov6.utils.general import download_ckpt
from yolov6.utils.checkpoint import load_checkpoint
from yolov6.utils.torch_utils import time_sync, get_model_info


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
                 shrink_size=640,
                 infer_on_rect=False,
                 verbose=False,
                 do_coco_metric=True,
                 do_pr_metric=False,
                 plot_curve=True,
                 plot_confusion_matrix=False,
                 specific_shape=False,
                 height=640,
                 width=640
                 ):
        assert do_pr_metric or do_coco_metric, 'ERROR: at least set one val metric'
        self.data = data
        self.batch_size = batch_size
        self.img_size = img_size
        self.conf_thres = conf_thres
        self.iou_thres = iou_thres
        self.device = device
        self.half = half
        self.save_dir = save_dir
        self.shrink_size = shrink_size
        self.infer_on_rect = infer_on_rect
        self.verbose = verbose
        self.do_coco_metric = do_coco_metric
        self.do_pr_metric = do_pr_metric
        self.plot_curve = plot_curve
        self.plot_confusion_matrix = plot_confusion_matrix
        self.specific_shape = specific_shape
        self.height = height
        self.width = width

    def init_model(self, model, weights, task):
        if task != 'train':
            if not os.path.exists(weights):
                download_ckpt(weights)
            model = load_checkpoint(weights, map_location=self.device)
            self.stride = int(model.stride.max())
            # switch to deploy
            from yolov6.layers.common import RepVGGBlock
            for layer in model.modules():
                if isinstance(layer, RepVGGBlock):
                    layer.switch_to_deploy()
                elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
                    layer.recompute_scale_factor = None  # torch 1.11.0 compatibility
            LOGGER.info("Switch model to deploy modality.")
            LOGGER.info("Model Summary: {}".format(get_model_info(model, self.img_size)))
        if self.device.type != 'cpu':
            model(torch.zeros(1, 3, self.img_size, self.img_size).to(self.device).type_as(next(model.parameters())))
        model.half() if self.half else model.float()
        return model

    def init_data(self, dataloader, task):
        '''Initialize dataloader.
        Returns a dataloader for task val or speed.
        '''
        self.is_coco = self.data.get("is_coco", False)
        self.ids = self.coco80_to_coco91_class() if self.is_coco else list(range(1000))
        if task != 'train':
            pad = 0.0
            eval_hyp = {
                "shrink_size":self.shrink_size,
            }
            rect = self.infer_on_rect
            dataloader = create_dataloader(self.data[task if task in ('train', 'val', 'test') else 'val'],
                                           self.img_size, self.batch_size, self.stride, hyp=eval_hyp, check_labels=True, pad=0.5, rect=True,
                                           data_dict=self.data, task=task, specific_shape=self.specific_shape, height=self.height, width=self.width)[0]
        return dataloader

    def predict_model(self, model, dataloader, task, issolo=False, weight_nums=66, bias_nums=1, dyconv_channels=66):
        '''Model prediction
        Predicts the whole dataset and gets the prediced results and inference time.
        '''
        self.speed_result = torch.zeros(4, device=self.device)
        pred_results = []
        pbar = tqdm(dataloader, desc=f"Inferencing model in {task} datasets.", ncols=NCOLS)
        weight_nums = [weight_nums]
        bias_nums = [bias_nums]
        # whether to compute metric and plot PR curve and P、R、F1 curve under iou50 match rule
        if self.do_pr_metric:
            stats, ap = [], []
            seen = 0
            iouv = torch.linspace(0.5, 0.95, 10)  # iou vector for mAP@0.5:0.95
            niou = iouv.numel()
            if self.plot_confusion_matrix:
                from yolov6.utils.metrics import ConfusionMatrix
                confusion_matrix = ConfusionMatrix(nc=model.nc)

        for i, (imgs, targets, paths, shapes, masks) in enumerate(pbar):
            # pre-process
            t1 = time_sync()
            imgs = imgs.to(self.device, non_blocking=True)
            imgs = imgs.half() if self.half else imgs.float()
            imgs /= 255
            self.speed_result[1] += time_sync() - t1  # pre-process time

            # Inference
            t2 = time_sync()
            toutputs, _ = model(imgs)
            self.speed_result[2] += time_sync() - t2  # inference time

            # post-process
            t3 = time_sync()
            if not issolo:
                loutputs = non_max_suppression_seg(toutputs, self.conf_thres, self.iou_thres, multi_label=True)
            else:
                loutputs = non_max_suppression_seg_solo(toutputs, self.conf_thres, self.iou_thres, multi_label=True)
            protos = toutputs[1][0]
            segments = []
            segconf = [loutputs[li][..., 0:] for li in range(len(loutputs))]
            outputs = [loutputs[li][..., :6] for li in range(len(loutputs))]
            if not issolo:
                segments = [self.handle_proto_test([protos[li].reshape(1, *(protos[li].shape[-3:]))], segconf[li], imgs.shape[-2:]) for li in range(len(loutputs))]
            else:
                segments = [self.handle_proto_solo([protos[li].reshape(1, *(protos[li].shape[-3:]))], segconf[li], imgs.shape[-2:], weight_sums=weight_nums, bias_sums=bias_nums, dyconv=dyconv_channels) for li in range(len(loutputs))]
            self.speed_result[3] += time_sync() - t3  # post-process time
            self.speed_result[0] += len(outputs)

            if self.do_pr_metric:
                import copy
                eval_outputs = copy.deepcopy([x.detach().cpu() for x in outputs])

            # save result
            # pred_results.extend(self.convert_to_coco_format_seg(outputs, imgs, paths, shapes, self.ids, segments))

            # for tensorboard visualization, maximum images to show: 8
            if i == 0:
                vis_num = min(len(imgs), 8)
                vis_outputs = outputs[:vis_num]
                vis_paths = paths[:vis_num]

            if not self.do_pr_metric:
                continue

            # Statistics per image
            # This code is based on
            # https://github.com/ultralytics/yolov5/blob/master/val.py
            for si, (pred, pred_masks) in enumerate(zip(eval_outputs, segments)):
                labels = targets[targets[:, 0] == si, 1:]
                nl = len(labels)
                tcls = labels[:, 0].tolist() if nl else []  # target class
                seen += 1
                correct_masks = torch.zeros(len(pred), niou, dtype=torch.bool)  # init
                correct = torch.zeros(len(pred), niou, dtype=torch.bool)  # init

                if len(pred) == 0:
                    if nl:
                        stats.append((correct_masks, correct, torch.Tensor(), torch.Tensor(), tcls))
                    continue

                # Masks
                midx = targets[:, 0] == si
                gt_masks = masks[midx]
                # Predictions
                predn = pred.clone()
                self.scale_coords(imgs[si].shape[1:], predn[:, :4], shapes[si][0], shapes[si][1])  # native-space pred

                # Assign all predictions as incorrect
                
                if nl:
                    from yolov6.utils.nms import xywh2xyxy

                    # target boxes
                    tbox = xywh2xyxy(labels[:, 1:5])
                    tbox[:, [0, 2]] *= imgs[si].shape[1:][1]
                    tbox[:, [1, 3]] *= imgs[si].shape[1:][0]

                    self.scale_coords(imgs[si].shape[1:], tbox, shapes[si][0], shapes[si][1])  # native-space labels

                    labelsn = torch.cat((labels[:, 0:1], tbox), 1)  # native-space labels

                    from yolov6.utils.metrics import process_batch    

                    correct = process_batch(predn, labelsn, iouv)
                    correct_masks = process_batch(predn, labelsn, iouv, pred_masks, gt_masks, overlap=False, masks=True)
                    if self.plot_confusion_matrix:
                        confusion_matrix.process_batch(predn, labelsn)

                # Append statistics (correct, conf, pcls, tcls)


                stats.append((correct_masks.cpu(), correct.cpu(), pred[:, 4].cpu(), pred[:, 5].cpu(), tcls))

        if self.do_pr_metric:
            # Compute statistics
            stats = [np.concatenate(x, 0) for x in zip(*stats)]  # to numpy
            if len(stats) and stats[0].any():
                from yolov6.utils.metrics import ap_per_class_box_and_mask, Metrics
                metrics = Metrics()
                # v5 method
                results = ap_per_class_box_and_mask(*stats, plot=self.plot_curve, save_dir=self.save_dir, names=model.names)
                metrics.update(results)
                nt = np.bincount(stats[4].astype(np.int64), minlength=model.nc)  # number of targets per class

                # Print results
                s = ('%22s' + '%15s' * 10) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)', 'Mask(P', 'R',
                                  'mAP50', 'mAP50-95)')
                LOGGER.info(s)
                pf = '%22s' + '%15i' * 2 + '%11.5g' * 8  # print format
                mr = metrics.mean_results()
                LOGGER.info(pf % ('all', seen, nt.sum(), *mr))
                return [mr[2], mr[3], mr[6], mr[7]], [], []

                if self.plot_confusion_matrix:
                    confusion_matrix.plot(save_dir=self.save_dir, names=list(model.names))
            else:
                return [0, 0, 0, 0], [], []

        return pred_results

    def parse_dynamic_params(self, flatten_kernels, weight_nums, bias_nums, dyconv_channels):
        """split kernel head prediction to conv weight and bias."""
        n_inst = flatten_kernels.size(0)
        n_layers = len(weight_nums)
        params_splits = list(
            torch.split_with_sizes(
                flatten_kernels, weight_nums + bias_nums, dim=1))
        weight_splits = params_splits[:n_layers]
        bias_splits = params_splits[n_layers:]
        for i in range(n_layers):
            if i < n_layers - 1:
                weight_splits[i] = weight_splits[i].reshape(
                    n_inst * dyconv_channels, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(n_inst *
                                                        dyconv_channels)
            else:
                weight_splits[i] = weight_splits[i].reshape(n_inst, -1, 1, 1)
                bias_splits[i] = bias_splits[i].reshape(n_inst)

        return weight_splits, bias_splits

    def handle_proto_solo(self, proto_list, oconfs, imgshape, weight_sums=66, bias_sums=1, dyconv=66, img_orishape=None):
        '''
        proto_list: [(bs, 32, w, h), ...]
        conf: (bs, l, 33) -> which_proto, 32
        '''
        def handle_proto_coord(proto):
            _ = proto.shape[-2:]
            x = torch.arange(0, 1, step = 1 / _[1]).unsqueeze(0).unsqueeze(0).repeat(1, _[0], 1).to(proto.dtype).to(proto.device)
            y = torch.arange(0, 1, step = 1 / _[0]).unsqueeze(0).T.unsqueeze(0).repeat(1, 1, _[1]).to(proto.dtype).to(proto.device)
            return torch.cat([proto, x, y]).reshape(1, -1, *_)
        
        def crop_mask(masks, boxes):
            """
            "Crop" predicted masks by zeroing out everything not in the predicted bbox.
            Vectorized by Chong (thanks Chong).

            Args:
                - masks should be a size [n, h, w] tensor of masks
                - boxes should be a size [n, 4] tensor of bbox coords in relative point form
            """

            n, h, w = masks.shape
            x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
            r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
            c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
            return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

        conf = oconfs[..., 6:]
        if conf.shape[0] == 0:
            return None
        
        xyxy = oconfs[..., :4]
        confs = conf[..., 1:]
        proto = proto_list[0][0]
        proto = handle_proto_coord(proto)
        s = proto.shape[-2:]
        num_inst = confs.shape[0]
        proto = proto.reshape(1, -1, *proto.shape[-2:])
        weights, biases = self.parse_dynamic_params(confs, weight_nums=weight_sums, bias_nums=bias_sums, dyconv_channels=dyconv)
        n_layers = len(weights)
        for i, (weight, bias) in enumerate(zip(weights, biases)):
            x = F.conv2d(
                proto, weight, bias=bias, stride=1, padding=0, groups=1)
            if i < n_layers - 1:
                x = F.relu(x)
        x = x.reshape(num_inst, *proto.shape[-2:]).unsqueeze(0)
        seg = x.sigmoid()
        masks = F.interpolate(seg, imgshape, mode='bilinear', align_corners=False)[0]
        if img_orishape:
            masks_ori = F.interpolate(seg, img_orishape, mode='nearest')[0]
        else:
            masks_ori = None
        masks = crop_mask(masks, xyxy).gt_(0.5)
        masks = masks.gt_(0.5)
        return masks
            


    def eval_model(self, pred_results, model, dataloader, task):
        '''Evaluate models
        For task speed, this function only evaluates the speed of model and outputs inference time.
        For task val, this function evaluates the speed and mAP by pycocotools, and returns
        inference time and mAP value.
        '''
        LOGGER.info(f'\nEvaluating speed.')
        self.eval_speed(task)

        if not self.do_coco_metric and self.do_pr_metric:
            return self.pr_metric_result
        LOGGER.info(f'\nEvaluating mAP by pycocotools.')
        if task != 'speed' and len(pred_results):
            if 'anno_path' in self.data:
                anno_json = self.data['anno_path']
            else:
                # generated coco format labels in dataset initialization
                task = 'val' if task == 'train' else task
                if not isinstance(self.data[task], list):
                    self.data[task] = [self.data[task]]
                dataset_root = os.path.dirname(os.path.dirname(self.data[task][0]))
                base_name = os.path.basename(self.data[task][0])
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

            #print each class ap from pycocotool result
            if self.verbose:

                import copy
                val_dataset_img_count = cocoEval.cocoGt.imgToAnns.__len__()
                val_dataset_anns_count = 0
                label_count_dict = {"images":set(), "anns":0}
                label_count_dicts = [copy.deepcopy(label_count_dict) for _ in range(model.nc)]
                for _, ann_i in cocoEval.cocoGt.anns.items():
                    if ann_i["ignore"]:
                        continue
                    val_dataset_anns_count += 1
                    nc_i = self.coco80_to_coco91_class().index(ann_i['category_id']) if self.is_coco else ann_i['category_id']
                    label_count_dicts[nc_i]["images"].add(ann_i["image_id"])
                    label_count_dicts[nc_i]["anns"] += 1

                s = ('%22s' + '%11s' * 10) % ('Class', 'Images', 'Instances', 'Box(P', 'R', 'mAP50', 'mAP50-95)', 'Mask(P', 'R',
                                  'mAP50', 'mAP50-95)')
                LOGGER.info(s)
                #IOU , all p, all cats, all gt, maxdet 100
                coco_p = cocoEval.eval['precision']
                coco_p_all = coco_p[:, :, :, 0, 2]
                map = np.mean(coco_p_all[coco_p_all>-1])

                coco_p_iou50 = coco_p[0, :, :, 0, 2]
                map50 = np.mean(coco_p_iou50[coco_p_iou50>-1])
                mp = np.array([np.mean(coco_p_iou50[ii][coco_p_iou50[ii]>-1]) for ii in range(coco_p_iou50.shape[0])])
                mr = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
                mf1 = 2 * mp * mr / (mp + mr + 1e-16)
                i = mf1.argmax()  # max F1 index

                pf = '%-16s' + '%12i' * 2 + '%12.3g' * 5  # print format
                LOGGER.info(pf % ('all', val_dataset_img_count, val_dataset_anns_count, mp[i], mr[i], mf1[i], map50, map))

                #compute each class best f1 and corresponding p and r
                for nc_i in range(model.nc):
                    coco_p_c = coco_p[:, :, nc_i, 0, 2]
                    map = np.mean(coco_p_c[coco_p_c>-1])

                    coco_p_c_iou50 = coco_p[0, :, nc_i, 0, 2]
                    map50 = np.mean(coco_p_c_iou50[coco_p_c_iou50>-1])
                    p = coco_p_c_iou50
                    r = np.linspace(.0, 1.00, int(np.round((1.00 - .0) / .01)) + 1, endpoint=True)
                    f1 = 2 * p * r / (p + r + 1e-16)
                    i = f1.argmax()
                    LOGGER.info(pf % (model.names[nc_i], len(label_count_dicts[nc_i]["images"]), label_count_dicts[nc_i]["anns"], p[i], r[i], f1[i], map50, map))
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

        gain = ratio_pad[0]
        pad = ratio_pad[1]

        coords[:, [0, 2]] -= pad[0]  # x padding
        coords[:, [0, 2]] /= gain[1]  # raw x gain
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

    def convert_to_coco_format_seg(self, outputs, imgs, paths, shapes, ids, masks):
        
        from pycocotools.mask import encode
        import time

        def single_encode(x):
            rle = encode(np.asarray(x[:, :, None], order='F', dtype='uint8'))[0]
            rle['counts'] = rle['counts'].decode('utf-8')
            return rle
            
        
        pred_results = []
        for i, pred in enumerate(outputs):
            if len(pred) == 0:
                continue
            pred_masks = masks[i].cpu().numpy()
            pred_masks = np.transpose(pred_masks, (2, 0, 1))
            a = time.time()
            with ThreadPool(64) as pool:
                rles = pool.map(single_encode, pred_masks)
            print("rle time")
            b = time.time()
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
                    "score": score,
                    'segmentation': rles[i]
                }
                pred_results.append(pred_data)
            c = time.time()
            print(b-a, c-b)
        return pred_results

    @staticmethod
    def check_task(task):
        if task not in ['train', 'val', 'test', 'speed']:
            raise Exception("task argument error: only support 'train' / 'val' / 'test' / 'speed' task.")

    @staticmethod
    def check_thres(conf_thres, iou_thres, task):
        '''Check whether confidence and iou threshold are best for task val/speed'''
        if task != 'train':
            if task == 'val' or task == 'test':
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
    def reload_dataset(data, task='val'):
        with open(data, errors='ignore') as yaml_file:
            data = yaml.safe_load(yaml_file)
        task = 'test' if task == 'test' else 'val'
        path = data.get(task, 'val')
        if not isinstance(path, list):
            path = [path]
        for p in path:
            if not os.path.exists(p):
                raise Exception(f'Dataset path {p} not found.')
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
            pad = 0.0
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

    

    @staticmethod
    def handle_proto_test(proto_list, oconfs, imgshape, img_orishape=None):
        '''
        proto_list: [(bs, 32, w, h), ...]
        conf: (bs, l, 33) -> which_proto, 32
        '''
        
    
        def crop_mask(masks, boxes):
            """
            "Crop" predicted masks by zeroing out everything not in the predicted bbox.
            Vectorized by Chong (thanks Chong).

            Args:
                - masks should be a size [n, h, w] tensor of masks
                - boxes should be a size [n, 4] tensor of bbox coords in relative point form
            """

            n, h, w = masks.shape
            x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
            r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
            c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
            return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

        conf = oconfs[..., 6:]
        if conf.shape[0] == 0:
            return None
        
        xyxy = oconfs[..., :4]
        confs = conf[..., 1:]
        proto = proto_list[0]
        
        s = proto.shape[-2:]
        seg = ((confs@proto.reshape(proto.shape[0], proto.shape[1], -1)).reshape(proto.shape[0], confs.shape[0], *s))
        seg = seg.sigmoid()
        masks = F.interpolate(seg, imgshape, mode='bilinear', align_corners=False)[0]
        if img_orishape:
            masks_ori = F.interpolate(seg, img_orishape, mode='nearest')[0]
        else:
            masks_ori = None
        masks = crop_mask(masks, xyxy).gt_(0.5)
        return masks
