#!/usr/bin/env python3
# -*- coding:utf-8 -*-
import os
import cv2
import time
import math
import torch
import numpy as np
import os.path as osp

from tqdm import tqdm
from pathlib import Path
from PIL import ImageFont
from collections import deque

import torch.nn.functional as F

from yolov6.utils.events import LOGGER, load_yaml
from yolov6.layers.common import DetectBackend
from yolov6.data.data_augment import letterbox
from yolov6.data.datasets import LoadData
from yolov6.utils.nms import non_max_suppression_seg, non_max_suppression_seg_solo
from yolov6.utils.torch_utils import get_model_info

class Inferer:
    def __init__(self, source, webcam, webcam_addr, weights, device, yaml, img_size, half):

        self.__dict__.update(locals())

        # Init model
        self.device = device
        self.img_size = img_size
        cuda = self.device != 'cpu' and torch.cuda.is_available()
        self.device = torch.device(f'cuda:{device}' if cuda else 'cpu')
        self.model = DetectBackend(weights, device=self.device)
        self.stride = self.model.stride
        self.class_names = load_yaml(yaml)['names']
        self.img_size = self.check_img_size(self.img_size, s=self.stride)  # check image size
        self.half = half

        # Switch model to deploy status
        self.model_switch(self.model.model, self.img_size)

        # Half precision
        if self.half & (self.device.type != 'cpu'):
            self.model.model.half()
        else:
            self.model.model.float()
            self.half = False

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup

        # Load data
        self.webcam = webcam
        self.webcam_addr = webcam_addr
        self.files = LoadData(source, webcam, webcam_addr)
        self.source = source


    def model_switch(self, model, img_size):
        ''' Model switch to deploy status '''
        from yolov6.layers.common import RepVGGBlock
        for layer in model.modules():
            if isinstance(layer, RepVGGBlock):
                layer.switch_to_deploy()
            elif isinstance(layer, torch.nn.Upsample) and not hasattr(layer, 'recompute_scale_factor'):
                layer.recompute_scale_factor = None  # torch 1.11.0 compatibility

        LOGGER.info("Switch model to deploy modality.")

    def infer(self, conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, save_img, hide_labels, hide_conf, view_img=True, issolo=True, weight_nums=66, bias_nums=1, dyconv_channels=66):
        ''' Model Inference and results visualization '''
        vid_path, vid_writer, windows = None, None, []
        print(issolo)
        fps_calculator = CalcFPS()
        weight_nums = [weight_nums]
        bias_nums = [bias_nums]
        for img_src, img_path, vid_cap in tqdm(self.files):
            img, img_src = self.process_image(img_src, self.img_size, self.stride, self.half)
            img = img.to(self.device)
            if len(img.shape) == 3:
                img = img[None]
                # expand for batch dim
            t1 = time.time()
            pred_results = self.model(img)
            if not issolo:
                loutputs = non_max_suppression_seg(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            else:
                loutputs = non_max_suppression_seg_solo(pred_results, conf_thres, iou_thres, classes, agnostic_nms, max_det=max_det)
            protos = pred_results[1][0]
            segments = []
            print(len(loutputs))
            segconf = [loutputs[li][..., 0:] for li in range(len(loutputs))]
            det = [loutputs[li][..., :6] for li in range(len(loutputs))][0]
            if not issolo:
                segments = [self.handle_proto_test([protos[li].reshape(1, *(protos[li].shape[-3:]))], segconf[li], img.shape[-2:]) for li in range(len(loutputs))][0]
            else:
                segments = [self.handle_proto_solo([protos[li].reshape(1, *(protos[li].shape[-3:]))], segconf[li], img.shape[-2:], weight_sums=weight_nums, bias_sums=bias_nums, dyconv=dyconv_channels) for li in range(len(loutputs))][0]
            t2 = time.time()

            

            if self.webcam:
                save_path = osp.join(save_dir, self.webcam_addr)
                txt_path = osp.join(save_dir, self.webcam_addr)
            else:
                # Create output files in nested dirs that mirrors the structure of the images' dirs
                print(osp.dirname(img_path))
                print(osp.dirname(self.source))
                rel_path = "test"
                save_path = osp.join(save_dir, rel_path, osp.basename(img_path))  # im.jpg
                txt_path = osp.join(save_dir, rel_path, 'labels', osp.splitext(osp.basename(img_path))[0])
                os.makedirs(osp.join(save_dir, rel_path), exist_ok=True)

            gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
            img_ori = img_src.copy()

            # check image and font
            assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
            self.font_check()
            if len(det):
                det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
                
                
                ii = len(det) - 1
                segments = self.rescale_mask(img.shape[2:], segments.cpu().numpy(), img_src.shape)
                print(segments.shape)
                segments = segments.transpose(2, 0, 1)
                for *xyxy, conf, cls in reversed(det):
                    if save_txt:  # Write to file
                        xywh = (self.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
                        line = (cls, *xywh, conf)
                        with open(txt_path + '.txt', 'a') as f:
                            f.write(('%g ' * len(line)).rstrip() % line + '\n')

                    if save_img:
                        print(cls)
                        class_num = int(cls)  # integer class
                        label = None if hide_labels else (self.class_names[class_num] if hide_conf else f'{self.class_names[class_num]} {conf:.2f}')

                        img_ori = self.plot_box_and_label(img_ori, max(round(sum(img_ori.shape) / 2 * 0.003), 2), xyxy, label, color=self.generate_colors(class_num, True), segment=segments[ii])
                    ii -= 1

                img_src = np.asarray(img_ori)


            # FPS counter
            fps_calculator.update(1.0 / (t2 - t1))
            avg_fps = fps_calculator.accumulate()

            if self.files.type == 'video':
                self.draw_text(
                    img_src,
                    f"FPS: {avg_fps:0.1f}",
                    pos=(20, 20),
                    font_scale=1.0,
                    text_color=(204, 85, 17),
                    text_color_bg=(255, 255, 255),
                    font_thickness=2,
                )

            if view_img:
                if img_path not in windows:
                    windows.append(img_path)
                    cv2.namedWindow(str(img_path), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
                    cv2.resizeWindow(str(img_path), img_src.shape[1], img_src.shape[0])
                cv2.imshow(str(img_path), img_src)
                cv2.waitKey(1)  # 1 millisecond

            # Save results (image with detections)
            if save_img:
                if self.files.type == 'image':
                    cv2.imwrite(save_path, img_src)
                else:  # 'video' or 'stream'
                    if vid_path != save_path:  # new video
                        vid_path = save_path
                        if isinstance(vid_writer, cv2.VideoWriter):
                            vid_writer.release()  # release previous video writer
                        if vid_cap:  # video
                            fps = vid_cap.get(cv2.CAP_PROP_FPS)
                            w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                            h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                        else:  # stream
                            fps, w, h = 30, img_ori.shape[1], img_ori.shape[0]
                        save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
                        vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
                    vid_writer.write(img_src)

    @staticmethod
    def process_image(img_src, img_size, stride, half):
        '''Process image before image inference.'''
        image = letterbox(img_src, img_size, stride=stride)[0]
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image))
        image = image.half() if half else image.float()  # uint8 to fp16/32
        image /= 255  # 0 - 255 to 0.0 - 1.0

        return image, img_src

    @staticmethod
    def rescale(ori_shape, boxes, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = (ori_shape[1] - target_shape[1] * ratio) / 2, (ori_shape[0] - target_shape[0] * ratio) / 2

        boxes[:, [0, 2]] -= padding[0]
        boxes[:, [1, 3]] -= padding[1]
        boxes[:, :4] /= ratio

        boxes[:, 0].clamp_(0, target_shape[1])  # x1
        boxes[:, 1].clamp_(0, target_shape[0])  # y1
        boxes[:, 2].clamp_(0, target_shape[1])  # x2
        boxes[:, 3].clamp_(0, target_shape[0])  # y2

        return boxes

    @staticmethod
    def rescale_mask(ori_shape, masks, target_shape):
        '''Rescale the output to the original image shape'''
        ratio = min(ori_shape[0] / target_shape[0], ori_shape[1] / target_shape[1])
        padding = int((ori_shape[1] - target_shape[1] * ratio) / 2), int((ori_shape[0] - target_shape[0] * ratio) / 2)


        masks = masks[:, padding[1]: ori_shape[0]- padding[1], padding[0]: ori_shape[1] - padding[0]]
        masks = masks.transpose(1, 2, 0)
        masks = cv2.resize(masks, target_shape[:2][::-1])
        if len(masks.shape) == 2:
            masks = masks.reshape(*masks.shape, 1)

        return masks

    def check_img_size(self, img_size, s=32, floor=0):
        """Make sure image size is a multiple of stride s in each dimension, and return a new shape list of image."""
        if isinstance(img_size, int):  # integer i.e. img_size=640
            new_size = max(self.make_divisible(img_size, int(s)), floor)
        elif isinstance(img_size, list):  # list i.e. img_size=[640, 480]
            new_size = [max(self.make_divisible(x, int(s)), floor) for x in img_size]
        else:
            raise Exception(f"Unsupported type of img_size: {type(img_size)}")

        if new_size != img_size:
            print(f'WARNING: --img-size {img_size} must be multiple of max stride {s}, updating to {new_size}')
        return new_size if isinstance(img_size,list) else [new_size]*2

    def make_divisible(self, x, divisor):
        # Upward revision the value x to make it evenly divisible by the divisor.
        return math.ceil(x / divisor) * divisor

    @staticmethod
    def handle_proto(proto_list, oconfs, imgshape, det):
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
        
        xyxy = oconfs[..., :4]
        which_proto = conf[..., 0]
        confs = conf[..., 1:]
        res = []
        protos = proto_list[0]
        for i, proto in enumerate([protos, protos, protos]):
            s = proto.shape[-2:]
            tconfs = confs[which_proto[..., 0] == i]
            if tconfs.shape[0] == 0:
                continue
            tseg = ((tconfs@proto.reshape(proto.shape[0], proto.shape[1], -1)).reshape(proto.shape[0], tconfs.shape[1], *s))
            print("a:")
            print(which_proto[..., 0] == i)
            tseg=tseg.sigmoid()
            masks = F.interpolate(tseg, imgshape, mode='nearest')[0]
            #return masks
            print(xyxy[which_proto[..., 0] == i][0].shape)
            masks = crop_mask(masks, xyxy[which_proto[..., 0] == i][0])[0]
            res.append(masks.gt_(0.5))
        return torch.cat(res, dim = 0), xyxy[which_proto[..., 0] == i][0]
    

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
    
    # def handle_proto_solo(self, proto_list, oconfs, imgshape, weight_sums=66, bias_sums=66, dyconv=66, img_orishape=None):
    #     '''
    #     proto_list: [(bs, 32, w, h), ...]
    #     conf: (bs, l, 33) -> which_proto, 32
    #     '''
    #     def crop_mask(masks, boxes):
    #         """
    #         "Crop" predicted masks by zeroing out everything not in the predicted bbox.
    #         Vectorized by Chong (thanks Chong).

    #         Args:
    #             - masks should be a size [n, h, w] tensor of masks
    #             - boxes should be a size [n, 4] tensor of bbox coords in relative point form
    #         """

    #         n, h, w = masks.shape
    #         x1, y1, x2, y2 = torch.chunk(boxes[:, :, None], 4, 1)  # x1 shape(1,1,n)
    #         r = torch.arange(w, device=masks.device, dtype=x1.dtype)[None, None, :]  # rows shape(1,w,1)
    #         c = torch.arange(h, device=masks.device, dtype=x1.dtype)[None, :, None]  # cols shape(h,1,1)
    #         return masks * ((r >= x1) * (r < x2) * (c >= y1) * (c < y2))

    #     conf = oconfs[..., 6:]
    #     if conf.shape[0] == 0:
    #         return None
        
    #     xyxy = oconfs[..., :4]
    #     confs = conf[..., 1:]
    #     proto = proto_list[0]
    #     s = proto.shape[-2:]
    #     num_inst = confs.shape[0]
    #     proto = proto.reshape(1, -1, *proto.shape[-2:])
    #     proto = proto.repeat(num_inst, 1, 1, 1)
    #     weights, biases = self.parse_dynamic_params(confs, weight_nums=weight_sums, bias_nums=bias_sums, dyconv_channels=dyconv)
    #     n_layers = len(weights)
    #     for i, (weight, bias) in enumerate(zip(weights, biases)):
    #         x = F.conv2d(
    #             proto, weight, bias=bias, stride=1, padding=0, groups=num_inst)
    #         if i < n_layers - 1:
    #             x = F.relu(x)
    #     x = x.reshape(num_inst, *proto.shape[-2:])
    #     seg = x.sigmoid()
    #     masks = F.interpolate(seg, imgshape, mode='bilinear', align_corners=False)[0]
    #     if img_orishape:
    #         masks_ori = F.interpolate(seg, img_orishape, mode='nearest')[0]
    #     else:
    #         masks_ori = None
    #     masks = crop_mask(masks, xyxy).gt_(0.5)
    #     return masks
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
            
            



    @staticmethod
    def draw_text(
        img,
        text,
        font=cv2.FONT_HERSHEY_SIMPLEX,
        pos=(0, 0),
        font_scale=1,
        font_thickness=2,
        text_color=(0, 255, 0),
        text_color_bg=(0, 0, 0),
    ):

        offset = (5, 5)
        x, y = pos
        text_size, _ = cv2.getTextSize(text, font, font_scale, font_thickness)
        text_w, text_h = text_size
        rec_start = tuple(x - y for x, y in zip(pos, offset))
        rec_end = tuple(x + y for x, y in zip((x + text_w, y + text_h), offset))
        cv2.rectangle(img, rec_start, rec_end, text_color_bg, -1)
        cv2.putText(
            img,
            text,
            (x, int(y + text_h + font_scale - 1)),
            font,
            font_scale,
            text_color,
            font_thickness,
            cv2.LINE_AA,
        )

        return text_size

    @staticmethod
    def plot_box_and_label(image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX, segment=None):
        # Add one xyxy box to image with label
        p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
        common_color = [[128,0,0], [255,0,0],[255,0,255],[255,102,0],[51,51,0],[0,51,0],[51,204,204],[0,128,128],[0,204,255]]
        if segment is not None:
            import random
            # ii=random.randint(0, len(common_color)-1)
            colr = np.asarray(color)
            colr = colr.reshape(1,3).repeat((image.shape[0] * image.shape[1]), axis = 0).reshape(image.shape[0], image.shape[1], 3)
            image = cv2.addWeighted(image, 1, (colr * segment.reshape(*segment.shape[:2], 1)).astype(image.dtype), 0.8, 1)
        cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
        if label:
            tf = max(lw - 1, 1)  # font thickness
            w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
            outside = p1[1] - h - 3 >= 0  # label fits outside box
            p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
            cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
            cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,
                        thickness=tf, lineType=cv2.LINE_AA)
        return image

    @staticmethod
    def font_check(font='./yolov6/utils/Arial.ttf', size=10):
        # Return a PIL TrueType Font, downloading to CONFIG_DIR if necessary
        assert osp.exists(font), f'font path not exists: {font}'
        try:
            return ImageFont.truetype(str(font) if font.exists() else font.name, size)
        except Exception as e:  # download if missing
            return ImageFont.truetype(str(font), size)

    @staticmethod
    def box_convert(x):
        # Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
        y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
        y[:, 2] = x[:, 2] - x[:, 0]  # width
        y[:, 3] = x[:, 3] - x[:, 1]  # height
        return y

    @staticmethod
    def generate_colors(i, bgr=False):
        hex = ('FF3838', 'FF9D97', 'FF701F', 'FFB21D', 'CFD231', '48F90A', '92CC17', '3DDB86', '1A9334', '00D4BB',
               '2C99A8', '00C2FF', '344593', '6473FF', '0018EC', '8438FF', '520085', 'CB38FF', 'FF95C8', 'FF37C7')
        palette = []
        for iter in hex:
            h = '#' + iter
            palette.append(tuple(int(h[1 + i:1 + i + 2], 16) for i in (0, 2, 4)))
        num = len(palette)
        color = palette[int(i) % num]
        return (color[2], color[1], color[0]) if bgr else color
    
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

class CalcFPS:
    def __init__(self, nsamples: int = 50):
        self.framerate = deque(maxlen=nsamples)

    def update(self, duration: float):
        self.framerate.append(duration)

    def accumulate(self):
        if len(self.framerate) > 1:
            return np.average(self.framerate)
        else:
            return 0.0
