import cv2
import tensorrt as trt
import numpy as np
import time

import torch
import torchvision
from collections import OrderedDict, namedtuple


def torch_dtype_from_trt(dtype):
    if dtype == trt.bool:
        return torch.bool
    elif dtype == trt.int8:
        return torch.int8
    elif dtype == trt.int32:
        return torch.int32
    elif dtype == trt.float16:
        return torch.float16
    elif dtype == trt.float32:
        return torch.float32
    else:
        raise TypeError('%s is not supported by torch' % dtype)


def torch_device_from_trt(device):
    if device == trt.TensorLocation.DEVICE:
        return torch.device('cuda')
    elif device == trt.TensorLocation.HOST:
        return torch.device('cpu')
    else:
        return TypeError('%s is not supported by torch' % device)


def get_input_shape(engine):
    """Get input shape of the TensorRT YOLO engine."""
    binding = engine[0]
    assert engine.binding_is_input(binding)
    binding_dims = engine.get_binding_shape(binding)
    if len(binding_dims) == 4:
        return tuple(binding_dims[2:])
    elif len(binding_dims) == 3:
        return tuple(binding_dims[1:])
    else:
        raise ValueError('bad dims of binding %s: %s' % (binding, str(binding_dims)))

def letterbox(im, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleup=False, stride=32, return_int=False):
    # Resize and pad image while meeting stride-multiple constraints
    shape = im.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better val mAP)
        r = min(r, 1.0)

    # Compute padding
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding

    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, stride), np.mod(dh, stride)  # wh padding

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        im = cv2.resize(im, new_unpad, interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
    if not return_int:
        return im, r, (dw, dh)
    else:
        return im, r, (left, top)


class Processor():
    def __init__(self, model, num_classes=80, num_layers=3, anchors=1, device=torch.device('cuda:0'), return_int=False, scale_exact=False, force_no_pad=False, is_end2end=False):
        # load tensorrt engine)
        self.return_int = return_int
        self.scale_exact = scale_exact
        self.force_no_pad = force_no_pad
        self.is_end2end = is_end2end
        Binding = namedtuple('Binding', ('name', 'dtype', 'shape', 'data', 'ptr'))
        self.logger = trt.Logger(trt.Logger.INFO)
        trt.init_libnvinfer_plugins(self.logger, namespace="")
        self.runtime = trt.Runtime(self.logger)
        with open(model, "rb") as f:
            self.engine = self.runtime.deserialize_cuda_engine(f.read())
        self.input_shape = get_input_shape(self.engine)
        self.bindings = OrderedDict()
        self.input_names = list()
        self.output_names = list()
        for index in range(self.engine.num_bindings):
            name = self.engine.get_binding_name(index)
            if self.engine.binding_is_input(index):
                self.input_names.append(name)
            else:
                self.output_names.append(name)
            dtype = trt.nptype(self.engine.get_binding_dtype(index))
            shape = tuple(self.engine.get_binding_shape(index))
            data = torch.from_numpy(np.empty(shape, dtype=np.dtype(dtype))).to(device)
            self.bindings[name] = Binding(name, dtype, shape, data, int(data.data_ptr()))

        self.binding_addrs = OrderedDict((n, d.ptr) for n, d in self.bindings.items())
        self.context = self.engine.create_execution_context()
        assert self.engine
        assert self.context

        self.nc = num_classes  # number of classes
        self.no = num_classes + 5  # number of outputs per anchor
        self.nl = num_layers  # number of detection layers
        if isinstance(anchors, (list, tuple)):
            self.na = len(anchors[0]) // 2
        else:
            self.na = anchors
        self.anchors = anchors
        self.grid = [torch.zeros(1, device=device)] * num_layers
        self.prior_prob = 1e-2
        self.inplace = True
        stride = [8, 16, 32]  # strides computed during build
        self.stride = torch.tensor(stride, device=device)
        self.shape = [80, 40, 20]
        self.device = device

    def detect(self, img):
        """Detect objects in the input image."""
        resized, _ = self.pre_process(img, self.input_shape)
        outputs = self.inference(resized)
        return outputs

    def pre_process(self, img_src, input_shape=None,):
        """Preprocess an image before TRT YOLO inferencing.
        """
        input_shape = input_shape if input_shape is not None else self.input_shape
        image, ratio, pad = letterbox(img_src, input_shape, auto=False, return_int=self.return_int, scaleup=True)
        # Convert
        image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        image = torch.from_numpy(np.ascontiguousarray(image)).to(self.device).float()
        image = image / 255.  # 0 - 255 to 0.0 - 1.0
        return image, pad

    def inference(self, inputs):
        self.binding_addrs[self.input_names[0]] = int(inputs.data_ptr())
        #self.binding_addrs['x2paddle_image_arrays'] = int(inputs.data_ptr())
        self.context.execute_v2(list(self.binding_addrs.values()))
        if self.is_end2end:
            nums = self.bindings['num_dets'].data
            boxes = self.bindings['det_boxes'].data
            scores = self.bindings['det_scores'].data
            classes = self.bindings['det_classes'].data
            output = torch.cat((boxes, scores[:,:,None], classes[:,:,None]), axis=-1)
        else:
            output = self.bindings[self.output_names[0]].data
        #output = self.bindings['save_infer_model/scale_0.tmp_0'].data
        return output

    def output_reformate(self, outputs):
        z = []
        for i in range(self.nl):
            cls_output = outputs[3*i].reshape((1, -1, self.shape[i], self.shape[i]))
            reg_output = outputs[3*i+1].reshape((1, -1, self.shape[i], self.shape[i]))
            obj_output = outputs[3*i+2].reshape((1, -1, self.shape[i], self.shape[i]))

            y = torch.cat([reg_output, obj_output.sigmoid(), cls_output.sigmoid()], 1)
            bs, _, ny, nx = y.shape
            y = y.view(bs, -1, self.no, ny, nx).permute(0, 1, 3, 4, 2).contiguous()

            if self.grid[i].shape[2:4] != y.shape[2:4]:
                d = self.stride.device
                yv, xv = torch.meshgrid([torch.arange(ny).to(d), torch.arange(nx).to(d)], indexing='ij')
                self.grid[i] = torch.stack((xv, yv), 2).view(1, self.na, ny, nx, 2).float()
            if self.inplace:
                y[..., 0:2] = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
                y[..., 2:4] = torch.exp(y[..., 2:4]) * self.stride[i]  # wh
            else:
                xy = (y[..., 0:2] + self.grid[i]) * self.stride[i]  # xy
                wh = torch.exp(y[..., 2:4]) * self.stride[i]  # wh
                y = torch.cat((xy, wh, y[..., 4:]), -1)
            z.append(y.view(bs, -1, self.no))
        return torch.cat(z, 1)

    def post_process(self, outputs, img_shape, conf_thres=0.5, iou_thres=0.6):
        if self.is_end2end:
            det_t = outputs
        else:
            det_t = self.non_max_suppression(outputs, conf_thres, iou_thres, multi_label=True)
        self.scale_coords(self.input_shape, det_t[0][:, :4], img_shape[0], img_shape[1])
        return det_t[0]

    @staticmethod
    def xywh2xyxy(x):
        # Convert boxes with shape [n, 4] from [x, y, w, h] to [x1, y1, x2, y2] where x1y1 is top-left, x2y2=bottom-right
        y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def non_max_suppression(self, prediction, conf_thres=0.25, iou_thres=0.45, classes=None, agnostic=False, multi_label=False, max_det=300):
        """Runs Non-Maximum Suppression (NMS) on inference results.
        This code is borrowed from: https://github.com/ultralytics/yolov5/blob/47233e1698b89fc437a4fb9463c815e9171be955/utils/general.py#L775
        Args:
            prediction: (tensor), with shape [N, 5 + num_classes], N is the number of bboxes.
            conf_thres: (float) confidence threshold.
            iou_thres: (float) iou threshold.
            classes: (None or list[int]), if a list is provided, nms only keep the classes you provide.
            agnostic: (bool), when it is set to True, we do class-independent nms, otherwise, different class would do nms respectively.
            multi_label: (bool), when it is set to True, one box can have multi labels, otherwise, one box only huave one label.
            max_det:(int), max number of output bboxes.

        Returns:
             list of detections, echo item is one tensor with shape (num_boxes, 6), 6 is for [xyxy, conf, cls].
        """
        num_classes = prediction.shape[2] - 5  # number of classes
        pred_candidates = prediction[..., 4] > conf_thres  # candidates

        # Check the parameters.
        assert 0 <= conf_thres <= 1, f'conf_thresh must be in 0.0 to 1.0, however {conf_thres} is provided.'
        assert 0 <= iou_thres <= 1, f'iou_thres must be in 0.0 to 1.0, however {iou_thres} is provided.'

        # Function settings.
        max_wh = 4096  # maximum box width and height
        max_nms = 30000  # maximum number of boxes put into torchvision.ops.nms()
        time_limit = 10.0  # quit the function when nms cost time exceed the limit time.
        multi_label &= num_classes > 1  # multiple labels per box

        tik = time.time()
        output = [torch.zeros((0, 6), device=prediction.device)] * prediction.shape[0]
        for img_idx, x in enumerate(prediction):  # image index, image inference
            x = x[pred_candidates[img_idx]]  # confidence

            # If no box remains, skip the next process.
            if not x.shape[0]:
                continue

            # confidence multiply the objectness
            x[:, 5:] *= x[:, 4:5]  # conf = obj_conf * cls_conf

            # (center x, center y, width, height) to (x1, y1, x2, y2)
            box = self.xywh2xyxy(x[:, :4])

            # Detections matrix's shape is  (n,6), each row represents (xyxy, conf, cls)
            if multi_label:
                box_idx, class_idx = (x[:, 5:] > conf_thres).nonzero(as_tuple=False).T
                x = torch.cat((box[box_idx], x[box_idx, class_idx + 5, None], class_idx[:, None].float()), 1)
            else:  # Only keep the class with highest scores.
                conf, class_idx = x[:, 5:].max(1, keepdim=True)
                x = torch.cat((box, conf, class_idx.float()), 1)[conf.view(-1) > conf_thres]

            # Filter by class, only keep boxes whose category is in classes.
            if classes is not None:
                x = x[(x[:, 5:6] == torch.tensor(classes, device=x.device)).any(1)]

            # Check shape
            num_box = x.shape[0]  # number of boxes
            if not num_box:  # no boxes kept.
                continue
            elif num_box > max_nms:  # excess max boxes' number.
                x = x[x[:, 4].argsort(descending=True)[:max_nms]]  # sort by confidence

            # Batched NMS
            class_offset = x[:, 5:6] * (0 if agnostic else max_wh)  # classes
            boxes, scores = x[:, :4] + class_offset, x[:, 4]  # boxes (offset by class), scores
            keep_box_idx = torchvision.ops.nms(boxes, scores, iou_thres)  # NMS
            if keep_box_idx.shape[0] > max_det:  # limit detections
                keep_box_idx = keep_box_idx[:max_det]

            output[img_idx] = x[keep_box_idx]
            if (time.time() - tik) > time_limit:
                print(f'WARNING: NMS cost time exceed the limited {time_limit}s.')
                break  # time limit exceeded

        return output

    def scale_coords(self, img1_shape, coords, img0_shape, ratio_pad=None):
        # Rescale coords (xyxy) from img1_shape to img0_shape
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
