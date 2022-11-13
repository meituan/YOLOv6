from tools.infer import Inferer
from pathlib import Path
from yolov6.data.data_augment import letterbox
from yolov6.utils.nms import non_max_suppression
import torch
import numpy as np
from yolov6.layers.common import DetectBackend
from yolov6.utils.events import LOGGER, load_yaml
import time
import os.path as osp
import os
import cv2
import math

class Yolov6Detector:
	def __init__(self, weights, yaml, device='0',img_size=416):
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
		# Switch model to deploy status
		self.model_switch(self.model.model)
		# Half precision
		self.model.model.float()
		self.half = False
		if self.device.type != 'cpu':
			self.model(torch.zeros(1, 3, *self.img_size).to(self.device).type_as(next(self.model.model.parameters())))  # warmup

	def make_divisible(self, x, divisor):
		# Upward revision the value x to make it evenly divisible by the divisor.
		return math.ceil(x / divisor) * divisor

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

	def model_switch(self, model):
		''' Model switch to deploy status '''
		from yolov6.layers.common import RepVGGBlock
		for layer in model.modules():
			if isinstance(layer, RepVGGBlock):
				layer.switch_to_deploy()

		LOGGER.info("Switch model to deploy modality.")

	def pre_process_image(self,img_src, img_size, stride):
		'''Process image before image inference.'''
		image = letterbox(img_src, img_size, stride=stride)[0]
		# Convert
		image = image.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
		image = torch.from_numpy(np.ascontiguousarray(image))
		image =  image.float()  # uint8 to fp16/32
		image /= 255  # 0 - 255 to 0.0 - 1.0
		return image, img_src


	def plot_box_and_label(self,image, lw, box, label='', color=(128, 128, 128), txt_color=(255, 255, 255), font=cv2.FONT_HERSHEY_COMPLEX):
		# Add one xyxy box to image with label
		p1, p2 = (int(box[0]), int(box[1])), (int(box[2]), int(box[3]))
		cv2.rectangle(image, p1, p2, color, thickness=lw, lineType=cv2.LINE_AA)
		if label:
			tf = max(lw - 1, 1)  # font thickness
			w, h = cv2.getTextSize(label, 0, fontScale=lw / 3, thickness=tf)[0]  # text width, height
			outside = p1[1] - h - 3 >= 0  # label fits outside box
			p2 = p1[0] + w, p1[1] - h - 3 if outside else p1[1] + h + 3
			cv2.rectangle(image, p1, p2, color, -1, cv2.LINE_AA)  # filled
			cv2.putText(image, label, (p1[0], p1[1] - 2 if outside else p1[1] + h + 2), font, lw / 3, txt_color,thickness=tf, lineType=cv2.LINE_AA)



	def rescale(self, ori_shape, boxes, target_shape):
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

	def box_convert(sef, x):
		# Convert boxes with shape [n, 4] from [x1, y1, x2, y2] to [x, y, w, h] where x1y1=top-left, x2y2=bottom-right
		y = x.clone() if isinstance(x, torch.Tensor) else np.copy(x)
		y[:, 0] = (x[:, 0] + x[:, 2]) / 2  # x center
		y[:, 1] = (x[:, 1] + x[:, 3]) / 2  # y center
		y[:, 2] = x[:, 2] - x[:, 0]  # width
		y[:, 3] = x[:, 3] - x[:, 1]  # height
		return y


	def detect(self,img_src, view_img = False):
		img, img_src = self.pre_process_image(img_src, self.img_size, self.stride)
		img = img.to(self.device)
		if len(img.shape) == 3:
			img = img[None]
			# expand for batch dim
		t1 = time.time()
		pred_results = self.model(img)
		det = non_max_suppression(pred_results, conf_thres=0.4, iou_thres=0.45, classes=None, max_det=1)[0]
		t2 = time.time()

		gn = torch.tensor(img_src.shape)[[1, 0, 1, 0]]  # normalization gain whwh
		img_ori = img_src.copy()

		# check image and font
		assert img_ori.data.contiguous, 'Image needs to be contiguous. Please apply to input images with np.ascontiguousarray(im).'
		lines = []
		if len(det) and view_img:
			det[:, :4] = self.rescale(img.shape[2:], det[:, :4], img_src.shape).round()
			for *xyxy, conf, cls in reversed(det):
				xywh = (self.box_convert(torch.tensor(xyxy).view(1, 4)) / gn).view(-1).tolist()  # normalized xywh
				line = (cls, *xywh, conf)
				lines.append(line)

				# if save_img:
				class_num = int(cls)  # integer class
				label =  f'{self.class_names[class_num]} {conf:.2f}'

				self.plot_box_and_label(img_ori, 1, xyxy, label, color=(0,0,255))

			img_src = np.asarray(img_ori)

			cv2.namedWindow('test', cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
			cv2.resizeWindow('test', img_src.shape[1], img_src.shape[0])
			cv2.imshow('test', img_src)
			cv2.waitKey(0)  # 1 millisecond
		
		if len(det) > 0:
			relative_detections = det[0][0:4] / self.img_size[0]
			return det[0][0:5],relative_detections
		else:
			return None, None


if __name__ == "__main__":
	weights=Path(r'/media/access/New Volume/YOLOv6/runs/train/exp2 (with heavy augmentation)/weights/best_stop_aug_ckpt.pt').as_posix()
	device = '0'
	yaml = Path(r"/media/access/New Volume/YOLOv6/data/dataset.yaml").as_posix()
	img_size = 416

	img = cv2.imread(Path(r"/media/access/New Volume/YOLOv6/data/custom_dataset/images/val/6998831_10201_0.png").as_posix())

	detector = Yolov6Detector(weights, device, yaml, img_size)
	detector.detect(img, view_img=False)
	a= 2

	# inferer.infer(conf_thres, iou_thres, classes, agnostic_nms, max_det, save_dir, save_txt, save_img, hide_labels, hide_conf, view_img)

	a = 2