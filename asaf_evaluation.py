from pathlib import Path
from asaf_yolov6_detector_class import Yolov6Detector
from tqdm import tqdm
import cv2
import numpy as np
import boto3
from pygit2 import Repository, Commit
from datetime import datetime

def draw_bbox_on_image(img, box, color=(0,0,255)):
	if box is None:
		return img
	img = cv2.rectangle(img, (box[0],box[1]), (box[2], box[3]), color, 1)
	return img



def get_iou(box1, box2):
	""" We assume that the box follows the format:
		box1 = [x1,y1,x2,y2], and box2 = [x3,y3,x4,y4],
		where (x1,y1) and (x3,y3) represent the top left coordinate,
		and (x2,y2) and (x4,y4) represent the bottom right coordinate """
	x1, y1, x2, y2 = box1	
	x3, y3, x4, y4 = box2
	x_inter1 = max(x1, x3)
	y_inter1 = max(y1, y3)
	x_inter2 = min(x2, x4)
	y_inter2 = min(y2, y4)
	width_inter = abs(x_inter2 - x_inter1)
	height_inter = abs(y_inter2 - y_inter1)
	area_inter = width_inter * height_inter
	width_box1 = abs(x2 - x1)
	height_box1 = abs(y2 - y1)
	width_box2 = abs(x4 - x3)
	height_box2 = abs(y4 - y3)
	area_box1 = width_box1 * height_box1
	area_box2 = width_box2 * height_box2
	area_union = area_box1 + area_box2 - area_inter
	iou = area_inter / area_union
	return iou





def evaluate(detector:Yolov6Detector, imgs_path:Path, annotations_path:Path):
	sorted_images  = sorted(list(imgs_path.iterdir()), key = lambda x: x.name.split('.'))
	sorted_txts  = sorted(list(annotations_path.iterdir()), key = lambda x: x.name.split('.'))
	couples = list(zip(sorted_images,sorted_txts))
	TP, TN, FP, FN = 0,0,0,0
	for couple in tqdm(couples, total= len(couples)):
		img = cv2.imread(couple[0].as_posix())
		if img.shape != (416,416,3):
			img = cv2.resize(img, (416,416))
		detections, relative_detections = detector.detect(img) # (x1, y1, x2, y2)
		detected_bbox= relative_detections.cpu().numpy() if relative_detections is not None else None
		with open(couple[1].as_posix(), 'r') as f:
			value = f.readline()
			value = eval('['+ value.replace(' ', ',')+']')
			annotations = np.array([value[1], value[2], value[3], value[4]]) if len(value)>3 else None
			tl = (annotations[0] - 0.5*annotations[2], annotations[1] - 0.5*annotations[3]) if len(value)>3 else None
			br = (annotations[0] + 0.5*annotations[2], annotations[1] + 0.5*annotations[3]) if len(value)>3 else None
			annotated_bbox = np.array([tl[0], tl[1], br[0], br[1]]) if len(value)>3 else None

		if annotated_bbox is None and detected_bbox is None: #means both bboxes are none
			TN+=1
		elif annotated_bbox is not None and detected_bbox is None:
			FN+=1
		elif annotated_bbox is None and detected_bbox is not None:
			FP+=1
		else:
			iou = get_iou(detected_bbox,annotated_bbox)
			if iou > 0.97:
				TP+=1
			if iou < 0.97:
				FP+=1

	
		detected_bbox_for_drawing = [round(it) for it in detections.cpu().numpy()] if detections is not None else None
		annotated_bbox_for_drawing = [round(it*416) for it in annotated_bbox] if detections is not None else None
		img = draw_bbox_on_image(img, detected_bbox_for_drawing,(0,0,255)) # detected in red
		img = draw_bbox_on_image(img, annotated_bbox_for_drawing,(255,0,0)) # annotated in blue
		img_to_show = cv2.resize(img, (1280, 720), cv2.INTER_LINEAR)
		cv2.imshow('test', img_to_show)
		cv2.waitKey(1)




	accuracy = (TP + TN)/(TP + TN + FP + FN)
	precision = TP/(TP + FP)
	recall = TP/(TP + FN)
	f1_score = (2 * precision * recall)/(precision + recall)
	a =2







if __name__ == "__main__":
	weights=Path(r'/media/access/New Volume/YOLOv6/runs/train/exp2 (with heavy augmentation)/weights/best_stop_aug_ckpt.pt').as_posix()
	yaml = Path(r"/media/access/New Volume/YOLOv6/data/dataset.yaml").as_posix()
	detector = Yolov6Detector(weights,yaml)

	imgs_path = Path(r"/media/access/New Volume/YOLOv6/data/custom_dataset/images/val")
	annotations_path = Path(r"/media/access/New Volume/YOLOv6/data/custom_dataset/labels/val")
	evaluate(detector,imgs_path,annotations_path)