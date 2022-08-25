import numpy as np
import cv2
import json
import sys


def box_area(boxes):
    """
    Args:
        boxes(np.ndarray): [N, 4]
    return: [N]
    """
    return (boxes[:, 2] - boxes[:, 0]) * (boxes[:, 3] - boxes[:, 1])


def box_iou(box1, box2):
    """
    Args:
        box1(np.ndarray): [N, 4]
        box2(np.ndarray): [M, 4]
    return: [N, M]
    """
    area1 = box_area(box1)
    area2 = box_area(box2)
    lt = np.maximum(box1[:, np.newaxis, :2], box2[:, :2])
    rb = np.minimum(box1[:, np.newaxis, 2:], box2[:, 2:])
    wh = rb - lt
    wh = np.maximum(0, wh)
    inter = wh[:, :, 0] * wh[:, :, 1]
    iou = inter / (area1[:, np.newaxis] + area2 - inter)
    return iou


def nms(boxes, scores, iou_threshold):
    """
    Non Max Suppression numpy implementation.
    args:
        boxes(np.ndarray): [N, 4]
        scores(np.ndarray): [N, 1]
        iou_threshold(float): Threshold of IoU.
    """
    idxs = scores.argsort()
    keep = []
    while idxs.size > 0:
        max_score_index = idxs[-1]
        max_score_box = boxes[max_score_index][None, :]
        keep.append(max_score_index)
        if idxs.size == 1:
            break
        idxs = idxs[:-1]
        other_boxes = boxes[idxs]
        ious = box_iou(max_score_box, other_boxes)
        idxs = idxs[ious[0] <= iou_threshold]

    keep = np.array(keep)
    return keep


class YOLOPostProcess(object):
    """
    Post process of YOLO-series network.
    args:
        score_threshold(float): Threshold to filter out bounding boxes with low
                confidence score. If not provided, consider all boxes.
        nms_threshold(float): The threshold to be used in NMS.
        multi_label(bool): Whether keep multi label in boxes.
        keep_top_k(int): Number of total bboxes to be kept per image after NMS
                step. -1 means keeping all bboxes after NMS step.
    """

    def __init__(self,
                 score_threshold=0.25,
                 nms_threshold=0.5,
                 multi_label=False,
                 keep_top_k=300):
        self.score_threshold = score_threshold
        self.nms_threshold = nms_threshold
        self.multi_label = multi_label
        self.keep_top_k = keep_top_k

    def _xywh2xyxy(self, x):
        # Convert from [x, y, w, h] to [x1, y1, x2, y2]
        y = np.copy(x)
        y[:, 0] = x[:, 0] - x[:, 2] / 2  # top left x
        y[:, 1] = x[:, 1] - x[:, 3] / 2  # top left y
        y[:, 2] = x[:, 0] + x[:, 2] / 2  # bottom right x
        y[:, 3] = x[:, 1] + x[:, 3] / 2  # bottom right y
        return y

    def _non_max_suppression(self, prediction):
        max_wh = 4096  # (pixels) minimum and maximum box width and height
        nms_top_k = 30000

        cand_boxes = prediction[..., 4] > self.score_threshold  # candidates
        output = [np.zeros((0, 6))] * prediction.shape[0]

        for batch_id, boxes in enumerate(prediction):
            # Apply constraints
            boxes = boxes[cand_boxes[batch_id]]
            if not boxes.shape[0]:
                continue
            # Compute conf (conf = obj_conf * cls_conf)
            boxes[:, 5:] *= boxes[:, 4:5]

            # Box (center x, center y, width, height) to (x1, y1, x2, y2)
            convert_box = self._xywh2xyxy(boxes[:, :4])

            # Detections matrix nx6 (xyxy, conf, cls)
            if self.multi_label:
                i, j = (boxes[:, 5:] > self.score_threshold).nonzero()
                boxes = np.concatenate(
                    (convert_box[i], boxes[i, j + 5, None],
                     j[:, None].astype(np.float32)),
                    axis=1)
            else:
                conf = np.max(boxes[:, 5:], axis=1)
                j = np.argmax(boxes[:, 5:], axis=1)
                re = np.array(conf.reshape(-1) > self.score_threshold)
                conf = conf.reshape(-1, 1)
                j = j.reshape(-1, 1)
                boxes = np.concatenate((convert_box, conf, j), axis=1)[re]

            num_box = boxes.shape[0]
            if not num_box:
                continue
            elif num_box > nms_top_k:
                boxes = boxes[boxes[:, 4].argsort()[::-1][:nms_top_k]]

            # Batched NMS
            c = boxes[:, 5:6] * max_wh
            clean_boxes, scores = boxes[:, :4] + c, boxes[:, 4]
            keep = nms(clean_boxes, scores, self.nms_threshold)
            # limit detection box num
            if keep.shape[0] > self.keep_top_k:
                keep = keep[:self.keep_top_k]
            output[batch_id] = boxes[keep]
        return output

    def __call__(self, outs, scale_factor):
        preds = self._non_max_suppression(outs)
        bboxs, box_nums = [], []
        for i, pred in enumerate(preds):
            if len(pred.shape) > 2:
                pred = np.squeeze(pred)
            if len(pred.shape) == 1:
                pred = pred[np.newaxis, :]
            pred_bboxes = pred[:, :4]
            scale = np.tile(scale_factor[i][::-1], (2))
            pred_bboxes /= scale
            bbox = np.concatenate(
                [
                    pred[:, -1][:, np.newaxis], pred[:, -2][:, np.newaxis],
                    pred_bboxes
                ],
                axis=-1)
            bboxs.append(bbox)
            box_num = bbox.shape[0]
            box_nums.append(box_num)
        bboxs = np.concatenate(bboxs, axis=0)
        box_nums = np.array(box_nums)
        return {'bbox': bboxs, 'bbox_num': box_nums}


def coco_metric(anno_file, bboxes_list, bbox_nums_list, image_id_list):
    try:
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
    except:
        print(
            "[ERROR] Not found pycocotools, please install by `pip install pycocotools`"
        )
        sys.exit(1)

    coco_gt = COCO(anno_file)
    cats = coco_gt.loadCats(coco_gt.getCatIds())
    clsid2catid = {i: cat['id'] for i, cat in enumerate(cats)}
    results = []
    for bboxes, bbox_nums, image_id in zip(bboxes_list, bbox_nums_list,
                                           image_id_list):
        results += _get_det_res(bboxes, bbox_nums, image_id, clsid2catid)

    output = "bbox.json"
    with open(output, 'w') as f:
        json.dump(results, f)

    coco_dt = coco_gt.loadRes(output)
    coco_eval = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval.evaluate()
    coco_eval.accumulate()
    coco_eval.summarize()
    return coco_eval.stats


def _get_det_res(bboxes, bbox_nums, image_id, label_to_cat_id_map):
    det_res = []
    k = 0
    for i in range(len(bbox_nums)):
        cur_image_id = int(image_id[i][0])
        det_nums = bbox_nums[i]
        for j in range(det_nums):
            dt = bboxes[k]
            k = k + 1
            num_id, score, xmin, ymin, xmax, ymax = dt.tolist()
            if int(num_id) < 0:
                continue
            category_id = label_to_cat_id_map[int(num_id)]
            w = xmax - xmin
            h = ymax - ymin
            bbox = [xmin, ymin, w, h]
            dt_res = {
                'image_id': cur_image_id,
                'category_id': category_id,
                'bbox': bbox,
                'score': score
            }
            det_res.append(dt_res)
    return det_res
