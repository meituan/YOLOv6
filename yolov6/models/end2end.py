import torch
import torch.nn as nn
import random

class ORT_NMS(torch.autograd.Function):

    @staticmethod
    def forward(ctx,
                boxes,
                scores,
                max_output_boxes_per_class=torch.tensor([100]),
                iou_threshold=torch.tensor([0.45]),
                score_threshold=torch.tensor([0.25])):
        device = boxes.device
        batch = scores.shape[0]
        num_det = random.randint(0, 100)
        batches = torch.randint(0, batch, (num_det,)).sort()[0].to(device)
        idxs = torch.arange(100, 100 + num_det).to(device)
        zeros = torch.zeros((num_det,), dtype=torch.int64).to(device)
        selected_indices = torch.cat([batches[None], zeros[None], idxs[None]], 0).T.contiguous()
        selected_indices = selected_indices.to(torch.int64)
        return selected_indices

    @staticmethod
    def symbolic(g, boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold):
        return g.op("NonMaxSuppression", boxes, scores, max_output_boxes_per_class, iou_threshold, score_threshold)

class TRT_NMS(torch.autograd.Function):
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=0,
        iou_threshold=0.45,
        max_output_boxes=100,
        plugin_version="1",
        score_activation=0,
        score_threshold=0.25,
    ):
        batch_size, num_boxes, num_classes = scores.shape
        num_det = torch.randint(0, max_output_boxes, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, max_output_boxes, 4)
        det_scores = torch.randn(batch_size, max_output_boxes)
        det_classes = torch.randint(0, num_classes, (batch_size, max_output_boxes), dtype=torch.int32)

        return num_det, det_boxes, det_scores, det_classes

    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 background_class=-1,
                 box_coding=0,
                 iou_threshold=0.45,
                 max_output_boxes=100,
                 plugin_version="1",
                 score_activation=0,
                 score_threshold=0.25):
        out = g.op("TRT::EfficientNMS_TRT",
                     boxes,
                     scores,
                     background_class_i=background_class,
                     box_coding_i=box_coding,
                     iou_threshold_f=iou_threshold,
                     max_output_boxes_i=max_output_boxes,
                     plugin_version_s=plugin_version,
                     score_activation_i=score_activation,
                     score_threshold_f=score_threshold,
                     outputs=4)
        nums, boxes, scores, classes = out
        return nums,boxes,scores,classes



class ONNX_ORT(nn.Module):

    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=640, device=None):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.max_obj = torch.tensor([max_obj]).to(device)
        self.iou_threshold = torch.tensor([iou_thres]).to(device)
        self.score_threshold = torch.tensor([score_thres]).to(device)
        self.max_wh = max_wh
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=self.device)

    def forward(self, x):
        box = x[:, :, :4]
        conf = x[:, :, 4:5]
        score = x[:, :, 5:]
        score *= conf
        box @= self.convert_matrix
        objScore, objCls = score.max(2, keepdim=True)
        dis = objCls.float() * self.max_wh
        nmsbox = box + dis
        objScore1 = objScore.transpose(1, 2).contiguous()
        selected_indices = ORT_NMS.apply(nmsbox, objScore1, self.max_obj, self.iou_threshold, self.score_threshold)
        X, Y = selected_indices[:, 0], selected_indices[:, 2]
        resBoxes = box[X, Y, :]
        resClasses = objCls[X, Y, :].float()
        resScores = objScore[X, Y, :]
        X = X.unsqueeze(1).float()
        return torch.concat([X, resBoxes, resClasses, resScores], 1)

class ONNX_TRT(nn.Module):

    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None ,device=None):
        super().__init__()
        assert max_wh is None
        self.device = device if device else torch.device('cpu')
        self.background_class = -1,
        self.box_coding = 0,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=self.device)

    def forward(self, x):
        box = x[:, :, :4]
        conf = x[:, :, 4:5]
        score = x[:, :, 5:]
        score *= conf
        box @= self.convert_matrix
        num_det, det_boxes, det_scores, det_classes = TRT_NMS.apply(box, score, self.background_class, self.box_coding,
                                                                    self.iou_threshold, self.max_obj,
                                                                    self.plugin_version, self.score_activation,
                                                                    self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes


class End2End(nn.Module):

    def __init__(self, model, max_obj=100, iou_thres=0.45, score_thres=0.25, max_wh=None, device=None):
        super().__init__()
        device = device if device else torch.device('cpu')
        self.model = model.to(device)
        self.patch_model = ONNX_TRT if max_wh is None else ONNX_ORT
        self.end2end = self.patch_model(max_obj, iou_thres, score_thres, max_wh, device)
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        x = self.end2end(x)
        return x
