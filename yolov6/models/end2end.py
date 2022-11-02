import torch
import torch.nn as nn
import random


class ORT_NMS(torch.autograd.Function):
    '''ONNX-Runtime NMS operation'''
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


class TRT8_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        background_class=-1,
        box_coding=1,
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
                 box_coding=1,
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
        return nums, boxes, scores, classes

class TRT7_NMS(torch.autograd.Function):
    '''TensorRT NMS operation'''
    @staticmethod
    def forward(
        ctx,
        boxes,
        scores,
        plugin_version="1",
        shareLocation=1,
        backgroundLabelId=-1,
        numClasses=80,
        topK=1000,
        keepTopK=100,
        scoreThreshold=0.25,
        iouThreshold=0.45,
        isNormalized=0,
        clipBoxes=0,
        scoreBits=16,
        caffeSemantics=1,
    ):
        batch_size, num_boxes, numClasses = scores.shape
        num_det = torch.randint(0, keepTopK, (batch_size, 1), dtype=torch.int32)
        det_boxes = torch.randn(batch_size, keepTopK, 4)
        det_scores = torch.randn(batch_size, keepTopK)
        det_classes = torch.randint(0, numClasses, (batch_size, keepTopK)).float()
        return num_det, det_boxes, det_scores, det_classes
    @staticmethod
    def symbolic(g,
                 boxes,
                 scores,
                 plugin_version='1',
                 shareLocation=1,
                 backgroundLabelId=-1,
                 numClasses=80,
                 topK=1000,
                 keepTopK=100,
                 scoreThreshold=0.25,
                 iouThreshold=0.45,
                 isNormalized=0,
                 clipBoxes=0,
                 scoreBits=16,
                 caffeSemantics=1,
                 ):
        out = g.op("TRT::BatchedNMSDynamic_TRT", # BatchedNMS_TRT BatchedNMSDynamic_TRT
                   boxes,
                   scores,
                   shareLocation_i=shareLocation,
                   plugin_version_s=plugin_version,
                   backgroundLabelId_i=backgroundLabelId,
                   numClasses_i=numClasses,
                   topK_i=topK,
                   keepTopK_i=keepTopK,
                   scoreThreshold_f=scoreThreshold,
                   iouThreshold_f=iouThreshold,
                   isNormalized_i=isNormalized,
                   clipBoxes_i=clipBoxes,
                   scoreBits_i=scoreBits,
                   caffeSemantics_i=caffeSemantics,
                   outputs=4)
        nums, boxes, scores, classes = out
        return nums, boxes, scores, classes


class ONNX_ORT(nn.Module):
    '''onnx module with ONNX-Runtime NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None):
        super().__init__()
        self.device = device if device else torch.device("cpu")
        self.max_obj = torch.tensor([max_obj]).to(device)
        self.iou_threshold = torch.tensor([iou_thres]).to(device)
        self.score_threshold = torch.tensor([score_thres]).to(device)
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=self.device)

    def forward(self, x):
        batch, anchors, _ = x.shape
        box = x[:, :, :4]
        conf = x[:, :, 4:5]
        score = x[:, :, 5:]
        score *= conf

        nms_box = box @ self.convert_matrix
        nms_score = score.transpose(1, 2).contiguous()

        selected_indices = ORT_NMS.apply(nms_box, nms_score, self.max_obj, self.iou_threshold, self.score_threshold)
        batch_inds, cls_inds, box_inds = selected_indices.unbind(1)
        selected_score = nms_score[batch_inds, cls_inds, box_inds].unsqueeze(1)
        selected_box = nms_box[batch_inds, box_inds, ...]

        dets = torch.cat([selected_box, selected_score], dim=1)

        batched_dets = dets.unsqueeze(0).repeat(batch, 1, 1)
        batch_template = torch.arange(0, batch, dtype=batch_inds.dtype, device=batch_inds.device)
        batched_dets = batched_dets.where((batch_inds == batch_template.unsqueeze(1)).unsqueeze(-1),batched_dets.new_zeros(1))

        batched_labels = cls_inds.unsqueeze(0).repeat(batch, 1)
        batched_labels = batched_labels.where((batch_inds == batch_template.unsqueeze(1)),batched_labels.new_ones(1) * -1)

        N = batched_dets.shape[0]

        batched_dets = torch.cat((batched_dets, batched_dets.new_zeros((N, 1, 5))), 1)
        batched_labels = torch.cat((batched_labels, -batched_labels.new_ones((N, 1))), 1)

        _, topk_inds = batched_dets[:, :, -1].sort(dim=1, descending=True)

        topk_batch_inds = torch.arange(batch, dtype=topk_inds.dtype, device=topk_inds.device).view(-1, 1)
        batched_dets = batched_dets[topk_batch_inds, topk_inds, ...]
        det_classes = batched_labels[topk_batch_inds, topk_inds, ...]
        det_boxes, det_scores = batched_dets.split((4, 1), -1)
        det_scores = det_scores.squeeze(-1)
        num_det = (det_scores > 0).sum(1, keepdim=True)
        return num_det, det_boxes, det_scores, det_classes

class ONNX_TRT7(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.shareLocation = 1
        self.backgroundLabelId = -1
        self.numClasses = 80
        self.topK = 1000
        self.keepTopK = max_obj
        self.scoreThreshold = score_thres
        self.iouThreshold = iou_thres
        self.isNormalized = 0
        self.clipBoxes = 0
        self.scoreBits = 16
        self.caffeSemantics = 1
        self.plugin_version = '1'
        self.convert_matrix = torch.tensor([[1, 0, 1, 0], [0, 1, 0, 1], [-0.5, 0, 0.5, 0], [0, -0.5, 0, 0.5]],
                                           dtype=torch.float32,
                                           device=self.device)
    def forward(self, x):
        box = x[:, :, :4]
        conf = x[:, :, 4:5]
        score = x[:, :, 5:]
        score *= conf
        box @= self.convert_matrix
        box = box.unsqueeze(2)
        self.numClasses = int(score.shape[2])
        num_det, det_boxes, det_scores, det_classes = TRT7_NMS.apply(box, score, self.plugin_version,
                                                                     self.shareLocation,
                                                                     self.backgroundLabelId,
                                                                     self.numClasses,
                                                                     self.topK,
                                                                     self.keepTopK,
                                                                     self.scoreThreshold,
                                                                     self.iouThreshold,
                                                                     self.isNormalized,
                                                                     self.clipBoxes,
                                                                     self.scoreBits,
                                                                     self.caffeSemantics,
                                                                     )
        return num_det, det_boxes, det_scores, det_classes.int()


class ONNX_TRT8(nn.Module):
    '''onnx module with TensorRT NMS operation.'''
    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None):
        super().__init__()
        self.device = device if device else torch.device('cpu')
        self.background_class = -1,
        self.box_coding = 1,
        self.iou_threshold = iou_thres
        self.max_obj = max_obj
        self.plugin_version = '1'
        self.score_activation = 0
        self.score_threshold = score_thres

    def forward(self, x):
        box = x[:, :, :4]
        conf = x[:, :, 4:5]
        score = x[:, :, 5:]
        score *= conf
        num_det, det_boxes, det_scores, det_classes = TRT8_NMS.apply(box, score, self.background_class, self.box_coding,
                                                                    self.iou_threshold, self.max_obj,
                                                                    self.plugin_version, self.score_activation,
                                                                    self.score_threshold)
        return num_det, det_boxes, det_scores, det_classes


class End2End(nn.Module):
    '''export onnx or tensorrt model with NMS operation.'''
    def __init__(self, model, max_obj=100, iou_thres=0.45, score_thres=0.25, device=None, ort=False,  trt_version=8, with_preprocess=False):
        super().__init__()
        device = device if device else torch.device('cpu')
        self.with_preprocess = with_preprocess
        self.model = model.to(device)
        TRT = ONNX_TRT8 if trt_version >= 8  else ONNX_TRT7
        self.patch_model = ONNX_ORT if ort else TRT
        self.end2end = self.patch_model(max_obj, iou_thres, score_thres, device)
        self.end2end.eval()

    def forward(self, x):
        if self.with_preprocess:
            x = x[:,[2,1,0],...]
            x = x * (1/255)
        x = self.model(x)
        if isinstance(x, list):
            x = x[0]
        else:
            x = x
        x = self.end2end(x)
        return x
