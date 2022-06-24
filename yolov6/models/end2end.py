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





class ONNX_ORT(nn.Module):

    def __init__(self, max_obj=100, iou_thres=0.45, score_thres=0.35, device=None, max_wh=640):
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



class End2End(nn.Module):

    def __init__(self, model, max_obj=100, iou_thres=0.45, score_thres=0.35, max_wh=4096, device=None):
        super().__init__()
        device = device if device else torch.device('cpu')
        self.model = model.to(device)
        self.end2end = ONNX_ORT(max_obj, iou_thres, score_thres, device, max_wh)
        self.end2end.eval()

    def forward(self, x):
        x = self.model(x)
        out = self.end2end(x)
        return out