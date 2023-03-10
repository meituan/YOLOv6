import torch
import numpy as np
from .figure_iou import pairwise_bbox_iou

# reference: https://github.com/dongdonghy/repulsion_loss_pytorch/blob/master/repulsion_loss.py
def IoG(gt_box, pred_box):
    inter_xmin = torch.max(gt_box[:, 0], pred_box[:, 0])
    inter_ymin = torch.max(gt_box[:, 1], pred_box[:, 1])
    inter_xmax = torch.min(gt_box[:, 2], pred_box[:, 2])
    inter_ymax = torch.min(gt_box[:, 3], pred_box[:, 3])
    Iw = torch.clamp(inter_xmax - inter_xmin, min=0)
    Ih = torch.clamp(inter_ymax - inter_ymin, min=0)
    I = Iw * Ih
    G = ((gt_box[:, 2] - gt_box[:, 0]) * (gt_box[:, 3] - gt_box[:, 1])).clamp(1e-6)
    return I / G

def smooth_ln(x, sigma=0.5):
    return torch.where(
        torch.le(x, sigma),
        -torch.log(1 - x),
        ((x - sigma) / (1 - sigma)) - np.log(1 - sigma)
    )

def repulsion_loss(pbox, gtbox, fg_mask, sigma_repgt=0.9, sigma_repbox=0, pnms=0, gtnms=0):#nms=0
    loss_repgt=torch.zeros(1).to(pbox.device)
    loss_repbox=torch.zeros(1).to(pbox.device)
    bbox_mask = fg_mask.unsqueeze(-1).repeat([1, 1, 4])
    bs = 0
    pbox = pbox.detach()
    gtbox = gtbox.detach()

    for idx in range(pbox.shape[0]):
        num_pos = bbox_mask[idx].sum()
        if num_pos <= 0:
            continue
        _pbox_pos = torch.masked_select(pbox[idx],bbox_mask[idx]).reshape([-1, 4])
        _gtbox_pos = torch.masked_select(gtbox[idx],bbox_mask[idx]).reshape([-1, 4])
        
        bs += 1
        pgiou = pairwise_bbox_iou(_pbox_pos, _gtbox_pos, box_format='xyxy')
        ppiou = pairwise_bbox_iou(_pbox_pos, _pbox_pos, box_format='xyxy')

        pgiou = pgiou.cuda().data.cpu().numpy()
        ppiou = ppiou.cuda().data.cpu().numpy()
        _gtbox_pos_cpu = _gtbox_pos.cuda().data.cpu().numpy()
       
        for j in range(pgiou.shape[0]):
            for z in range(j, pgiou.shape[0]):
                ppiou[j, z] = 0
                if (_gtbox_pos_cpu[j][0]==_gtbox_pos_cpu[z][0]) and (_gtbox_pos_cpu[j][1]==_gtbox_pos_cpu[z][1]) \
                        and (_gtbox_pos_cpu[j][2]==_gtbox_pos_cpu[z][2]) and (_gtbox_pos_cpu[j][3]==_gtbox_pos_cpu[z][3]):
                    pgiou[j, z] = 0
                    pgiou[z, j] = 0
                    ppiou[z, j] = 0

        pgiou = torch.from_numpy(pgiou).to(pbox.device).detach()
        ppiou = torch.from_numpy(ppiou).to(pbox.device).detach()
        # repgt
        max_iou, _ = torch.max(pgiou, 1)
        pg_mask = torch.gt(max_iou, gtnms)
        num_repgt = pg_mask.sum()
        #print('num_repgt',num_repgt)
        if num_repgt > 0:
            pgiou_pos = pgiou[pg_mask, :]
            _, argmax_iou_sec = torch.max(pgiou_pos, 1)
            pbox_sec = _pbox_pos[pg_mask, :]
            gtbox_sec = _gtbox_pos[argmax_iou_sec, :]
            IOG = IoG(gtbox_sec, pbox_sec)
            loss_repgt += smooth_ln(IOG, sigma_repgt).mean()


        # repbox
        pp_mask = torch.gt(ppiou, pnms)
        num_pbox = pp_mask.sum()
        #print('num_pbox',num_pbox)
        if num_pbox > 0:
            loss_repbox += smooth_ln(ppiou, sigma_repbox).mean()

    loss_repgt /= bs
    loss_repbox /= bs

    torch.cuda.empty_cache()
    
    return loss_repgt.squeeze(0), loss_repbox.squeeze(0)