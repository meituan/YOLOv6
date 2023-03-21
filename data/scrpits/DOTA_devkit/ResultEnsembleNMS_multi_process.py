"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for 15 result files, for the format, you can refer to "http://captain.whu.edu.cn/DOTAweb/tasks.html"
    search for PATH_TO_BE_CONFIGURED to config the paths
    Note, the evaluation is on the large scale images
"""
import os
import numpy as np
import re
import time
import sys
sys.path.insert(0,'..')
import DOTA_devkit.dota_utils as util
import DOTA_devkit.polyiou as polyiou
import pdb
import math
from multiprocessing import Pool
from functools import partial
import shutil
import argparse

## the thresh for nms when merge image
nms_thresh = 0.2

def py_cpu_nms_poly(dets, thresh):
    scores = dets[:, 8]
    polys = []
    areas = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        for j in range(order.size - 1):
            iou = polyiou.iou_poly(polys[i], polys[order[j + 1]])
            ovr.append(iou)
        ovr = np.array(ovr)

        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(ovr <= thresh)[0]
        # print('inds: ', inds)

        order = order[inds + 1]

    return keep


def py_cpu_nms_poly_fast(dets, thresh):
    obbs = dets[:, 0:-1]
    x1 = np.min(obbs[:, 0::2], axis=1)
    y1 = np.min(obbs[:, 1::2], axis=1)
    x2 = np.max(obbs[:, 0::2], axis=1)
    y2 = np.max(obbs[:, 1::2], axis=1)
    scores = dets[:, 8]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)

    polys = []
    for i in range(len(dets)):
        tm_polygon = polyiou.VectorDouble([dets[i][0], dets[i][1],
                                            dets[i][2], dets[i][3],
                                            dets[i][4], dets[i][5],
                                            dets[i][6], dets[i][7]])
        polys.append(tm_polygon)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        ovr = []
        i = order[0]
        keep.append(i)
        # if order.size == 0:
        #     break
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])
        # w = np.maximum(0.0, xx2 - xx1 + 1)
        # h = np.maximum(0.0, yy2 - yy1 + 1)
        w = np.maximum(0.0, xx2 - xx1)
        h = np.maximum(0.0, yy2 - yy1)
        hbb_inter = w * h
        hbb_ovr = hbb_inter / (areas[i] + areas[order[1:]] - hbb_inter)
        # h_keep_inds = np.where(hbb_ovr == 0)[0]
        h_inds = np.where(hbb_ovr > 0)[0]
        tmp_order = order[h_inds + 1]
        for j in range(tmp_order.size):
            iou = polyiou.iou_poly(polys[i], polys[tmp_order[j]])
            hbb_ovr[h_inds[j]] = iou
            # ovr.append(iou)
            # ovr_index.append(tmp_order[j])

        # ovr = np.array(ovr)
        # ovr_index = np.array(ovr_index)
        # print('ovr: ', ovr)
        # print('thresh: ', thresh)
        try:
            if math.isnan(ovr[0]):
                pdb.set_trace()
        except:
            pass
        inds = np.where(hbb_ovr <= thresh)[0]

        # order_obb = ovr_index[inds]
        # print('inds: ', inds)
        # order_hbb = order[h_keep_inds + 1]
        order = order[inds + 1]
        # pdb.set_trace()
        # order = np.concatenate((order_obb, order_hbb), axis=0).astype(np.int)
    return keep

def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    #print('dets:', dets)
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    ## index for dets
    order = scores.argsort()[::-1]


    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]

    return keep

def nmsbynamedict(nameboxdict, nms, thresh):
    nameboxnmsdict = {x: [] for x in nameboxdict}
    for imgname in nameboxdict:
        #print('imgname:', imgname)
        #keep = py_cpu_nms(np.array(nameboxdict[imgname]), thresh)
        #print('type nameboxdict:', type(nameboxnmsdict))
        #print('type imgname:', type(imgname))
        #print('type nms:', type(nms))
        keep = nms(np.array(nameboxdict[imgname]), thresh)
        #print('keep:', keep)
        outdets = []
        #print('nameboxdict[imgname]: ', nameboxnmsdict[imgname])
        for index in keep:
            # print('index:', index)
            outdets.append(nameboxdict[imgname][index])
        nameboxnmsdict[imgname] = outdets
    return nameboxnmsdict
def poly2origpoly(poly, x, y, rate):
    origpoly = []
    for i in range(int(len(poly)/2)):
        tmp_x = float(poly[i * 2] + x) / float(rate)
        tmp_y = float(poly[i * 2 + 1] + y) / float(rate)
        origpoly.append(tmp_x)
        origpoly.append(tmp_y)
    return origpoly

def mergesingle(dstpath, nms, fullname):
    name = util.custombasename(fullname)
    #print('name:', name)
    dstname = os.path.join(dstpath, name + '.txt')
    print(dstname)
    with open(fullname, 'r') as f_in:
        nameboxdict = {}
        lines = f_in.readlines()
        splitlines = [x.strip().split(' ') for x in lines]
        for splitline in splitlines:
            # splitline = [(该目标所处图片名称), confidence, x1, y1, x2, y2, x3, y3, x4, y4]
            #subname = splitline[0]
            oriname = splitline[0]
            #splitname = subname.split('__')
            #oriname = splitname[0]
            # pattern1 = re.compile(r'__\d+___\d+')
            # #print('subname:', subname)
            # x_y = re.findall(pattern1, subname)
            # x_y_2 = re.findall(r'\d+', x_y[0])
            # x, y = int(x_y_2[0]), int(x_y_2[1])

            # pattern2 = re.compile(r'__([\d+\.]+)__\d+___')

            # rate = re.findall(pattern2, subname)[0]

            confidence = splitline[1]
            poly = list(map(float, splitline[2:]))
            #origpoly = poly2origpoly(poly, x, y, rate)
            det = poly # shape(8)
            det.append(confidence)
            det = list(map(float, det))
            if (oriname not in nameboxdict):
                nameboxdict[oriname] = []
            nameboxdict[oriname].append(det)
        nameboxnmsdict = nmsbynamedict(nameboxdict, nms, nms_thresh)
        with open(dstname, 'w') as f_out:
            for imgname in nameboxnmsdict:
                for det in nameboxnmsdict[imgname]:
                    #print('det:', det)
                    confidence = det[-1]
                    confidence = round(confidence, 2)
                    bbox = det[0:-1]

                    bbox[0] = round(bbox[0], 1)
                    bbox[1] = round(bbox[1], 1)
                    bbox[2] = round(bbox[2], 1)
                    bbox[3] = round(bbox[3], 1)
                    bbox[4] = round(bbox[4], 1)
                    bbox[5] = round(bbox[5], 1)
                    bbox[6] = round(bbox[6], 1)
                    bbox[7] = round(bbox[7], 1)

                    outline = imgname + ' ' + str(confidence) + ' ' + ' '.join(map(str, bbox))
                    #print('outline:', outline)
                    f_out.write(outline + '\n')

def mergebase_parallel(srcpath, dstpath, nms):
    pool = Pool(16)
    filelist = util.GetFileFromThisRootDir(srcpath)

    mergesingle_fn = partial(mergesingle, dstpath, nms)
    # pdb.set_trace()
    pool.map(mergesingle_fn, filelist)

def mergebase(srcpath, dstpath, nms):
    filelist = util.GetFileFromThisRootDir(srcpath)
    for filename in filelist:
        mergesingle(dstpath, nms, filename)

def mergebyrec(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000'
    # dstpath = r'E:\bod-dataset\results\bod-v3_rfcn_2000000_nms'
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)  # delete output folderX
    os.makedirs(dstpath)
    
    mergebase(srcpath,
              dstpath,
              py_cpu_nms)
def mergebypoly(srcpath, dstpath):
    """
    srcpath: result files before merge and nms
    dstpath: result files after merge and nms
    """
    # srcpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/comp4_test_results'
    # dstpath = r'/home/dingjian/evaluation_task1/result/faster-rcnn-59/testtime'
    if os.path.exists(dstpath):
        shutil.rmtree(dstpath)  # delete output folderX
    os.makedirs(dstpath)

    # mergebase(srcpath,
    #           dstpath,
    #           py_cpu_nms_poly)
    mergebase_parallel(srcpath,
              dstpath,
              py_cpu_nms_poly_fast)

def parse_args():
    parser = argparse.ArgumentParser(description='MMDet test (and eval) a model')
    parser.add_argument('--scrpath', default='/OrientedRepPoints/tools/parse_pkl/evaluation_results/orientedreppoints_ROIRT_ensemble', help='test config file path')
    parser.add_argument('--dstpath', default='/OrientedRepPoints/tools/parse_pkl/evaluation_results/orientedreppoints_ROIRT_ensemble_nms', help='checkpoint file')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    args = parse_args()

    mergebypoly(srcpath=args.scrpath,
                dstpath=args.dstpath)
    print('Result Merge Done!')
    # mergebyrec()