# --------------------------------------------------------
# mAOEevaluation
# --------------------------------------------------------

"""
    To use the code, users should to config detpath, annopath and imagesetfile
    detpath is the path for class result files. The evaluation is performed on the merging results.
"""

import numpy as np
import polyiou
from dota_poly2rbox import poly2rbox_single_v2, poly2rbox_single_v3

def parse_gt(filename):
    """
    :param filename: ground truth file to parse
    :return: all instances in a picture
    """
    objects = []
    with  open(filename, 'r') as f:
        while True:
            line = f.readline()
            if line:
                splitlines = line.strip().split(' ')
                object_struct = {}
                if (len(splitlines) < 9):
                    continue
                object_struct['name'] = splitlines[8]

                if (len(splitlines) == 9):
                    object_struct['difficult'] = 0
                elif (len(splitlines) == 10):
                    object_struct['difficult'] = int(splitlines[9])
                object_struct['bbox'] = [float(splitlines[0]),
                                         float(splitlines[1]),
                                         float(splitlines[2]),
                                         float(splitlines[3]),
                                         float(splitlines[4]),
                                         float(splitlines[5]),
                                         float(splitlines[6]),
                                         float(splitlines[7])]
                objects.append(object_struct)
            else:
                break
    return objects


def aoe_eval(detpath,
             annopath,
             imagesetfile,
             classname,
            # cachedir,
             ovthresh=0.5):
    """rec, prec, ap = aoe_eval(detpath,
                                annopath,
                                imagesetfile,
                                classname,
                                [ovthresh])

    detpath: Path to detections
        detpath.format(classname) should produce the detection results file.
    annopath: Path to annotations
        annopath.format(imagename) should be the xml annotations file.
    imagesetfile: Text file containing the list of images, one image per line.
    classname: Category name (duh)
    cachedir: Directory for caching the annotations
    [ovthresh]: Overlap threshold (default = 0.7)
    """

    with open(imagesetfile, 'r') as f:
        lines = f.readlines()
    imagenames = [x.strip() for x in lines]
    
    recs = {}
    for i, imagename in enumerate(imagenames):
        recs[imagename] = parse_gt(annopath.format(imagename))

    # extract gt objects for this class
    class_recs = {}
    npos = 0
    for imagename in imagenames:
        R = [obj for obj in recs[imagename] if obj['name'] == classname]
        bbox = np.array([x['bbox'] for x in R])
        difficult = np.array([x['difficult'] for x in R]).astype(np.bool)
        det = [False] * len(R)
        npos = npos + sum(~difficult)
        class_recs[imagename] = {'bbox': bbox,
                                 'difficult': difficult,
                                 'det': det}

    # read dets from Task1* files
    detfile = detpath.format(classname)
    with open(detfile, 'r') as f:
        lines = f.readlines()

    splitlines = [x.strip().split(' ') for x in lines]
    image_ids = [x[0] for x in splitlines]
    confidence = np.array([float(x[1]) for x in splitlines])

    BB = np.array([[float(z) for z in x[2:]] for x in splitlines])

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    # sorted_scores = np.sort(-confidence)

    ## note the usage only in numpy not for list
    BB = BB[sorted_ind, :]
    image_ids = [image_ids[x] for x in sorted_ind]
    # go down dets and mark TPs and FPs
    nd = len(image_ids)

    angle_dif_list = []
    for d in range(nd):
        R = class_recs[image_ids[d]]
        bb = BB[d, :].astype(float)
        ovmax = -np.inf
        BBGT = R['bbox'].astype(float)

        ## compute det bb with each BBGT
        if BBGT.size > 0:
            # 1. calculate the overlaps between hbbs, if the iou between hbbs are 0, the iou between obbs are 0, too.
            # pdb.set_trace()
            BBGT_xmin =  np.min(BBGT[:, 0::2], axis=1)
            BBGT_ymin = np.min(BBGT[:, 1::2], axis=1)
            BBGT_xmax = np.max(BBGT[:, 0::2], axis=1)
            BBGT_ymax = np.max(BBGT[:, 1::2], axis=1)
            bb_xmin = np.min(bb[0::2])
            bb_ymin = np.min(bb[1::2])
            bb_xmax = np.max(bb[0::2])
            bb_ymax = np.max(bb[1::2])

            ixmin = np.maximum(BBGT_xmin, bb_xmin)
            iymin = np.maximum(BBGT_ymin, bb_ymin)
            ixmax = np.minimum(BBGT_xmax, bb_xmax)
            iymax = np.minimum(BBGT_ymax, bb_ymax)
            iw = np.maximum(ixmax - ixmin + 1., 0.)
            ih = np.maximum(iymax - iymin + 1., 0.)
            inters = iw * ih

            # union
            uni = ((bb_xmax - bb_xmin + 1.) * (bb_ymax - bb_ymin + 1.) +
                   (BBGT_xmax - BBGT_xmin + 1.) *
                   (BBGT_ymax - BBGT_ymin + 1.) - inters)

            overlaps = inters / uni

            BBGT_keep_mask = overlaps > 0
            BBGT_keep = BBGT[BBGT_keep_mask, :]
            # BBGT_keep_index = np.where(overlaps > 0)[0]

            def calcoverlaps(BBGT_keep, bb):
                overlaps = []
                for index, GT in enumerate(BBGT_keep):
                    overlap = polyiou.iou_poly(polyiou.VectorDouble(BBGT_keep[index]), polyiou.VectorDouble(bb))
                    overlaps.append(overlap)
                return overlaps
            if len(BBGT_keep) > 0:
                overlaps = calcoverlaps(BBGT_keep, bb)
                ovmax = np.max(overlaps)
                if ovmax > ovthresh:
                    jmax = np.argmax(overlaps)
                    angle_box_GT = poly2rbox_single_v3(BBGT_keep[jmax])
                    angel_GT = angle_box_GT[-1]

                    angle_box_bb = poly2rbox_single_v3(bb)
                    angel_bb = angle_box_bb[-1]

                    angle_dif = abs(angel_bb - angel_GT) * 57.32
                    angle_dif_list.append(angle_dif)

    return angle_dif_list

def main():
    # detpath = r'/mnt/SSD/lwt_workdir/data/dota_angle/result_merge_fasterrcnn/{:s}.txt'
    # annopath = r'/mnt/SSD/lwt_workdir/data/dota_new/val/labelTxt/{:s}.txt' # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    # imagesetfile = r'/mnt/SSD/lwt_workdir/data/dota_new/val/test.txt'
    detpath = r'/data1/OrientedRepPoints/tools/parse_pkl/evaluation_results/40epoch_detection_results_merged/Task1_{:s}.txt'
    annopath = r'/dataset/buyingjia/Dota/Dota_V1.0/val/labelTxt/{:s}.txt'  # change the directory to the path of val/labelTxt, if you want to do evaluation on the valset
    imagesetfile = r'/data1/OrientedRepPoints/tools/parse_pkl/evaluation_results/imgnamefile_val1.0.txt'
    
    # For DOTA-v1.0
    classnames = ['plane', 'baseball-diamond', 'bridge', 'ground-track-field', 'small-vehicle', 'large-vehicle', 'ship', 'tennis-court',
                'basketball-court', 'storage-tank',  'soccer-ball-field', 'roundabout', 'harbor', 'swimming-pool', 'helicopter']
    
    # for hrsc2016
    # classnames = ['ship']
    
    # for ucas_aod
    # classname = ['airplane', 'car']
    
    classaps = []
    for classname in classnames:
        print('classname:', classname)
        angel_dif_list = aoe_eval(detpath,
             annopath,
             imagesetfile,
             classname,
             ovthresh=0.7) # set 0.7 as default

        angle_dif = 0.0

        for item in angel_dif_list:
            angle_dif = angle_dif+item

        angle_dif_ave = angle_dif/len(angel_dif_list)
        print('angle_dif_ave: ', angle_dif_ave)
        classaps.append(angle_dif_ave)

    print('mAOE: ', sum(classaps)/len(classaps))
if __name__ == '__main__':
    main()
