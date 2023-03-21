import os
import math
import argparse
import os.path as osp

import numpy as np

def cal_line_length(point1, point2):
    return math.sqrt(math.pow(point1[0] - point2[0], 2) + math.pow(point1[1] - point2[1], 2))

def get_best_begin_point_single(coordinate):
    x1, y1, x2, y2, x3, y3, x4, y4 = coordinate
    xmin = min(x1, x2, x3, x4)
    ymin = min(y1, y2, y3, y4)
    xmax = max(x1, x2, x3, x4)
    ymax = max(y1, y2, y3, y4)
    combinate = [[[x1, y1], [x2, y2], [x3, y3], [x4, y4]], [[x2, y2], [x3, y3], [x4, y4], [x1, y1]],
                 [[x3, y3], [x4, y4], [x1, y1], [x2, y2]], [[x4, y4], [x1, y1], [x2, y2], [x3, y3]]]
    dst_coordinate = [[xmin, ymin], [xmax, ymin], [xmax, ymax], [xmin, ymax]]
    force = 100000000.0
    force_flag = 0
    for i in range(4):
        temp_force = cal_line_length(combinate[i][0], dst_coordinate[0]) \
            + cal_line_length(combinate[i][1], dst_coordinate[1]) \
            + cal_line_length(combinate[i][2], dst_coordinate[2]) \
            + cal_line_length(combinate[i][3], dst_coordinate[3])
        if temp_force < force:
            force = temp_force
            force_flag = i
    if force_flag != 0:
        pass
        # print("choose one direction!")
    return np.array(combinate[force_flag]).reshape(8)

def poly2rbox_single(poly):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rrect:[x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)

    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])

    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) +
                    (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) +
                    (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

    angle = 0
    width = 0
    height = 0

    if edge1 > edge2:
        width = edge1
        height = edge2
        angle = np.arctan2(
            np.float(pt2[1] - pt1[1]), np.float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        width = edge2
        height = edge1
        angle = np.arctan2(
            np.float(pt4[1] - pt1[1]), np.float(pt4[0] - pt1[0]))

    if angle > np.pi*3/4:
        angle -= np.pi
    if angle < -np.pi/4:
        angle += np.pi

    x_ctr = np.float(pt1[0] + pt3[0]) / 2
    y_ctr = np.float(pt1[1] + pt3[1]) / 2
    rbox = np.array([x_ctr, y_ctr, width, height, angle])

    return rbox

def norm_angle(angle, range=[-np.pi / 4, np.pi]):
    return (angle - range[0]) % range[1] + range[0]


def poly2rbox_single_v2(poly):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rrect:[x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)

    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])

    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) +
                    (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) +
                    (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

    angle = 0
    width = 0
    height = 0

    if edge1 > edge2:
        width = edge1
        height = edge2
        angle = np.arctan2(
            np.float(pt2[1] - pt1[1]), np.float(pt2[0] - pt1[0]))
    elif edge2 >= edge1:
        width = edge2
        height = edge1
        angle = np.arctan2(
            np.float(pt4[1] - pt1[1]), np.float(pt4[0] - pt1[0]))

    # if angle > np.pi*3/4:
    #     angle -= np.pi
    # if angle < -np.pi/4:
    #     angle += np.pi
    angle = norm_angle(angle)

    x_ctr = np.float(pt1[0] + pt3[0]) / 2
    y_ctr = np.float(pt1[1] + pt3[1]) / 2
    
    return float(x_ctr), float(y_ctr), float(width), float(height), float(angle)


def poly2rbox_single_v3(poly):
    """
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    to
    rrect:[x_ctr,y_ctr,w,h,angle]
    """
    poly = np.array(poly[:8], dtype=np.float32)

    pt1 = (poly[0], poly[1])
    pt2 = (poly[2], poly[3])
    pt3 = (poly[4], poly[5])
    pt4 = (poly[6], poly[7])

    edge1 = np.sqrt((pt1[0] - pt2[0]) * (pt1[0] - pt2[0]) +
                    (pt1[1] - pt2[1]) * (pt1[1] - pt2[1]))
    edge2 = np.sqrt((pt2[0] - pt3[0]) * (pt2[0] - pt3[0]) +
                    (pt2[1] - pt3[1]) * (pt2[1] - pt3[1]))

    max_edge = max(edge1, edge2)
    min_edge = min(edge1, edge2)
    ratio = max_edge / min_edge
    # print(ratio)
    if ratio < 1.15:

        width = max_edge
        height = min_edge
        angle1 = np.arctan2(np.float(pt2[1] - pt1[1]), np.float(pt2[0] - pt1[0]))
        # elif edge2 >= edge1:
        angle2 = np.arctan2(np.float(pt4[1] - pt1[1]), np.float(pt4[0] - pt1[0]))

        angle1_norm = norm_angle(angle1)
        angle2_norm = norm_angle(angle2)
        # if abs(angle1_norm) > abs(angle2_norm):
        #     final_angle = angle2_norm
        # else:
        #     final_angle = angle1_norm
        if abs(angle1_norm) > abs(angle2_norm):
            final_angle = angle2_norm
        else:
            final_angle = angle1_norm

    else:
        final_angle = 0
        width = 0
        height = 0

        if edge1 > edge2:
            width = edge1
            height = edge2
            final_angle = np.arctan2(
                np.float(pt2[1] - pt1[1]), np.float(pt2[0] - pt1[0]))
        elif edge2 >= edge1:
            width = edge2
            height = edge1
            final_angle = np.arctan2(
                np.float(pt4[1] - pt1[1]), np.float(pt4[0] - pt1[0]))

        final_angle = norm_angle(final_angle)

    x_ctr = np.float(pt1[0] + pt3[0]) / 2
    y_ctr = np.float(pt1[1] + pt3[1]) / 2

    return float(x_ctr), float(y_ctr), float(width), float(height), float(final_angle)


def rbox2poly_single(rrect):
    """
    rrect:[x_ctr,y_ctr,w,h,angle]
    to
    poly:[x0,y0,x1,y1,x2,y2,x3,y3]
    """
    x_ctr, y_ctr, width, height, angle = rrect[:5]
    tl_x, tl_y, br_x, br_y = -width/2, -height/2, width/2, height/2
    rect = np.array([[tl_x, br_x, br_x, tl_x], [tl_y, tl_y, br_y, br_y]])
    R = np.array([[np.cos(angle), -np.sin(angle)],
                  [np.sin(angle), np.cos(angle)]])
    poly = R.dot(rect)
    x0, x1, x2, x3 = poly[0, :4] + x_ctr
    y0, y1, y2, y3 = poly[1, :4] + y_ctr
    poly = np.array([x0, y0, x1, y1, x2, y2, x3, y3], dtype=np.float32)
    poly = get_best_begin_point_single(poly)
    return poly

def convert2rbox(src_path):
    image_path = osp.join(src_path, 'images/')
    src_label_path = osp.join(src_path, 'labelTxt/')
    dst_label_path = osp.join(src_path, 'labelTxtRbox/')
    if not osp.exists(dst_label_path):
        os.mkdir(dst_label_path)

    image_list = os.listdir(image_path)
    image_list.sort()

    for image in image_list:
        img_name = osp.basename(image)
        print(img_name)
        ann_name = img_name.split('.')[0]+'.txt'
        lab_path = osp.join(src_label_path, ann_name)
        dst_path = osp.join(dst_label_path, ann_name)
        out_str = ''

        # import time
        # half the time used by poly2rbox
        with open(lab_path, 'r') as f:
            for ann_line in f.readlines():
                ann_line = ann_line.strip().split(' ')
                bbox = [np.float32(ann_line[i]) for i in range(8)]
                # 8 point to 5 point xywha
                x_ctr, y_ctr, width, height, angle = poly2rbox_single(bbox)
                class_name = ann_line[8]
                difficult = int(ann_line[9])

                out_str += "{} {} {} {} {} {} {}\n".format(str(x_ctr), str(
                    y_ctr), str(width), str(height), str(angle), class_name, difficult)
        with open(dst_path, 'w') as fdst:
            fdst.write(out_str)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('-p', '--path', type=str, required=True)
    args = parser.parse_args()
    convert2rbox(args.path)
