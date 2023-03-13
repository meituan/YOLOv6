import glob
import xml.etree.ElementTree as ET
from collections import Counter

import cv2
import numpy as np

from trans_tools import sortedEightPoints


def find_class_id_coarse(class_id_fine):
    if class_id_fine <= 11:
        class_id_coarse = 0
    elif class_id_fine >= 12 and class_id_fine <= 20:
        class_id_coarse = 1
    elif class_id_fine >= 21 and class_id_fine <= 30:
        class_id_coarse = 2
    elif class_id_fine >= 31 and class_id_fine <= 34:
        class_id_coarse = 3
    else:
        class_id_coarse = 4
    return class_id_coarse


def calculate_angel(point_a, point_b, point_c):
    ba = point_a - point_b
    bc = point_c - point_b
    ac = point_c - point_a
    cosine_angle = (np.linalg.norm(ba) ** 2 + np.linalg.norm(bc) ** 2 - np.linalg.norm(ac) ** 2) / (
        2 * np.linalg.norm(ba) * np.linalg.norm(bc)
    )
    pAngle = np.arccos(cosine_angle)
    angle = np.degrees(pAngle)
    angle = np.around(angle, 2)
    # angle = np.rint(angle)
    return angle


def calculate_theta(point_bottom, point_right, longSide, shortSide):
    a = point_bottom[1] - point_right[1]
    b = point_right[0] - point_bottom[0]
    c = calculate_distance(point_bottom, point_right)
    if b == 0:
        if c == longSide:
            theta = 90.0
        else:
            theta = 0.0

        if np.around(longSide, 2) == np.around(shortSide, 2):
            theta = 0.0
        return theta
    else:
        tan_angle = a / b
        angle = np.arctan(tan_angle)
        theta = np.degrees(angle)

        if np.around(longSide, 2) == np.around(shortSide, 2):
            theta = 90.0 - theta

        else:
            if c == longSide:
                theta = 180.0 - theta
            else:
                theta = 90.0 - theta
        # ! 边界条件 int[0, 180), 向下到179便于后面CSL取整
        if 179 < theta <= 180:
            theta = 179
        return theta


def calculate_center_point(poly):
    x_center = np.sum(poly[:, 0]) / 4.0
    y_center = np.sum(poly[:, 1]) / 4.0
    return x_center, y_center


def calculate_distance(point_a, point_b):
    return np.linalg.norm(point_a - point_b)


def calculate_square(long_side, shortside):
    return long_side * shortside


def calculate_longside_format(poly):
    x_center, y_center = calculate_center_point(poly)
    point_left = poly[-1]
    point_right = poly[1]
    point_bottom = poly[2]
    width = calculate_distance(point_right, point_bottom)
    height = calculate_distance(point_left, point_bottom)
    longSide = max(width, height)
    shortSide = min(width, height)
    theta = calculate_theta(point_bottom, point_right, longSide, shortSide)

    # if theta < 0 or theta > 180:
    #     print('hello')

    return ((x_center, y_center), (longSide, shortSide), theta)


# * 判断中心点溢出
def is_center_point_overflow(center_point, max_width, max_height):
    if center_point[0] <= max_width - 1 and center_point[1] <= max_height - 1:
        return False
    else:
        return True


# * 判断面积是否过小，
def is_square_deficiency(longSide, shortSide):
    if calculate_square(longSide, shortSide) <= square_threshold:
        return True
    else:
        return False


def sort_poly(poly):
    # ! 误差引起排序问题
    # * 做小数点三位数截断
    copy_poly = np.copy(poly)
    poly = np.around(poly, 3)

    y_order = np.argsort(poly[:, 1])

    top_index = y_order[0]
    bottom_index = y_order[-1]

    if poly[y_order[0], 1] == poly[y_order[1], 1]:
        x_order = np.argmin(poly[y_order[0:2], 0])
        top_index = y_order[x_order]

    if poly[y_order[-1], 1] == poly[y_order[-2], 1]:
        x_order = np.argmax(poly[y_order[-2:], 0])
        bottom_index = y_order[x_order + 2]

    leftAndRight_index = [0, 1, 2, 3]
    leftAndRight_index.remove(top_index)
    leftAndRight_index.remove(bottom_index)
    leftAndRight_order = np.argsort(poly[leftAndRight_index, 0])
    left_index = leftAndRight_index[leftAndRight_order[0]]
    right_index = leftAndRight_index[leftAndRight_order[1]]

    poly_order = [top_index, right_index, bottom_index, left_index]

    new_poly = copy_poly[poly_order]

    return new_poly


# opencv-python=4.5.3
def minAreaRect2longSideFormat(rectangle_inf):
    width = rectangle_inf[1][0]
    height = rectangle_inf[1][1]
    theta = rectangle_inf[-1]
    longSide = max(width, height)
    shortSide = min(width, height)
    if theta == 90:
        if longSide == width:
            pass
        else:
            theta = 0
        # * 正方形
        if np.around(longSide, 2) == np.around(shortSide, 2):
            theta = 0
    else:
        # * 正四边形 minAreaRect 会直接判定为左侧补角，符合要求 做长度截断判断即可
        if np.around(longSide, 2) == np.around(shortSide, 2):
            pass
        else:
            if longSide == width:
                pass
            else:
                theta += 90

    if 179 < theta <= 180:
        theta = 179

    return (rectangle_inf[0], (longSide, shortSide), theta)


def longSideFormat2minAreaRect(longSide_inf):
    longSide = longSide_inf[1][0]
    shortSide = longSide_inf[1][1]
    theta = longSide_inf[-1]
    width = longSide
    height = shortSide
    if theta == 0:
        width = shortSide
        height = longSide
        theta = 90
    # ! 有概率非全部 不影响绘图
    # elif theta == 90:
    #     width = longSide
    #     height = shortSide
    #     theta = 0
    else:
        # * 正四边形
        if np.around(longSide, 2) == np.around(shortSide, 2):
            width = longSide
            height = shortSide
            pass
        if theta > 90:
            width = shortSide
            height = longSide
            theta -= 90
        else:
            pass

    if theta >= 180:
        raise ValueError("theta >= 180")

    return (longSide_inf[0], (width, height), theta)


# for test
def plot_box(img_path_prefix, img_path, poly):
    img = cv2.imread(img_path_prefix + img_path)
    poly = np.int0(np.rint(poly))
    cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=[0, 255, 0], thickness=2)
    cv2.imshow("box", img)
    cv2.waitKey(0)


# for test
def plot_cv2_box(img_path_prefix, img_path, poly, rectangle_inf):
    img = cv2.imread(img_path_prefix + img_path)
    box = cv2.boxPoints(rectangle_inf)
    box = np.int0(box)
    poly = np.int0(np.rint(poly))

    cv2.drawContours(image=img, contours=[poly], contourIdx=-1, color=[0, 255, 0], thickness=2)

    cv2.drawContours(image=img, contours=[box], contourIdx=-1, color=[255, 0, 0], thickness=2)
    cv2.imshow("box", img)
    cv2.waitKey(0)


def plotLongSideFormatImage(img_path_prefix, img_path, label_annotations):
    """plot longSide format Image

    Args:
        img_path_prefix ([str]): [path_prefix]
        label_annotations ([list]): [[label], [label]...]
        label: [x_center y_center longSide shortSide theta class_id_fine class_id_coarse difficult_flag]
    """
    img = cv2.imread(img_path_prefix + img_path)

    for label in label_annotations:
        longSide_inf = ((label[0], label[1]), (label[2], label[3]), label[4])
        class_id_fine = label[5]
        class_id_coarse = label[6]
        rectangle_inf = longSideFormat2minAreaRect(longSide_inf)
        box = cv2.boxPoints(rectangle_inf)
        box = np.int0(box)

        cv2.drawContours(image=img, contours=[box], contourIdx=-1, color=[255, 0, 0], thickness=2)

    # cv2.imshow('box', img)
    # cv2.waitKey(0)
    cv2.imwrite("./copy_img/" + img_path, img)


if __name__ == "__main__":

    square_threshold = 32

    # classes = ['Boeing737', 'Boeing747', 'Boeing777', 'Boeing787', 'C919', 'A220', 'A321', 'A330', 'A350', 'ARJ21', 'other-airplane',
    #            'Passenger Ship', 'Motorboat', 'Fishing Boat', 'Tugboat', 'Engineering Ship', 'Liquid Cargo Ship', 'Dry Cargo Ship', 'Warship', 'other-ship',
    #            'Small Car', 'Bus', 'Cargo Truck', 'Dump Truck', 'Van', 'Trailer', 'Tractor', 'Excavator', 'Truck Tractor', 'other-vehicle',
    #            'Basketball Court', 'Tennis Court', 'Football Field', 'Baseball Field',
    #            'Intersection', 'Roundabout', 'Bridge']

    classes = [
        "airplane",
        "airport",
        "baseballfield",
        "basketballcourt",
        "bridge",
        "chimney",
        "dam",
        "Expressway-Service-area",
        "Expressway-toll-station",
        "golffield",
        "groundtrackfield",
        "harbor",
        "overpass",
        "ship",
        "stadium",
        "storagetank",
        "tenniscourt",
        "trainstation",
        "vehicle",
        "windmill",
    ]
    label_list = glob.glob("./Annotations/Oriented Bounding Boxes/*.xml")

    # part1_path = './part1.txt'
    # part2_path = './part2.txt'

    # images_size = dict()

    # with open(part1_path) as f:
    #     for line in f.readlines():
    #         inf = line.split(' ')
    #         images_size.update({inf[0]: [int(inf[1]), int(inf[2])]})

    # with open(part2_path) as f:
    #     for line in f.readlines():
    #         inf = line.split(' ')
    #         images_size.update({inf[0]: [int(inf[1]), int(inf[2])]})

    final_list = []
    # image_path_prefix = 'E:/FAIRIM/images/'

    count_1 = 0
    count_2 = 0
    count_3 = 0
    count_4 = 0
    count_5 = 0
    count_all = 0
    count_right = 0

    choose_list = []
    for label in label_list:

        # image_path = label.split('\\')[-1].split('.')[0] + '.tif'

        tree = ET.parse(label)
        # objects_tag = tree.find('objects')
        objects = tree.findall("object")

        label_annotations = []

        flag_choose = 0
        for i in objects:
            label_annotation = []
            # object_type = i.find('type').text
            object_name = i.find("name").text
            # points = i.find('points')

            x1 = eval(i.find("robndbox").find("x_left_top").text)
            y1 = eval(i.find("robndbox").find("y_left_top").text)
            x2 = eval(i.find("robndbox").find("x_right_top").text)
            y2 = eval(i.find("robndbox").find("y_right_top").text)
            x3 = eval(i.find("robndbox").find("x_left_bottom").text)
            y3 = eval(i.find("robndbox").find("y_left_bottom").text)
            x4 = eval(i.find("robndbox").find("x_right_bottom").text)
            y4 = eval(i.find("robndbox").find("y_right_bottom").text)

            box = [x1, y1, x2, y2, x3, y3, x4, y4]
            sorted_box = sortedEightPoints(box)

            point_x = [x1, x2, x3, x4]
            point_y = [y1, y2, y3, y4]

            # for point in points:
            #     co = point.text
            #     co = co.split(',')
            #     point_x.append(eval(co[0]))
            #     point_y.append(eval(co[1]))

            count_all += 1

            # ! 排除: 五点情况
            # if point_x[-1] != point_x[0] or point_y[-1] != point_y[0]:
            #     count_1 += 1
            #     continue

            # point_x.pop()
            # point_y.pop()

            poly = np.array(
                [
                    [point_x[0], point_y[0]],
                    [point_x[1], point_y[1]],
                    [point_x[2], point_y[2]],
                    [point_x[3], point_y[3]],
                ]
            )

            poly = np.rint(poly).astype("int")

            poly_sorted = np.array(
                [
                    [sorted_box[0], sorted_box[1]],
                    [sorted_box[2], sorted_box[3]],
                    [sorted_box[4], sorted_box[5]],
                    [sorted_box[6], sorted_box[7]],
                ]
            )
            poly_sorted = np.rint(poly_sorted).astype("int")

            # ! 排除: 同点 \ 同线 \ 三角形 三种情况
            dict_point_x = dict(Counter(poly[:, 0]))
            dict_point_y = dict(Counter(poly[:, 1]))

            flag = 0

            for key, value in dict_point_x.items():
                if value == 3 or value == 4:
                    flag = 1
                    break
            for key, value in dict_point_y.items():
                if value == 3 or value == 4:
                    flag = 1
                    break

            if flag == 1:
                count_2 += 1
                continue

            # * first step 角点排序， 找到xmin ymin的点， 定为top点

            # y_order = np.argsort(poly[:, 1])

            # top_index = y_order[0]
            # bottom_index = y_order[-2]

            # if poly[y_order[0], 1] == poly[y_order[1], 1]:
            #     x_order = np.argmin(poly[y_order[0:2], 0])
            #     top_index = y_order[x_order]

            # if poly[y_order[2], 1] == poly[y_order[3], 1]:
            #     x_order = np.argmax(poly[y_order[3:4], 0])
            #     bottom_index = y_order[x_order]

            # # bottom_index = (top_index + 2) % 4

            # leftAndRight_index = [0, 1, 2, 3]
            # leftAndRight_index.remove(top_index)
            # leftAndRight_index.remove(bottom_index)
            # leftAndRight_order = np.argsort(poly[leftAndRight_index, 0])
            # left_index = leftAndRight_index[leftAndRight_order[0]]
            # right_index = leftAndRight_index[leftAndRight_order[1]]

            # poly_order = [top_index,
            #               right_index,
            #               bottom_index,
            #               left_index]

            poly = poly_sorted

            point_1 = poly[[0]]
            point_2 = poly[[1]]
            point_3 = poly[[2]]
            point_4 = poly[[3]]

            angle_1 = calculate_angel(point_4, point_1, point_2)
            angle_2 = calculate_angel(point_1, point_2, point_3)
            angle_3 = calculate_angel(point_2, point_3, point_4)
            angle_4 = calculate_angel(point_1, point_4, point_3)

            # print(angle_1 + angle_2 + angle_3 + angle_4)

            angle_list = [angle_1, angle_2, angle_3, angle_4]

            angle_list = dict(Counter(angle_list))

            print(angle_list)

            # ! 排除: 凹四边形
            if angle_1 >= 180 or angle_2 >= 180 or angle_3 >= 180 or angle_4 >= 180:
                count_3 += 0
                continue

            # max_height = images_size[image_path][0]
            # max_width = images_size[image_path][1]
            max_height = 800
            max_width = 800

            # * 先判断正矩形， 再判断越界

            regular_rectangle_flag = 0

            # * 判断正矩形
            # print(angle_list)
            if len(angle_list.keys()) < 2:
                # angle 保留两位小数做截断，正矩形可以直接做近似计算long_side short_side theta
                # 3个直角边的情况出现在小数保留精度上，可以随角度4一同做近似计算
                # * 可以考虑直接做np.rint()
                # * right angle 数量为4或者3合并为正矩形直接考虑，其他情况
                if angle_list[90] == 4 or angle_list[90] == 3:
                    regular_rectangle_flag = 1

                    if (
                        np.sum(poly[:, 0] >= max_width - 1) >= 1
                        or np.sum(poly[:, 1] >= max_height - 1) >= 1
                        or np.sum(poly[:, 0] <= 0) >= 1
                        or np.sum(poly[:, 1] <= 0) >= 1
                    ):
                        difficult_flag = 2
                    else:
                        difficult_flag = 0

                    longSide_inf = calculate_longside_format(poly)
                    new_rectangle_inf = longSideFormat2minAreaRect(longSide_inf)
                    x_center = longSide_inf[0][0]
                    y_center = longSide_inf[0][1]
                    longSide = longSide_inf[1][0]
                    shortSide = longSide_inf[1][1]
                    theta = longSide_inf[-1]

                    if is_center_point_overflow((x_center, y_center), max_width, max_height):
                        # ! 排除：溢出
                        count_4 += 1
                        continue
                    else:
                        if is_square_deficiency(longSide, shortSide):
                            count_5 += 1
                            continue
                        else:
                            count_right += 1

                            class_id_fine = classes.index(object_name)
                            # class_id_coarse = find_class_id_coarse(
                            #     class_id_fine)
                            # * 顺序
                            # * x_center y_center longSide shortSide theta class_id_fine class_id_coarse difficult_flag
                            label_annotation.append(x_center)
                            label_annotation.append(y_center)
                            label_annotation.append(longSide)
                            label_annotation.append(shortSide)
                            label_annotation.append(theta)
                            label_annotation.append(class_id_fine)
                            # label_annotation.append(class_id_coarse)
                            label_annotation.append(difficult_flag)
                            # label_annotation = ','.join(label_annotation)
                            label_annotations.append(label_annotation)

                            xmin = np.min(poly[:, 0])
                            xmax = np.max(poly[:, 0])
                            ymin = np.min(poly[:, 1])
                            ymax = np.max(poly[:, 1])
                            from icecream import ic

                            # ic(xmin, xmax, ymin, ymax)
                            if xmax - xmin >= 320 or ymax - ymin >= 320:
                                flag_choose = 1
                            continue

            # * 判断普通四边形越界情况
            if regular_rectangle_flag == 0:

                if (
                    np.sum(poly[:, 0] >= max_width - 1) >= 1
                    or np.sum(poly[:, 1] >= max_height - 1) >= 1
                    or np.sum(poly[:, 0] <= 0) >= 1
                    or np.sum(poly[:, 1] <= 0) >= 1
                ):
                    difficult_flag = 2
                else:
                    difficult_flag = 0

                rectangle_inf = cv2.minAreaRect(poly)
                longSide_inf = minAreaRect2longSideFormat(rectangle_inf)

                x_center = longSide_inf[0][0]
                y_center = longSide_inf[0][1]
                longSide = longSide_inf[1][0]
                shortSide = longSide_inf[1][1]
                theta = longSide_inf[-1]

                if is_center_point_overflow((x_center, y_center), max_width, max_height):
                    # ! 排除：溢出
                    count_4 += 1
                    continue
                else:
                    if is_square_deficiency(longSide, shortSide):
                        # ! 排除: 面积过小
                        count_5 += 1
                        continue
                    else:
                        count_right += 1

                        class_id_fine = classes.index(object_name)
                        # class_id_coarse = find_class_id_coarse(class_id_fine)
                        # * 顺序
                        # * x_center y_center longSide shortSide theta class_id_fine class_id_coarse difficult_flag
                        label_annotation.append(x_center)
                        label_annotation.append(y_center)
                        label_annotation.append(longSide)
                        label_annotation.append(shortSide)
                        label_annotation.append(theta)
                        label_annotation.append(class_id_fine)
                        # label_annotation.append(class_id_coarse)
                        label_annotation.append(difficult_flag)
                        # label_annotation = ','.join(label_annotation)
                        label_annotations.append(label_annotation)

                        xmin = np.min(poly[:, 0])
                        xmax = np.max(poly[:, 0])
                        ymin = np.min(poly[:, 1])
                        ymax = np.max(poly[:, 1])
                        from icecream import ic

                        # ic(xmin, xmax, ymin, ymax)
                        if xmax - xmin >= 320 or ymax - ymin >= 320:
                            flag_choose = 1
                            # ic(label, class_id_fine, object_name)
                            # assert 1 == 0
                            # assert 1 == 0
                        continue

        if flag_choose == 1:
            pass
            # choose_list.append(label + '.\n')
            # from pathlib import Path
            # path = Path(label).with_suffix('.tif')
            # plotLongSideFormatImage(image_path_prefix, path.name, label_annotations)
        # label_annotations.insert(0, image_path)
        # label_annotations = ' '.join(label_annotations) + '\n'
        # final_list.append(label_annotations)

    # ic(len(choose_list))

    # with open('./label_list.txt', 'w+') as f:
    #     f.writelines(final_list)

    # with open('./demo2.txt', 'w+') as f:
    #     f.writelines(choose_list)

    print("总目标: ", count_all)
    print("有效目标: ", count_right)
    print("五点情况: ", count_1)
    print("同点,同线,三角情况: ", count_2)
    print("凹四边形情况: ", count_3)
    print("中心坐标溢出情况: ", count_4)
    print("面积过小情况: ", count_5)
