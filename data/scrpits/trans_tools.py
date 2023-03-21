from collections import Counter
from pathlib import Path
from typing import Counter, List, Tuple

import cv2
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Polygon
from pendulum import now
from rich import print


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


def minAreaRect2longSideFormat(
    rectangle_inf: Tuple[Tuple[float, float], Tuple[float, float], float]
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
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
        # theta = 179
        ...
    if theta < 0 or theta >= 180:
        # raise ValueError("theta < 0 or theta >= 180")
        pass

    return (rectangle_inf[0], (longSide, shortSide), theta)


def longSideFormat2minAreaRect(
    longSide_inf: Tuple[Tuple[float, float], Tuple[float, float], float]
) -> Tuple[Tuple[float, float], Tuple[float, float], float]:
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


def plotLongSideFormatImage(img_path_prefix: str, img_path: str, label_annotations: List[List[float]]) -> None:
    """plot longSide format Image

    Args:
        img_path_prefix ([str]): [path_prefix]
        label_annotations ([list]): [[label], [label]...]
        label: [x_center y_center longSide shortSide theta class_id_fine class_id_coarse difficult_flag]
    """
    img = cv2.imread(img_path_prefix + img_path)

    for label in label_annotations:
        longSide_inf = ((label[0], label[1]), (label[2], label[3]), label[4])
        rectangle_inf = longSideFormat2minAreaRect(longSide_inf)
        box = cv2.boxPoints(rectangle_inf)
        box = np.int0(box)

        cv2.drawContours(image=img, contours=[box], contourIdx=-1, color=[255, 0, 0], thickness=2)

    cv2.imshow("box", img)
    cv2.waitKey(0)


def showLabels(image_path: Path, objects):
    img = cv2.imread(str(image_path))

    plt.imshow(img)
    plt.axis("off")

    ax = plt.gca()
    ax.set_autoscale_on(False)
    polygons = []
    color = []
    # circles = []
    # r = 5
    for object in objects:
        c = (np.random.random((1, 3)) * 0.6 + 0.4).tolist()[0]

        for index, content in enumerate(object):
            if isinstance(content, str):
                object[index] = eval(content)

        longSide_inf = ((object[4], object[5]), (object[6], object[7]), object[-1])
        rectangle_inf = longSideFormat2minAreaRect(longSide_inf)
        poly = cv2.boxPoints(rectangle_inf)
        # poly = np.int0(poly)

        polygons.append(Polygon(poly))
        color.append(c)
        # point = poly[0]
        # circle = Circle((point[0], point[1]), r)
        # circles.append(circle)

    p = PatchCollection(polygons, facecolors=color, linewidths=0, alpha=0.4)
    ax.add_collection(p)
    p = PatchCollection(polygons, facecolors="none", edgecolors=color, linewidths=2)
    ax.add_collection(p)
    # p = PatchCollection(circles, facecolors='red')
    # ax.add_collection(p)

    plt.show()


def sortedEightPoints(box: list[int]):
    """
        box: [x1, y1, x2, y2, x3, y3, x4, y4]
    """
    # * corners 八点坐标
    corners = np.array(
        [
            [int(float(box[0])), int(float(box[1]))],
            [int(float(box[2])), int(float(box[3]))],
            [int(float(box[4])), int(float(box[5]))],
            [int(float(box[6])), int(float(box[7]))],
        ]
    )

    # NOTE 先按 y 轴坐标升序排序（从小到大）
    corners_sort: np.ndarray = corners[np.argsort(corners[:, 1])]
    # NOTE 前两点按 x 轴坐标升序排序，后两点按x轴坐标降序排序，保持顺时针顺序
    corners_sorted: np.ndarray = corners_sort[
        np.append(np.argsort(corners_sort[0:2, 0]), np.argsort(corners_sort[2:4, 0])[::-1] + 2)
    ]

    # NOTE 前两点x轴坐标相同, 一边垂直
    if corners_sorted[0, 0] == corners_sorted[1, 0]:
        temp_max = corners_sorted[0, 1]
        # NOTE 垂直边在左侧
        if corners_sorted[0, 0] < min(corners_sorted[2, 0], corners_sorted[3, 0]):
            if temp_max > corners_sorted[1, 1]:
                pass
            else:
                corners_sorted = corners_sorted[[1, 0, 2, 3]]
        # NOTE 垂直边在右侧
        else:
            if temp_max > corners_sorted[1, 1]:
                corners_sorted = corners_sorted[[1, 0, 2, 3]]
            else:
                pass

    # NOTE 后两点x轴坐标相同, 一边垂直
    if corners_sorted[2, 0] == corners_sorted[3, 0]:
        temp_max = corners_sorted[2, 1]
        # NOTE 垂直边在左侧
        if corners_sorted[2, 0] < min(corners_sorted[0, 0], corners_sorted[1, 0]):
            if temp_max > corners_sorted[3, 1]:
                pass
            else:
                corners_sorted = corners_sorted[[0, 1, 3, 2]]
        # NOTE 垂直边在右侧
        else:
            if temp_max > corners_sorted[3, 1]:
                corners_sorted = corners_sorted[[0, 1, 3, 2]]
            else:
                pass

    corners_x_max: int = corners[np.argmax(corners[:, 0])][0]
    corners_y_max: int = corners[np.argmax(corners[:, 1])][1]
    corners_x_min: int = corners[np.argmin(corners[:, 0])][0]
    corners_y_min: int = corners[np.argmin(corners[:, 1])][1]
    corners_fixed: np.ndarray = corners_sorted

    # * 16种情况排序
    if (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        # NOTE 包含正矩形 0 度角特殊情况，划分到这一类别
        # 不做变动
        # https://markdown-1309012516.cos.ap-beijing.myqcloud.com/2022_6_13_15_53_35_1655106815197.png
        pass
    # @2
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed = corners_fixed[[1, 2, 3, 0]]
    # @2
    elif (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed[0, 1] = corners_y_min
        corners_fixed[2, 1] = corners_y_max
    # @3
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        corners_fixed[1, 0] = corners_x_max
        corners_fixed[3, 0] = corners_x_min
    # @4
    elif (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        corners_fixed[2, 1] = corners_y_max
    # @5
    elif (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed[0, 1] = corners_y_min
    # @6
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        corners_fixed[1, 1] = corners_y_min
        corners_fixed = corners_fixed[[1, 2, 3, 0]]
    # @7
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed[3, 1] = corners_y_max
        corners_fixed = corners_fixed[[1, 2, 3, 0]]
    # @8
    elif (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        corners_fixed[3, 0] = corners_x_min
    # @9
    elif (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed[0, 1] = corners_y_min
        corners_fixed[3, 0] = corners_x_min
    # @10
    elif (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        corners_fixed[2, 1] = corners_y_max
        corners_fixed[3, 0] = corners_x_min
    # @11
    elif (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed[2, 0] = corners_x_max
        corners_fixed = corners_fixed[[1, 2, 3, 0]]
    # @12
    # X_max Xmin Y_max Y_min => X3 X4 X3 X1 @ 13
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        corners_fixed[1, 0] = corners_x_max
    # @13
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed[0, 1] = corners_y_min
        corners_fixed[1, 0] = corners_x_max
    # @14
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        corners_fixed[1, 0] = corners_x_max
        corners_fixed[2, 1] = corners_y_max
    # @15
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed[0, 0] = corners_x_min
        corners_fixed = corners_fixed[[1, 2, 3, 0]]

    # [x1, y1, x2, y2, x3, y3, x4, y4]
    corners_info: list[int] = [x for corner in corners_fixed.tolist() for x in corner]
    # 面积比

    # ratio: float = cv2.contourArea(corners_fixed) / (
    #     float(corners_x_max - corners_x_min) * float(corners_y_max - corners_y_min)
    # )
    # points = np.array(
    #     [
    #         [int(corners_fixed[0][0]), int(str(corners_fixed[0][1]))],
    #         [int(corners_fixed[1][0]), int(corners_fixed[1][1])],
    #         [int(corners_fixed[2][0]), int(corners_fixed[2][1])],
    #         [int(corners_fixed[3][0]), int(corners_fixed[3][1])],
    #     ]
    # )

    # rect = cv2.minAreaRect(points)
    # longSide_inf = minAreaRect2longSideFormat(rect)
    # angle: float = longSide_inf[-1]
    # return [corners_x_min, corners_y_min, corners_x_max, corners_y_max], corners_info, ratio, angle
    return corners_info


def eightPointsToFiveAugments(box: list[int], img_height, img_width, min_area):
    error_label_flag = False
    longSide_inf = ((0, 0), (0, 0), 0)
    corners_info = [0, 0, 0, 0, 0, 0, 0, 0]
    ratio = 0
    info = ""
    # NOTE 负坐标截断，先截断再求外接矩和后截断再求外接矩误差不大
    for i, value in enumerate(box):
        if value < 0:
            box[i] = 0

    # * corners 八点坐标
    corners = np.array(
        [
            [int(float(box[0])), int(float(box[1]))],
            [int(float(box[2])), int(float(box[3]))],
            [int(float(box[4])), int(float(box[5]))],
            [int(float(box[6])), int(float(box[7]))],
        ]
    )

    # NOTE 同点/同线/三角情况
    dict_point_x = dict(Counter(corners[:, 0]))
    dict_point_y = dict(Counter(corners[:, 1]))

    # 检查 x 坐标
    for key, value in dict_point_x.items():
        if value == 3 or value == 4:
            error_label_flag = True
            info = "同点/同线/三角情况"
            return longSide_inf, corners_info, ratio, error_label_flag, info
    # 检查 y 坐标
    for key, value in dict_point_y.items():
        if value == 3 or value == 4:
            error_label_flag = True
            info = "同点/同线/三角情况"
            return longSide_inf, corners_info, ratio, error_label_flag, info

    # NOTE 先按 y 轴坐标升序排序（从小到大）
    corners_sort: np.ndarray = corners[np.argsort(corners[:, 1])]
    # NOTE 前两点按 x 轴坐标升序排序，后两点按x轴坐标降序排序，保持顺时针顺序
    corners_sorted: np.ndarray = corners_sort[
        # The above code is creating a list of coordinates for a box.
        np.append(np.argsort(corners_sort[0:2, 0]), np.argsort(corners_sort[2:4, 0])[::-1] + 2)
    ]

    # # NOTE 向量方向判断排序
    # vector_0 = corners_sorted[0] - corners_sorted[1]
    # vector_1 = corners_sorted[3] - corners_sorted[2]

    # # 左朝向
    # if corners_sorted[0, 0] < corners_sorted[3, 0]:
    #     if vector_1[1] < 0:
    #        corners_sorted = corners_sorted[[0, 1, 3, 2]]
    # # 右朝向
    # else:
    #     if vector_1[1] > 0:
    #            corners_sorted = corners_sorted[[0, 1, 3, 2]]

    # NOTE 前两点x轴坐标相同, 一边垂直
    if corners_sorted[0, 0] == corners_sorted[1, 0]:
        temp_max = corners_sorted[0, 1]
        # NOTE 垂直边在左侧
        if corners_sorted[0, 0] < min(corners_sorted[2, 0], corners_sorted[3, 0]):
            if temp_max > corners_sorted[1, 1]:
                pass
            else:
                corners_sorted = corners_sorted[[1, 0, 2, 3]]
        # NOTE 垂直边在右侧
        else:
            if temp_max > corners_sorted[1, 1]:
                corners_sorted = corners_sorted[[1, 0, 2, 3]]
            else:
                pass

    # NOTE 后两点x轴坐标相同, 一边垂直
    if corners_sorted[2, 0] == corners_sorted[3, 0]:
        temp_max = corners_sorted[2, 1]
        # NOTE 垂直边在左侧
        if corners_sorted[2, 0] < min(corners_sorted[0, 0], corners_sorted[1, 0]):
            if temp_max > corners_sorted[3, 1]:
                pass
            else:
                corners_sorted = corners_sorted[[0, 1, 3, 2]]
        # NOTE 垂直边在右侧
        else:
            if temp_max > corners_sorted[3, 1]:
                corners_sorted = corners_sorted[[0, 1, 3, 2]]
            else:
                pass

    point_0 = corners_sorted[[0]]
    point_1 = corners_sorted[[1]]
    point_2 = corners_sorted[[2]]
    point_3 = corners_sorted[[3]]

    angle_0 = calculate_angel(point_3, point_0, point_1)
    angle_1 = calculate_angel(point_0, point_1, point_2)
    angle_2 = calculate_angel(point_1, point_2, point_3)
    angle_3 = calculate_angel(point_2, point_3, point_0)
    angle_sum = angle_0 + angle_1 + angle_2 + angle_3
    angle_list = [angle_0, angle_1, angle_2, angle_3]

    # NOTE 溢出 第一次check
    if 358 <= angle_sum <= 362:
        pass
    else:
        if max(angle_list) == angle_0:
            # 没遇到这种情况
            new_corners_sorted = corners_sorted[[1, 0, 3, 2]]

        elif max(angle_list) == angle_1:
            new_corners_sorted = corners_sorted[[1, 0, 2, 3]]
            # 移位
            # new_corners_sorted = new_corners_sorted[[1, 2, 3, 0]]
            if new_corners_sorted[0, 1] > new_corners_sorted[1, 1]:
                new_corners_sorted = new_corners_sorted[[1, 2, 3, 0]]
            else:
                new_corners_sorted = new_corners_sorted[[0, 3, 2, 1]]

        elif max(angle_list) == angle_2:
            new_corners_sorted = corners_sorted[[0, 1, 3, 2]]

        elif max(angle_list) == angle_3:
            new_corners_sorted = corners_sorted[[0, 1, 3, 2]]
            # 移位
            if new_corners_sorted[0, 1] > new_corners_sorted[1, 1]:
                new_corners_sorted = new_corners_sorted[[1, 2, 3, 0]]
            else:
                new_corners_sorted = new_corners_sorted[[0, 3, 2, 1]]
        else:
            return ((0, 0), (0, 0), 0), 0, 0, 0, 0
            # print("hello world")
            # pass
            # new_corners_sorted = corners_sorted


        # NOTE 第二次check
        point_0 = new_corners_sorted[[0]]
        point_1 = new_corners_sorted[[1]]
        point_2 = new_corners_sorted[[2]]
        point_3 = new_corners_sorted[[3]]

        angle_0 = calculate_angel(point_3, point_0, point_1)
        angle_1 = calculate_angel(point_0, point_1, point_2)
        angle_2 = calculate_angel(point_1, point_2, point_3)
        angle_3 = calculate_angel(point_2, point_3, point_0)
        angle_sum = angle_0 + angle_1 + angle_2 + angle_3
        angle_list = [angle_0, angle_1, angle_2, angle_3]

        if 358 <= angle_sum <= 362:
            corners_sorted = new_corners_sorted
        else:
            # 大概率是凹四边形，角点排序没问题
            # corners_sorted = new_corners_sorted
            # t1 = (corners_sorted[3, 0] - corners_sorted[0, 0]) * (corners_sorted[0, 1] - corners_sorted[2, 1]) - (
            #     corners_sorted[3, 1] - corners_sorted[0, 1]
            # ) * (corners_sorted[0, 0] - corners_sorted[2, 0])
            # t2 = (corners_sorted[0, 0] - corners_sorted[1, 0]) * (corners_sorted[3, 1] - corners_sorted[1, 1]) - (
            #     corners_sorted[0, 1] - corners_sorted[1, 1]
            # ) * (corners_sorted[3, 0] - corners_sorted[1, 0])
            # t3 = (corners_sorted[2, 0] - corners_sorted[1, 0]) * (corners_sorted[3, 1] - corners_sorted[1, 1]) - (
            #     corners_sorted[2, 1] - corners_sorted[1, 1]
            # ) * (corners_sorted[3, 0] - corners_sorted[1, 0])
            # t4 = (corners_sorted[1, 0] - corners_sorted[0, 0]) * (corners_sorted[0, 1] - corners_sorted[2, 1]) - (
            #     corners_sorted[1, 1] - corners_sorted[0, 1]
            # ) * (corners_sorted[0, 0] - corners_sorted[2, 0])

            # if t2 * t3 * t1 * t4 >= 1e-6:
            #     # print("[bold blue]凹四边形")
            #     pass
            # else:
            # print("凸四边形")
            # print("debug")
            # print(box)
            # print(corners_sorted)
            # print(angle_0, angle_1, angle_2, angle_3, angle_sum)
            # error_label_flag = True
            # info = "内角之和溢出"
            # return longSide_inf, corners_info, ratio, error_label_flag, info

            # 凹四边形直接丢进去判断
            print('[bold blue]凹四边形')


    # # NOTE 凹四边形情况 / 排序bug
    # if angle_1 >= 180 or angle_2 >= 180 or angle_3 >= 180 or angle_4 >= 180:
    #     error_label_flag = True
    #     info = "凹四边形"
    #     return longSide_inf, corners_info, ratio, error_label_flag, info

    corners_x_max: int = corners[np.argmax(corners[:, 0])][0]
    corners_y_max: int = corners[np.argmax(corners[:, 1])][1]
    corners_x_min: int = corners[np.argmin(corners[:, 0])][0]
    corners_y_min: int = corners[np.argmin(corners[:, 1])][1]
    corners_fixed: np.ndarray = corners_sorted

    # * 16种情况排序和调整
    # @1
    if (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        # NOTE 包含正矩形 0 度角特殊情况，划分到这一类别
        # 不做变动
        # https://markdown-1309012516.cos.ap-beijing.myqcloud.com/2022_6_13_15_53_35_1655106815197.png
        pass
    # @2
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed = corners_fixed[[1, 2, 3, 0]]
    # @3
    elif (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed[0, 1] = corners_y_min
        corners_fixed[2, 1] = corners_y_max
    # @4
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        corners_fixed[1, 0] = corners_x_max
        corners_fixed[3, 0] = corners_x_min
    # @5
    elif (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        corners_fixed[2, 1] = corners_y_max
    # @6
    elif (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed[0, 1] = corners_y_min
    # @7
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        corners_fixed[1, 1] = corners_y_min
        corners_fixed = corners_fixed[[1, 2, 3, 0]]
    # @8
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed[3, 1] = corners_y_max
        corners_fixed = corners_fixed[[1, 2, 3, 0]]
    # @9
    elif (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        corners_fixed[3, 0] = corners_x_min
    # @9-1 / 第九种第一种变种
    # box = [33.0, 202.0, 41.0, 213.0, 1.0, 247.0, 2.0, 229.0]
    elif (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[2, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        corners_fixed[3, 0] = corners_x_min
    # @10
    elif (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed[0, 1] = corners_y_min
        corners_fixed[3, 0] = corners_x_min
    # @11
    elif (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        corners_fixed[2, 1] = corners_y_max
        corners_fixed[3, 0] = corners_x_min
    # @12
    elif (
        corners_x_max == corners_sorted[1, 0]
        and corners_x_min == corners_sorted[0, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed[2, 0] = corners_x_max
        corners_fixed = corners_fixed[[1, 2, 3, 0]]
    # @13
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        corners_fixed[1, 0] = corners_x_max
    # @14
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[2, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed[0, 1] = corners_y_min
        corners_fixed[1, 0] = corners_x_max
    # @15
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[0, 1]
    ):
        corners_fixed[1, 0] = corners_x_max
        corners_fixed[2, 1] = corners_y_max
    # @16
    elif (
        corners_x_max == corners_sorted[2, 0]
        and corners_x_min == corners_sorted[3, 0]
        and corners_y_max == corners_sorted[3, 1]
        and corners_y_min == corners_sorted[1, 1]
    ):
        corners_fixed[0, 0] = corners_x_min
        corners_fixed = corners_fixed[[1, 2, 3, 0]]
    else:
        print(box)
        print(corners_sorted)
        print("[blod red]buggggg!!!!")

    points = np.array(
        [
            [int(corners_fixed[0][0]), int(corners_fixed[0][1])],
            [int(corners_fixed[1][0]), int(corners_fixed[1][1])],
            [int(corners_fixed[2][0]), int(corners_fixed[2][1])],
            [int(corners_fixed[3][0]), int(corners_fixed[3][1])],
        ]
    )

    if not error_label_flag:
        rect = cv2.minAreaRect(points)
        longSide_inf = minAreaRect2longSideFormat(rect)

        x_center = longSide_inf[0][0]
        y_center = longSide_inf[0][1]
        longside = longSide_inf[1][0]
        shortside = longSide_inf[1][1]

        theta = longSide_inf[-1]
        area = longside * shortside

        if theta < 0 or theta >= 180:
            print("同线", theta)
            error_label_flag = True
            info = "cv2求角度值溢出, 同线情况"
            return longSide_inf, corners_info, ratio, error_label_flag, info

        # NOTE 中心点溢出
        if x_center >= img_width - 1 or y_center >= img_height - 1:
            error_label_flag = True
            info = "中心点溢出"
            print(longSide_inf)
            return longSide_inf, corners_info, ratio, error_label_flag, info

        try:
            ratio: float = area / (float(corners_x_max - corners_x_min) * float(corners_y_max - corners_y_min))
            if area <= min_area:
                error_label_flag = True
                info = "面积小于最小面积"
                return longSide_inf, corners_info, ratio, error_label_flag, info

        except ZeroDivisionError:
            ratio = 0
            error_label_flag = True
            info = "分母为0, 同线/同点/三角情况"
            return longSide_inf, corners_info, ratio, error_label_flag, info
    else:
        longSide_inf = ((0, 0), (0, 0), 0)
        ratio = 0

    # [x1, y1, x2, y2, x3, y3, x4, y4]
    corners_info: list[int] = [x for corner in corners_fixed.tolist() for x in corner]

    # for i, value in enumerate(corners_info):
    #     if value < 0:
    #         corners_info[i] = 0

    return longSide_inf, corners_info, ratio, error_label_flag, info


if __name__ == "__main__":
    # box = [33.0, 202.0, 41.0, 213.0, 1.0, 247.0, 2.0, 229.0]
    # box = [228.0, 147.0, 311.0, 135.0, 324.0, 219.0, 239.0, 233.0]
    # 平行四边形
    # box = [573.0, 540.0, 573.0, 516.0, 640.0, 518.0, 640.0, 544.0]
    # box = [461.0, 539.0, 462.0, 512.0, 538.0, 513.0, 537.0, 538.0]

    # 排序 bug1, 2/3角换顺序
    # box = [33.0, 202.0, 41.0, 213.0, 1.0, 247.0, 2.0, 229.0]
    # 排序 bug2, 2角
    # box = [202.0, 1079.0, 189.0, 1098.0, 87.0, 1045.0, 86.0, 1015.0]
    # 分类 bug1
    # box = [33.0, 202.0, 41.0, 213.0, 1.0, 247.0, 2.0, 229.0]

    # box = [736.0, 573.0, 738.0, 559.0, 1166.0, 575.0, 1167.0, 588.0]
    # box = [201.0, 639.0, 154.0, 640.0, 87.0, 605.0, 86.0, 575.0]
    # box = [735.0, 1023.0, 603.0, 1023.0, 417.0, 1021.0, 201.0, 871.0]
    # box = [1397.0, 1599.0, 1256.0, 1599.0, 1079.0, 1597.0, 863.0, 1447.0]
    # box = [1310.0, 1599.0, 1169.0, 1599.0, 992.0, 1597.0, 776.0, 1447.0]

    # 凹四边形
    # box = [1322.0, 296.0, 1320.0, 315.0, 1150.0, 340.0, 1147.0, 318.0]
    box = [8.0, 132.0, 25.0, 139.0, 8.0, 176.0, 7.0, 131.0]
    # box = [731.0, 3.0, 763.0, 1.0, 793.0, 13.0, 794.0, 34.0]
    print(eightPointsToFiveAugments(box, 800, 800, 32))
