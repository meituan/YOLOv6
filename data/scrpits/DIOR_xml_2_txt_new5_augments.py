import xml.etree.ElementTree as ET
from pathlib import Path

import numpy as np
from icecream import ic
from rich import print

from trans_tools import (eightPointsToFiveAugments, longSideFormat2minAreaRect,
                         minAreaRect2longSideFormat, showLabels,
                         sortedEightPoints)

# classes_info = [
#     "Airplane",
#     "Airport",
#     "Baseball field",
#     "Basketball court",
#     "Bridge",
#     "Chimney",
#     "Dam",
#     "Expressway service area",
#     "Expressway toll station",
#     "Golf course",
#     "Ground track field",
#     "Harbor",
#     "Overpass",
#     "Ship",
#     "Stadium",
#     "Storage tank",
#     "Tennis court",
#     "Train station",
#     "Vehicle",
#     "Wind mill",
# ]

classes_info = [
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

ic(len(classes_info))

# Note opencv-python=4.5.3

count = [0] * len(classes_info)

check_classes_info = []

bug_count = 0
all_count = 0
per_img_label_count = []


def get_label_info(xml_path, img_height, img_width):
    global check_classes_info
    global classes_info
    global count
    global bug_count
    global all_count
    global per_img_label_count

    tree = ET.parse(xml_path)
    # size = tree.find("size")
    # width = eval(size.find("width").text)
    # height = eval(size.find("height").text)
    # dior_annotations = tree.find('annotation')
    objects = tree.findall("object")
    # ic(len(objects))

    if len(objects) is not None:
        labels = [None] * len(objects)
        per_img_label_count.append(len(objects))

        for index, object in enumerate(objects):
            class_name = object.find("name").text
            if class_name not in check_classes_info:
                check_classes_info.append(class_name)
            class_id = classes_info.index(class_name)

            x1 = eval(object.find("robndbox").find("x_left_top").text)
            y1 = eval(object.find("robndbox").find("y_left_top").text)
            x2 = eval(object.find("robndbox").find("x_right_top").text)
            y2 = eval(object.find("robndbox").find("y_right_top").text)
            x3 = eval(object.find("robndbox").find("x_right_bottom").text)
            y3 = eval(object.find("robndbox").find("y_right_bottom").text)
            x4 = eval(object.find("robndbox").find("x_left_bottom").text)
            y4 = eval(object.find("robndbox").find("y_left_bottom").text)

            # NOTE angle ranger from -90 to 90
            # angle = eval(object.find("angle").text)


            box = [x1, y1, x2, y2, x3, y3, x4, y4]
            longside_info, eightpotins, _, error_label_flag, info = eightPointsToFiveAugments(box, 800, 800, 8)

            if error_label_flag:
                print(info)
                # print(longside_info)
                # print(eightpotins)
                bug_count += 1
                continue
            all_count += 1

            x = [eightpotins[0], eightpotins[2], eightpotins[4], eightpotins[6]]
            y = [eightpotins[1], eightpotins[3], eightpotins[5], eightpotins[7]]
            xmin = min(x)
            xmax = max(x)
            ymin = min(y)
            ymax = max(y)

            x_center = longside_info[0][0]
            y_center = longside_info[0][1]
            longside = longside_info[1][0]
            shortside = longside_info[1][1]
            angle = longside_info[2]
            # * 对于训练集, [xmin, ymin, xmax, yamx, x_center, y_center, longside, shortside, class_id, angle]
            # * xminyminxmaxymax+五参数
            all_info = [
                # xmin,
                # ymin,
                # xmax,
                # ymax,
                class_id,
                x_center / img_width,
                y_center / img_height,
                longside / img_width,
                shortside /img_height,
                angle,
            ]
            # * fot test set, [x_center, y_center, longside, shortside, angle, class_id]
            # all_info = [
            #     x_center,
            #     y_center,
            #     longside,
            #     shortside,
            #     angle,
            #     class_id,
            # ]
            labels[index] = [str(x) for x in all_info]

        # try:
        #     ind = labels.index(None)
        #     labels.pop(ind)
        # except:
        #     pass

        return labels
        # return None
    else:
        print("No labels find in xml")
        return None


# * 写入文件
def xml2txt(xml_path: Path):

    store_path = Path("./info_train/")
    if store_path.exists() is False:
        store_path.mkdir()
    test_list = []
    test_labels_list = []
    test_list_path = Path("./ImageSets/test.txt")
    test_store_path = store_path / test_list_path.name
    train_list = []
    train_labels_list = []
    train_list_path = Path("./ImageSets/train.txt")
    train_store_path = store_path / train_list_path.name
    train_val_list = []
    train_val_labels_list = []
    train_val_list_path = Path("./ImageSets/trainval.txt")
    train_val_store_path = store_path / train_val_list_path.name
    val_list = []
    val_labels_list = []
    val_list_path = Path("./ImageSets/val.txt")
    val_store_path = store_path / val_list_path.name

    with test_list_path.open("r") as f:
        for i in f.readlines():
            test_list.append(i.strip())
    with train_list_path.open("r") as f:
        for i in f.readlines():
            train_list.append(i.strip())
    with train_val_list_path.open("r") as f:
        for i in f.readlines():
            train_val_list.append(i.strip())
    with val_list_path.open("r") as f:
        for i in f.readlines():
            val_list.append(i.strip())

    for xml_path in xml_path.glob("*.xml"):
        labels = get_label_info(xml_path=xml_path)
        if len(labels) == 0:
            continue
        # print(labels)
        str_labels = [",".join(x) for x in labels if x != None]
        # img_labels = "/home/wh/HRSC2016/Images/" + xml_path.with_suffix(".png").name + " " + " ".join(str_labels) + "\n"
        # * 只写入文件名
        img_labels = xml_path.with_suffix(".jpg").name + " " + " ".join(str_labels) + "\n"

        if xml_path.stem in test_list:
            test_labels_list.append(img_labels)
        if xml_path.stem in train_list:
            train_labels_list.append(img_labels)
        if xml_path.stem in train_val_list:
            train_val_labels_list.append(img_labels)
        if xml_path.stem in val_list:
            val_labels_list.append(img_labels)

    ic(len(test_list), len(test_labels_list))
    ic(len(train_list), len(train_labels_list))
    ic(len(val_list), len(val_labels_list))
    ic(len(train_val_list), len(train_val_labels_list))

    with test_store_path.open("w+") as f:
        f.writelines(test_labels_list)
    with train_store_path.open("w+") as f:
        f.writelines(train_labels_list)
    with val_store_path.open("w+") as f:
        f.writelines(val_labels_list)
    with train_val_store_path.open("w+") as f:
        f.writelines(train_val_labels_list)


if __name__ == "__main__":
    import cv2

    xml_path = Path("./Annotations/Oriented Bounding Boxes/")
    txt_path = Path("./Annotations_txt/")
    img_prefix_path = Path("./Images")
    for xml_path in xml_path.glob("*.xml"):
        img_path = img_prefix_path / xml_path.with_suffix(".jpg").name
        # print(img_path)
        if not img_path.exists():
            continue
        img = cv2.imread(str(img_path))
        height, width, _ = img.shape
        labels = get_label_info(xml_path, height, width)
        if len(labels) == 0:
            continue
        str_labels = [" ".join(x) + "\n" for x in labels if x is not None]
        # img_labels = "/home/wh/HRSC2016/Images/" + xml_path.with_suffix(".png").name + " " + " ".join(str_labels) + "\n"
        # * 只写入文件名
        img_txt_path = txt_path / xml_path.with_suffix(".txt").name
        with img_txt_path.open("w+") as f:
            f.writelines(str_labels)

    # img_path = Path("./images/train/100001253.png")
    # img = cv2.imread(str(img_path))
    # h, w, _ = img.shape
    # xml_path = Path("./Annotations/100001253.xml")
    # objects = get_label_info(xml_path, h, w, 800)
    # print(objects)
    # showLabels(img_path, objects)
