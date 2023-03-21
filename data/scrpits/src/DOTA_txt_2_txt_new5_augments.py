from pathlib import Path

import cv2
import numpy as np
from icecream import ic
from rich import print
from rich.progress import track

from trans_tools import eightPointsToFiveAugments, showLabels

classes_info = [
    "plane",
    "baseball-diamond",
    "bridge",
    "ground-track-field",
    "small-vehicle",
    "large-vehicle",
    "ship",
    "tennis-court",
    "basketball-court",
    "storage-tank",
    "soccer-ball-field",
    "roundabout",
    "harbor",
    "swimming-pool",
    "helicopter",
]

ic(len(classes_info))

# note opencv-python=4.5.3

count = [0] * len(classes_info)

check_classes_info = []

bug_count = 0
all_count = 0
per_img_label_count = []


def get_label_info(txt_path, img_height, img_width, min_area=16):
    global check_classes_info
    global classes_info
    global count
    global bug_count
    global all_count
    global per_img_label_count

    # tree = et.parse(xml_path)
    # size = tree.find("size")
    # width = eval(size.find("width").text)
    # height = eval(size.find("height").text)
    # dior_annotations = tree.find('annotation')
    # objects = tree.findall("object")
    # ic(len(objects))

    objects = []
    with open(txt_path) as f:
        for i in f.readlines():
            objects.append(i.strip().split(" "))


    if len(objects) is not None:
        labels = [None] * len(objects)
        per_img_label_count.append(len(objects))

        for index, object in enumerate(objects):

            difficult_flag = object[-1]
            if difficult_flag == "2":
                continue

            class_name = object[-2]
            class_id = classes_info.index(class_name)

            x1 = eval(object[0])
            y1 = eval(object[1])
            x2 = eval(object[2])
            y2 = eval(object[3])
            x3 = eval(object[4])
            y3 = eval(object[5])
            x4 = eval(object[6])
            y4 = eval(object[7])

            # NOTE angle ranger from -90 to 90
            # angle = eval(object.find("angle").text)
            box = [x1, y1, x2, y2, x3, y3, x4, y4]

            longside_info, eightpotins, _, error_label_flag, info = eightPointsToFiveAugments(box, img_height, img_width, min_area)

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
            # all_info = [
            #     xmin,
            #     ymin,
            #     xmax,
            #     ymax,
            #     x_center,
            #     y_center,
            #     longside,
            #     shortside,
            #     class_id,
            #     angle,
            # ]
            # * fot test set, [x_center, y_center, longside, shortside, angle, class_id]
            all_info = [
                x_center,
                y_center,
                longside,
                shortside,
                angle,
                class_id,
            ]
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
def txt2txt():

    store_path = Path("/mnt/datasets/DOTAx/info_train_anchor")
    if store_path.exists() is False:
        store_path.mkdir()


    train_list = []
    train_list_path = Path("/mnt/datasets/DOTAx/train/labelTxt")

    val_list = []
    val_list_path = Path("/mnt/datasets/DOTAx/val/labelTxt")

    train_val_list = []

    train_val_store_path = store_path / "trainval.txt"

    for txt_path in track(list(train_list_path.glob("*.txt")), description="Processing train txt"):
        img_path = Path("/mnt/datasets/DOTAx/train/JPEGImages") / txt_path.with_suffix('.png').name
        img_height, img_width = cv2.imread(str(img_path)).shape[:2]
        labels = get_label_info(txt_path, img_height, img_width)
        if len(labels) == 0:
            continue
        # print(labels)
        str_labels = [",".join(x) for x in labels if x != None]
        # img_labels = "/home/wh/HRSC2016/Images/" + xml_path.with_suffix(".png").name + " " + " ".join(str_labels) + "\n"
        # * 只写入文件名
        img_labels = 'train/JPEGImages/' +  txt_path.with_suffix(".png").name + " " + " ".join(str_labels) + "\n"
        train_val_list.append(img_labels)

    for txt_path in track(list(val_list_path.glob("*.txt")), description="Processing val txt"):
        img_path = Path("/mnt/datasets/DOTAx/val/JPEGImages") / txt_path.with_suffix('.png').name
        img_height, img_width = cv2.imread(str(img_path)).shape[:2]
        labels = get_label_info(txt_path, img_height, img_width)
        if len(labels) == 0:
            continue
        # print(labels)
        str_labels = [",".join(x) for x in labels if x != None]
        # img_labels = "/home/wh/HRSC2016/Images/" + xml_path.with_suffix(".png").name + " " + " ".join(str_labels) + "\n"
        # * 只写入文件名
        img_labels = 'val/JPEGImages/' +  txt_path.with_suffix(".png").name + " " + " ".join(str_labels) + "\n"
        train_val_list.append(img_labels)

    ic(len(train_list), len(val_list))
    ic(len(train_val_list))

    with train_val_store_path.open("w+") as f:
        f.writelines(train_val_list)


if __name__ == "__main__":

    txt2txt()
    print('有问题label', bug_count)
    print('总label', all_count)
    print(max(per_img_label_count))
    print(min(per_img_label_count))
    print(np.mean(per_img_label_count))

    # for txt_path in track(list(Path("/mnt/datasets/DOTAx/train/labelTxt").glob("*.txt")), description="Processing train txt"):
    #     img_path = Path("/mnt/datasets/DOTAx/train/JPEGImages") / txt_path.with_suffix('.png').name
    #     img_height, img_width = cv2.imread(str(img_path)).shape[:2]
    #     labels = get_label_info(txt_path, img_height, img_width)
    #     showLabels(Path(img_path), labels)

    # print(sorted(check_classes_info))

    # TEST
    # print(get_label_info(Path("/mnt/datasets/DOTAx/train/labelTxt/P0000__1.0__0___0.txt")))
