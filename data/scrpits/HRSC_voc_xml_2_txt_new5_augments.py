import math
import xml.etree.ElementTree as ET
from pathlib import Path

from icecream import ic
from numpy import count_nonzero

from src.trans_tools import (
    eightPointsToFiveAugments,
    longSideFormat2minAreaRect,
    minAreaRect2longSideFormat,
    showLabels,
    sortedEightPoints,
)

# from numpy import short


"""
Class Name	    Class Short Name	Total amount	Train amount	Val amount	Train+Val amount	Test amount
Transpot	        Tra.	        1682	        820	            358	            1178            504
Yacht	            Yac.	        1188	        583	            214	            797	            391
Speedboat	        Spe.	        529	            166	            69	            235	            294
Auxiliary Ship	    Aux.	        426	            216	            95	            311	            115
Military Ship	    Mli.	        419	            204	            68	            272	            147
Tug	                Tug	            293	            169	            58	            227	            66
Fishing Boat	    Fis.	        276	            166	            49	            215	            61
Bulk Cargo Vessel	BCV.	        275	            144	            60	            204	            71
Cargo	            Car.	        249	            109	            51	            160	            89
Container	        Con.	        177	            92	            38	            130	            47
Cruise	            Cru.	        165	            86	            30	            116	            49
Deckbarge	        DeB.	        100	            51	            20	            71	            29
Tanker	            Tan.	        86	            45	            20	            65	            21
Deckship            DeS.	        69	            40	            12	            52	            17
Flat Traffic Ship	FTS.	        59	            42	            6	            48	            11
Floating Crane	    Flo.	        38	            16	            13	            29	            9
Multihull	        Mul.	        36	            14	            12	            26	            10
Barge	            Bar.	        34	            23	            4	            27	            7
Communication Ship	Com.	        14	            5	            3	            8	            6
Submarine	        Sub.	        12	            7	            1	            8	            4
"""

# classes_info = [
#     "Transpot",
#     "Yacht",
#     "Speedboat",
#     "Auxiliary Ship",
#     "Military Ship",
#     "Tug",
#     "Fishing Boat",
#     "Bulk Cargo Vessel",
#     "Cargo",
#     "Container",
#     "Cruise",
#     "Deckbarge",
#     "Tanker",
#     "Deckship",
#     "Flat Traffic Ship",
#     "Floating Crane",
#     "Multihull",
#     "Barge",
#     "Communication Ship",
#     "Submarine",
# ]

classes_info = ["ship"]

ic(len(classes_info))

# Note opencv-python=4.5.3

count = [0] * len(classes_info)


# 需要统计和排查两种，一个是角度统计，一个是label被覆盖的可能性
count_angle_num = [0] * 180
error_count = 0


def get_label_info(xml_path, img_height, img_weight, resize_img):

    global classes_info
    global count
    global count_angle_num
    global error_count
    tree = ET.parse(xml_path)
    # size = tree.find("size")
    # width = eval(size.find("width").text)
    # height = eval(size.find("height").text)
    HRSC_Objects = tree.find("HRSC_Objects")
    objects = HRSC_Objects.findall("HRSC_Object")
    # ic(len(objects))

    if len(objects) is not None:
        labels = [None] * len(objects)

        for index, object in enumerate(objects):
            class_name = object.find("Class_ID").text
            class_id = 0
            xmin = object.find("box_xmin").text
            ymin = object.find("box_ymin").text
            xmax = object.find("box_xmax").text
            ymax = object.find("box_ymax").text
            x_center = object.find("mbox_cx").text
            y_center = object.find("mbox_cy").text
            longside = object.find("mbox_w").text
            shortside = object.find("mbox_h").text
            angle = eval(object.find("mbox_ang").text)
            angle = math.degrees(angle)

            if angle < 0:
                angle += 180

            if 179 < angle <= 180:
                angle = 179

            count_angle_num[int(angle)] += 1
            # class_name = object.find("name").text
            # class_name = class_name.replace("_", " ").title()
            # class_id = classes_info.index(class_name)
            # count[class_id] += 1
            # bndbox = object.find("bndbox")

            # x1 = bndbox.find("x1").text
            # y1 = bndbox.find("y1").text
            # x2 = bndbox.find("x2").text
            # y2 = bndbox.find("y2").text
            # x3 = bndbox.find("x3").text
            # y3 = bndbox.find("y3").text
            # x4 = bndbox.find("x4").text
            # y4 = bndbox.find("y4").text
            # xmin = bndbox.find("xmin").text
            # ymin = bndbox.find("ymin").text
            # xmax = bndbox.find("xmax").text
            # ymax = bndbox.find("ymax").text
            # x_center = (float(xmin) + float(xmax)) / 2
            # y_center = (float(ymin) + float(ymax)) / 2

            # box = [x1, y1, x2, y2, x3, y3, x4, y4]

            # longside_info = eightPointsToFiveAugments(box)
            # x_center_1 = longside_info[0][0]
            # y_center_1 = longside_info[0][1]
            # ic(abs(x_center - x_center_1), abs(y_center - y_center_1))
            # longside = longside_info[1][0]
            # shortside = longside_info[1][1]
            # angle = longside_info[2]
            # * xminyminxmaxymax+五参数
            # NOTE 添加选取label
            all_info = [
                # xmin,
                # ymin,
                # xmax,
                # ymax,
                class_id,
                eval(x_center) / img_weight,
                eval(y_center) / img_height,
                eval(longside) / img_weight,
                eval(shortside) / img_height,
                angle,
            ]
            labels[index] = [str(x) for x in all_info]

        # Test
        # print(min(resize_img / height, resize_img / width))
        # scale_size = min(resize_img / height, resize_img / width)
        # print(scale_size)
        # down_size = 8.0
        # for i in range(len(labels)):
        #     x_center = eval(labels[i][4])
        #     y_center = eval(labels[i][5])
        #     for j in range(len(labels)):
        #         x_center_a = eval(labels[j][4])
        #         y_center_a = eval(labels[j][5])
        #         if i != j:
        #             if int(x_center * scale_size / down_size) == int(x_center_a * scale_size / down_size) and int(
        #                 y_center * scale_size / down_size
        #             ) == int(y_center_a * scale_size / down_size):
        #                 error_count += 1
        #                 print("hello")
        #                 # error_count += 1
        return labels
    else:
        print("No labels find in xml")
        return None


# * 写入文件
def xml2txt(xml_path: Path):

    store_path = Path("/home/wh/HRSC2016/info_train/")
    if store_path.exists() is False:
        store_path.mkdir()
    test_list = []
    test_labels_list = []
    test_list_path = Path("/home/wh/HRSC2016/ImageSets/test.txt")
    test_store_path = store_path / test_list_path.name
    train_list = []
    train_labels_list = []
    train_list_path = Path("/home/wh/HRSC2016/ImageSets/train.txt")
    train_store_path = store_path / train_list_path.name
    train_val_list = []
    train_val_labels_list = []
    train_val_list_path = Path("/home/wh/HRSC2016/ImageSets/trainval.txt")
    train_val_store_path = store_path / train_val_list_path.name
    val_list = []
    val_labels_list = []
    val_list_path = Path("/home/wh/HRSC2016/ImageSets/val.txt")
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
        str_labels = [",".join(x) for x in labels]
        # img_labels = "/home/wh/HRSC2016/Images/" + xml_path.with_suffix(".png").name + " " + " ".join(str_labels) + "\n"
        # * 只写入文件名
        img_labels = xml_path.with_suffix(".png").name + " " + " ".join(str_labels) + "\n"

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

    # xml_path = Path("./Annotations")
    # txt_path = Path("./Annotations_txt/")
    # img_prefix_path = Path("./Images")

    # for xml_path in xml_path.glob("*.xml"):
    #     img_path = img_prefix_path / xml_path.with_suffix(".png").name
    #     # print(img_path)
    #     if not img_path.exists():
    #         continue
    #     img = cv2.imread(str(img_path))
    #     height, width, _ = img.shape
    #     labels = get_label_info(xml_path, height, width, 800)

    #     if len(labels) == 0:
    #         continue

    #     str_labels = [" ".join(x) + "\n" for x in labels]
    #     # img_labels = "/home/wh/HRSC2016/Images/" + xml_path.with_suffix(".png").name + " " + " ".join(str_labels) + "\n"
    #     # * 只写入文件名
    #     img_txt_path = txt_path / xml_path.with_suffix(".txt").name

    #     with img_txt_path.open("w+") as f:
    #         f.writelines(str_labels)

    img_path = Path("./images/train/100001253.png")
    img = cv2.imread(str(img_path))
    h, w, _ = img.shape
    xml_path = Path("./Annotations/100001253.xml")
    objects = get_label_info(xml_path, h, w, 800)
    print(objects)
    showLabels(img_path, objects)
