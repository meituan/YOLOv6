"""
Yolov5-obb检测结果Json 文件转Voc Class Txt
--json_path 输入的json文件路径
--save_path 输出文件夹路径
"""

# from tqdm import tqdm
import argparse
import json
import os
import shutil
import cv2
import numpy as np
# For DOTA-v2.0
dotav2_classnames = [
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
    "container-crane",
    "airport",
    "helipad",
]

# For DOTA-v1.5
dotav15_classnames = [
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
    "container-crane",
]

# For DOTA-v1.0
dotav1_classnames = [
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

DOSR = [
    "Transpot",
    "Yacht",
    "Speedboat",
    "Auxiliary Ship",
    "Military Ship",
    "Tug",
    "Fishing Boat",
    "Bulk Cargo Vessel",
    "Cargo",
    "Container",
    "Cruise",
    "Deckbarge",
    "Tanker",
    "Deckship",
    "Flat Traffic Ship",
    "Floating Crane",
    "Multihull",
    "Barge",
    "Communication Ship",
    "Submarine",
]


DOTA_CLASSES = dotav1_classnames
# DOTA_CLASSES = DOSR


def parse_args():
    parser = argparse.ArgumentParser(description="TestJson2VocClassTxt")
    parser.add_argument("--json_path", type=str)
    parser.add_argument("--save_path", type=str)
    args = parser.parse_args()
    return args


if __name__ == "__main__":
    # json_path = "/home/haohao/YOLOR-New/DOTA/run_test-D2022-06-21T12+16+06/epoch_117_predictions.json"
    # save_path = "/home/haohao/YOLOR-New/DOTA/run_test-D2022-06-21T12+16+06/predictions"
    args = parse_args()
    json_path = args.json_path
    save_path = args.save_path

    ana_txt_save_path = save_path

    data = json.load(open(json_path, "r"))
    if os.path.exists(ana_txt_save_path):
        shutil.rmtree(ana_txt_save_path)  # delete output folderX
    os.makedirs(ana_txt_save_path)
    results = [None] * len(DOTA_CLASSES)

    for data_dict in data:
        img_name = data_dict["file_name"]
        score = data_dict["score"]
        poly = data_dict["poly"]
        classid = data_dict["category_id"]
        classname = DOTA_CLASSES[classid]


        # # NOTE DOTA 9, 11 类直接外界矩
        # if classid == 9 or classid == 11:
        #     poly = cv2.minAreaRect(np.array(poly).reshape(-1, 2))
        #     poly = poly.reshape(-1, 8).tolist()

        lines = "%s %s %s %s %s %s %s %s %s %s\n" % (
            img_name,
            score,
            poly[0],
            poly[1],
            poly[2],
            poly[3],
            poly[4],
            poly[5],
            poly[6],
            poly[7],
        )
        if results[classid] is None:
            results[classid] = []
        else:
            results[classid].append(lines)

    for i in range(len(DOTA_CLASSES)):
        with open(str(ana_txt_save_path + "/Task1_" + DOTA_CLASSES[i] + ".txt"), "w") as f:
            if results[i] is None:
                f.write("")
            else:
                f.writelines(results[i])

    print("Done!")
