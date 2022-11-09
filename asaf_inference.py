from tools.infer import *
from pathlib import Path





















if __name__ == '__main__':
    video = Path(r"/media/access/New Volume1/YOLOv6/data/custom_dataset/images/val").as_posix()
    engine = Path(r"/media/access/New Volume1/YOLOv6/runs/train/exp2 (with heavy augmentation)/weights/best_stop_aug_ckpt.pt").as_posix()
    config_file = Path(r"/media/access/New Volume1/YOLOv6/data/dataset.yaml")
    save_dir = Path(r"/home/access/Desktop/tmp/scoreboard_detector_yolov6").as_posix()
    save_txt = Path(r"/home/access/Desktop/tmp/why.txt").as_posix()
    inferer = Inferer(video,engine, 1,config_file,416,False)
    inferer.infer(conf_thres =0.25, iou_thres = 0.45, classes=None, max_det=2, save_dir=save_dir,save_txt= False,save_img=False, hide_labels=False, hide_conf=False,agnostic_nms=False,view_img=True)
    a = 2