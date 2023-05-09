# Training size explanation

YOLOv6 support three training size mode.

## 1. Square shape training
If you only pass one number to  `--img-size`, such as `--img-size 640`, the longer side of image will be keep ratio resized to 640, the shorter side will be scaled with the same ratio, then padded to 640. The image send to the model with resolution (640, 640, 3).

## 2. Rectangle shape training
If you pass `--img-size 640` and `--rect`, the longer side of image will be keep ratio resized to 640, the shorter side will be scaled with the same ratio, then it will be padded to multiple of 32 (if needed).
For example, if one image's shape is (720, 1280, 3), after keep ratio resize, it's shape will change to (360, 640, 3), however, 320 is not multiple of 32, so it will be padded to (384, 640, 3).

## 3. Specific shape

In the rectangle shape mode, the training process may have different traininng size, such as (1080, 1920, 3) and (1200, 1600, 3). If you want to specify one shape, you can use `--specific-shape` command and specify your training shape with `--height ` and `--width`, for example:
```
python tools/train.py --data data/dataset.yaml --conf configs/yolov6n.py --specific-shape --width 1920 --height 1080
```
Then, the resolution of the training data will be (1080, 1920, 3) regardless of the shape of the image in dataset.
