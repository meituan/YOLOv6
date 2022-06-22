# Train Custom Data

This guidence explains how to train your own custom data with YOLOv6 ( take fine-tuning YOLOv6-s model for example).

## 0. Before you start

Clone this repo and follow README.md to install requirements in a Python3.8 environment.


## 1. Prepare your own dataset

**Step 1** Prepare your own dataset with images. For labeling images, you can use tools like [Labelme](https://github.com/wkentaro/labelme).

**Step 2** Generate label files in YOLO format.

One image corresponds to one label file, and the label format example is presented as below.

```json
# class_id center_x center_y bbox_width bbox_height
0 0.300926 0.617063 0.601852 0.765873
1 0.575 0.319531 0.4 0.551562
```

- Each row represents one object.
- Class id starts from `0`.
- Boundingbox coordinates must be in normalized `xywh` format (from 0 - 1). If your boxes are in pixels, divide `center_x` and `bbox_width` by image width, and `center_y` and `bbox_height` by image height.

**Step 3** Organize directories.

Organize your train and val images and label files according to the example below.

```shell
# image directory
path/to/data/images/train/im0.jpg
path/to/data/images/val/im1.jpg
path/to/data/images/test/im2.jpg

# label directory
path/to/data/labels/train/im0.txt
path/to/data/labels/val/im1.txt
path/to/data/labels/test/im2.txt
```

**Step 4** Create `dataset.yaml` in `$YOLOv6_DIR/data`.

```yaml
train: path/to/data/images/train # train images
val: path/to/data/images/val # val images
test: path/to/data/images/test # test images (optional)

# Classes
nc: 20  # number of classes
names: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']  # class names

```


## 2. Create a config file

We use a config file to specify the network structure and training setting, including  optimizer and data augmentation hyperparameters.

If you create a new config file, please put it under the configs directory.
Or just use the provided config file in `$YOLOV6_HOME/configs/*_finetune.py`.

```python
## YOLOv6s Model config file
model = dict(
    type='YOLOv6s',
    pretrained='./weights/yolov6s.pt', # download pretrain model from YOLOv6 github if use pretrained model
    depth_multiple = 0.33,
    width_multiple = 0.50,
    ...
)
solver=dict(
    optim='SGD',
    lr_scheduler='Cosine',
    ...
)

data_aug = dict(
    hsv_h=0.015,
    hsv_s=0.7,
    hsv_v=0.4,
    ...
)
```



## 3. Train

Single GPU

```shell
python tools/train.py --batch 256 --conf configs/yolov6s_finetune.py --data data/data.yaml --device 0
```

Multi GPUs (DDP mode recommended)

```shell
python -m torch.distributed.launch --nproc_per_node 4 tools/train.py --batch 256 --conf configs/yolov6s_finetune.py --data data/data.yaml --device 0,1,2,3
```



## 4. Evaluation

```shell
python tools/eval.py --data data/data.yaml  --weights output_dir/name/weights/best_ckpt.pt --device 0
```



## 5. Inference

```shell
python tools/infer.py --weights output_dir/name/weights/best_ckpt.pt --source img.jpg --device 0
```



## 6. Deployment

Export as ONNX Format

```shell
python deploy/ONNX/export_onnx.py --weights output_dir/name/weights/best_ckpt.pt --device 0
```
