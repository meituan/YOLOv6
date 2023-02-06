# Train Custom Data

This guidence explains how to train your own custom data with YOLOv6 (take fine-tuning YOLOv6-s model for example).

## 0. Before you start

Clone this repo and follow README.md to install requirements in a Python3.8 environment.
```shell
$ git clone https://github.com/meituan/YOLOv6.git
```

## 1. Prepare your own dataset

**Step 1**: Prepare your own dataset with images. For labeling images, you can use tools like [Labelme](https://github.com/wkentaro/labelme) or [Roboflow](https://roboflow.com/).

**Step 2**: Generate label files in YOLO format.

One image corresponds to one label file, and the label format example is presented as below.

```json
# class_id center_x center_y bbox_width bbox_height
0 0.300926 0.617063 0.601852 0.765873
1 0.575 0.319531 0.4 0.551562
```


- Each row represents one object.
- Class id starts from `0`.
- Boundingbox coordinates must be in normalized `xywh` format (from 0 - 1). If your boxes are in pixels, divide `center_x` and `bbox_width` by image width, and `center_y` and `bbox_height` by image height.

**Step 3**: Organize directories.

Organize your directory of custom dataset as follows:

```shell
custom_dataset
├── images
│   ├── train
│   │   ├── train0.jpg
│   │   └── train1.jpg
│   ├── val
│   │   ├── val0.jpg
│   │   └── val1.jpg
│   └── test
│       ├── test0.jpg
│       └── test1.jpg
└── labels
    ├── train
    │   ├── train0.txt
    │   └── train1.txt
    ├── val
    │   ├── val0.txt
    │   └── val1.txt
    └── test
        ├── test0.txt
        └── test1.txt
```

**Step 4**: Create `dataset.yaml` in `$YOLOv6_DIR/data`.

```yaml
# Please insure that your custom_dataset are put in same parent dir with YOLOv6_DIR
train: ../custom_dataset/images/train # train images
val: ../custom_dataset/images/val # val images
test: ../custom_dataset/images/test # test images (optional)

# whether it is coco dataset, only coco dataset should be set to True.
is_coco: False

# Classes
nc: 20  # number of classes
names: ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle', 'bus', 'car', 'cat', 'chair', 'cow', 'diningtable', 'dog',
        'horse', 'motorbike', 'person', 'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']  # class names
```

## 2. Create a config file

We use a config file to specify the network structure and training setting, including  optimizer and data augmentation hyperparameters.

If you create a new config file, please put it under the `configs` directory.
Or just use the provided config file in `$YOLOV6_HOME/configs/*_finetune.py`. Download the pretrained model which you want to use from [here](https://github.com/meituan/YOLOv6#benchmark).

```python
## YOLOv6s Model config file
model = dict(
    type='YOLOv6s',
    pretrained='./weights/yolov6s.pt', # download the pretrained model from YOLOv6 github if you're going to use the pretrained model
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
# Be sure to open use_dfl mode in config file (use_dfl=True, reg_max=16) if you want to do self-distillation training further.
python tools/train.py --batch 32 --conf configs/yolov6s_finetune.py --data data/dataset.yaml --fuse_ab --device 0
```

Multi GPUs (DDP mode recommended)

```shell
# Be sure to open use_dfl mode in config file (use_dfl=True, reg_max=16) if you want to do self-distillation training further.
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 256 --conf configs/yolov6s_finetune.py --data data/dataset.yaml --fuse_ab --device 0,1,2,3,4,5,6,7
```

Self-distillation training

```shell
# Be sure to open use_dfl mode in config file (use_dfl=True, reg_max=16).
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 256 --conf configs/yolov6s_finetune.py --data data/dataset.yaml --distill --teacher_model_path your_model_path --device 0,1,2,3,4,5,6,7
```


## 4. Evaluation

```shell
python tools/eval.py --data data/data.yaml  --weights output_dir/name/weights/best_ckpt.pt --task val --device 0
```



## 5. Inference

```shell
python tools/infer.py --weights output_dir/name/weights/best_ckpt.pt --source img.jpg --device 0
```



## 6. Deployment

Export as [ONNX](https://github.com/meituan/YOLOv6/tree/main/deploy/ONNX) Format

```shell
# Without NMS OP, pure model.
python deploy/ONNX/export_onnx.py --weights output_dir/name/weights/best_ckpt.pt --simplify --device 0
# If you want to run with ONNX-Runtime (NMS integrated).
python deploy/ONNX/export_onnx.py --weights output_dir/name/weights/best_ckpt.pt --simplify --device 0 --dynamic-batch --end2end --ort
# If you want to run with TensorRT (NMS integrated).
python deploy/ONNX/export_onnx.py --weights output_dir/name/weights/best_ckpt.pt --simplify --device 0 --dynamic-batch --end2end
```
