# Train COCO Dataset

This guidence shows the training commands for reproducing our results on COCO Dataset.


## For Mobile models

#### YOLOv6Lite-S/M/L

```shell
python -m torch.distributed.launch --nproc_per_node 4 tools/train.py \
									--batch 128 \
									--img_size 416 \   # train with 416 and eval with 320
									--conf configs/yolov6_lite/yolov6_lite_s.py \   # yolov6lite_m/l
									--data data/coco.yaml \
									--epoch 400 \
									--device 0,1,2,3 \
									--name yolov6_lite_s_coco
```

## For P5 models

#### YOLOv6-N

```shell
# Step 1: Training a base model
# Be sure to open use_dfl mode in config file (use_dfl=True, reg_max=16)

python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
									--batch 128 \
									--conf configs/yolov6n.py \
									--data data/coco.yaml \
									--epoch 300 \
									--fuse_ab \
									--device 0,1,2,3,4,5,6,7 \
									--name yolov6n_coco

# Step 2: Self-distillation training
# Be sure to open use_dfl mode in config file (use_dfl=True, reg_max=16)
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
									--batch 128 \
									--conf configs/yolov6n.py \
									--data data/coco.yaml \
									--epoch 300 \
									--device 0,1,2,3,4,5,6,7 \
									--distill \
									--teacher_model_path runs/train/yolov6n_coco/weights/best_ckpt.pt \
									--name yolov6n_coco
```


#### YOLOv6-S/M/L

```shell
# Step 1: Training a base model
# Be sure to open use_dfl mode in config file (use_dfl=True, reg_max=16)

python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
									--batch 256 \
									--conf configs/yolov6s.py \ # yolov6m/yolov6l
									--data data/coco.yaml \
									--epoch 300 \
									--fuse_ab \
									--device 0,1,2,3,4,5,6,7 \
									--name yolov6s_coco # yolov6m_coco/yolov6l_coco

# Step 2: Self-distillation training
# Be sure to open use_dfl mode in config file (use_dfl=True, reg_max=16)
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
									--batch 256 \ # 128 for distillation of yolov6l
									--conf configs/yolov6s.py \ # yolov6m/yolov6l
									--data data/coco.yaml \
									--epoch 300 \
									--device 0,1,2,3,4,5,6,7 \
									--distill \
									--teacher_model_path runs/train/yolov6s_coco/weights/best_ckpt.pt \
									--name yolov6s_coco # yolov6m_coco/yolov6l_coco

```

## For P6 models

#### YOLOv6-N6/S6

```shell
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
									--batch 128 \
									--img 1280 \
									--conf configs/yolov6s6.py \ # yolov6n6
									--data data/coco.yaml \
									--epoch 300 \
									--bs_per_gpu 16 \
									--device 0,1,2,3,4,5,6,7 \
									--name yolov6s6_coco # yolov6n6_coco

```


#### YOLOv6-M6/L6

```shell
# Step 1: Training a base model
# Be sure to open use_dfl mode in config file (use_dfl=True, reg_max=16)
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
									--batch 128 \
									--conf configs/yolov6l6.py \ # yolov6m6
									--data data/coco.yaml \
									--epoch 300 \
									--bs_per_gpu 16 \
									--device 0,1,2,3,4,5,6,7 \
									--name yolov6l6_coco # yolov6m6_coco

# Step 2: Self-distillation training
# Be sure to open use_dfl mode in config file (use_dfl=True, reg_max=16)
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
									--batch 128 \
									--conf configs/yolov6l6.py \ # yolov6m6
									--data data/coco.yaml \
									--epoch 300 \
									--bs_per_gpu 16 \
									--device 0,1,2,3,4,5,6,7 \
									--distill \
									--teacher_model_path runs/train/yolov6l6_coco/weights/best_ckpt.pt \
									--name yolov6l6_coco # yolov6m6_coco

```
