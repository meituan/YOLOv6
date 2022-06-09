# YOLOv6

## Introduction

YOLOv6 is a single-stage object detection framework dedicated to industrial application, with hardware-friendly efficient design and high performance.

YOLOv6-nano achieves 35.0 mAP on COCOval dataset with 1242 FPS on T4 using TensorRT FP16 for bs32 inference, and YOLOv6-s achieves 43.1 mAP on COCOval dataset with 520 FPS on T4 using TensorRT FP16 for bs32 inference.

YOLOv6 is composed of following methods:
- Hardware-friendly Design for Backbone and Neck
- Efficient Decoupled Head with SIoU loss

## Coming soon
- [ ] YOLOv6 m/l/x model.
- [ ] Deployment for OPENVINO/MNN/TNN/NCNN...


## Quick Start

### Install

```shell
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
pip install -r requirements.txt  
```

### Inference 
First, download a pretrained model from the YOLOv6 release

Second, run inference with `tools/infer.py`

```shell
python tools/infer.py --weights yolov6s.pt --source [img.jpg / imgdir]
                                yolov6n.pt  
```

### Training

Single GPU 

```shell
python tools/train.py --batch 256 --conf configs/yolov6s.py --data data/coco.yaml --device 0
                                         configs/yolov6n.py
```

Multi GPUs (DDP mode recommended)

```shell
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 256 --conf configs/yolov6s.py --data data/coco.yaml --device 0,1,2,3,4,5,6,7
                                                                                        configs/yolov6n.py
```

- conf: select config file to specify network/optimizer/hyperparameters
- data: prepare COCO dataset and specify dataset paths in data.yaml


### Evaluation

Reproduce mAP on COCO val2017 dataset

```shell
python tools/eval.py --data data/coco.yaml  --batch 32 --weights yolov6s.pt --task val
                                                                 yolov6n.pt
```


### Deployment

Export as ONNX Format

```shell
python tools/export_onnx.py --weights yolov6s.pt --device 0
                                      yolov6n.pt
```


### Tutorials

*  [Train custom data](./docs/Train_custom_data.md)
*  [Test speed](./docs/Test_speed.md)



## Benchmark

 #### Nano model

| Model        | Size | mAP<sup>val<br/>0.5:0.95 | Speed<br/><sup>V100 fp16 b32 <br/>(ms) | Speed<br/><sup>V100 fp32 b32 <br/>(ms) | Speed<br/><sup>T4 trt fp16 b1 <br/>(fps) | Speed<br/><sup>T4 trt fp16 b32 <br/>(fps) | Params<br/> (M) | Flops<br/> (G) |
| :----------- | ---- | :----------------------- | :------------------------------------- | :------------------------------------- | ---------------------------------------- | ----------------------------------------- | --------------- | -------------- |
| YOLOX-nano   | 416  | 25.8                     | 0.5                                    | 0.6                                    | 737                                      | 1664                                       | 0.9             | 1.1            |
| YOLOX-tiny   | 416  | 32.8                     | 0.7                                    | 1.0                                    | 618                                      | 1120                                       | 5.1             | 6.5            |
| YOLOv5-nano     | 640  | 28.0                     | 0.6                                    | 1.0                                    | 584                                    | 672                                     | 1.9             | 4.5            |
| **YOLOv6-nano** | 416<br/>640  | 30.8<br/>35.0                     | 0.3<br/>0.5                                    | 0.4<br/>0.7                                    | 1100<br/>788                                    | 2716<br/>1242                                    | 4.3<br/>4.3             | 4.7<br/>11.1           |


#### Small model

| Model           | Size | mAP<sup>val<br/>0.5:0.95 | Speed<br/><sup>V100 fp16 b32 <br/>(ms) | Speed<br/><sup>V100 fp32 b32 <br/>(ms) | Speed<br/><sup>T4 trt fp16 b1 <br/>(fps) | Speed<br/><sup>T4 trt fp16 b32 <br/>(fps) | Params<br/> (M) | FLOPs<br/> (G) |
| :-------------- | :--- | :----------------------- | :------------------------------------- | :------------------------------------- | ---------------------------------------- | ----------------------------------------- | --------------- | -------------- |
| YOLOv5-s        | 640  | 37.4                     | 0.9                                    | 1.5                                    | 403                                    | 465                                     | 7.2             | 16.5           |
| YOLOX-s         | 640  | 40.5                     | 1.8                                    | 2.7                                    | 312                                    | 375                                     | 9.0             | 26.8           |
| PPYOLOE-s       | 640  | 42.7                     | n/a                                    | 2.2                                    | 218                                    | n/a                                       | 7.9             | 17.4           |
| **YOLOv6-tiny** | 640  | 41.3                     | 0.9                                    | 1.5                                    | 425                                    | 602                                     | 15.0            | 36.7           |
| **YOLOv6-s**    | 640  | 43.1                     | 1.0                                    | 1.7                                    | 373                                    | 520                                     | 17.2            | 44.2           |


- Comparison of the mAP and speed of different object detectors tested on COCO val2017 dataset.
- Speed results are tested in our environment using official codebase and model if not found from the corresponding official release. 
- Params and Flops of yolov6 are estimated on deploy model.