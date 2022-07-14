# RepOpt version implementation of YOLOv6
## Introduction
This is a RepOpt-version implementation of YOLOv6 according to RepOptimizer: https://arxiv.org/pdf/2205.15242.pdf @DingXiaoH \
It shows some advantages：
1. With only minor changes. it is compatible with the original repvgg version, and it is easy to reproduce the precision comparable with original version.
2. No more train/deploy transform. The target network is consistent when training and deploying.
3. A slight training acceleration of about 8%.
4. Last and the most important, It is quantization friendly. Compared to the original version, the mAP decrease of PTQ can be greatly improved. Furthermore, the architecture of RepOptimizer is friendly to wrap quant-models for QAT.

## Training
The training of V6-RepOpt can be divided into two stages, hyperparameter search and target network training.
1. hyperparameter search. This stage is used to get a suitable 'scale' for RepOptimizer, and the result checkpoint can be passed to stage2. Remember to add `training_mode='hyper_search'` in your config.
  ```
  python tools/train.py --batch 32 --conf configs/repopt/yolov6s_hs.py --data data/coco.yaml --device 0
  ```
  Or you can directly use the [pretrained scale](https://github.com/xingyueye/YOLOv6/releases/download/0.1.0/yolov6s_scale.pt) we provided and omit this stage.

2. Training. Add the flag of `training_mode='repopt'` and pretraind model `scales='./assets/yolov6s_scale.pt',` in your config
  ```
  python tools/train.py --batch 32 --conf configs/repopt/yolov6s_opt.py --data data/coco.yaml --device 0
  ```
## Evaluation
Reproduce mAP on COCO val2017 dataset, you can directly test our [pretrained model](https://github.com/xingyueye/YOLOv6/releases/download/0.1.0/yolov6s_opt.pt).
  ```
  python tools/eval.py --data data/coco.yaml --batch 32 --weights yolov6s_opt.pt --task val
  ```
## Benchmark
We train a yolov6s-repopt with 300epochs, the fp32 mAP is 42.4, while the mAP of PTQ is 40.5. More results is coming soon...
