<p align="center">
  <img src="assets/banner-YOLO.png" align="middle" width = "1000" />
</p>

简体中文 | [English](README.md)

## YOLOv6

官方论文: [YOLOv6: A Single-Stage Object Detection Framework for Industrial Applications](https://arxiv.org/abs/2209.02976)

<p align="center">
  <img src="assets/speed_comparision_v2.png" align="middle" width = "1000" />
</p>

YOLOv6 提供了一系列面向各种工业应用场景的模型，包括微小级(nano)，极小极(tiny)、小(small)，中(medium)，大模型(large)。为更好的实现精度和速度权衡，这些模型的结构会根据模型大小而有所不同。此外，我们还引入了一些策略和技巧方法来进一步提高性能，例如自蒸馏和更多的训练轮次，这些策略和技巧并不会增加模型推理延时。在工业部署时，我们采用通道蒸馏和图优化的量化感知训练来实现极致的推理性能。

YOLOv6-N 在 COCO 数据集上的 mAP 为 35.9% ，在 T4 显卡推理速度可达 1234 FPS。 YOLOv6-S 的 mAP 为 43.5%，推理速度为 495 FPS ，量化后的 YOLOv6-S 模型在 T4显卡 上的 FPS 可以加速到 869，mAP 为 43.3% 。 YOLOv6-T/M/L 也具有出色的性能，与其他检测器相比，我们的模型在基本相同的推理速度时，可以达到更高的精度。

## 更新日志

- [2022.11.04] 发布 [基础版模型](configs/base/README_cn.md) 简化训练部署流程。
- [2022.09.06] 定制化的模型量化加速方法 🚀 [量化教程](./tools/qat/README.md)
- [2022.09.05] 发布 M/L 模型，并且进一步提高了 N/T/S 模型的性能 ⭐️ [模型指标](#模型指标)
- [2022.06.23] 发布 N/T/S v1.0 版本模型。

## 模型指标
| 模型                                                       | 输入尺寸 | mAP<sup>val<br/>0.5:0.95              | 速度<sup>T4<br/>trt fp16 b1 <br/>(fps) | 速度<sup>T4<br/>trt fp16 b32 <br/>(fps) | Params<br/><sup> (M) | FLOPs<br/><sup> (G) |
| :----------------------------------------------------------- | ---- | :------------------------------------ | --------------------------------------- | ---------------------------------------- | -------------------- | ------------------- |
| [**YOLOv6-N**](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6n.pt) | 640  | 35.9<sup>300e</sup><br/>36.3<sup>400e | 802                                     | 1234                                     | 4.3                  | 11.1                |
| [**YOLOv6-T**](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6t.pt) | 640  | 40.3<sup>300e</sup><br/>41.1<sup>400e | 449                                     | 659                                      | 15.0                 | 36.7                |
| [**YOLOv6-S**](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6s.pt) | 640  | 43.5<sup>300e</sup><br/>43.8<sup>400e | 358                                     | 495                                      | 17.2                 | 44.2                |
| [**YOLOv6-M**](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6m.pt) | 640  | 49.5                                  | 179                                     | 233                                      | 34.3                 | 82.2                |
| [**YOLOv6-L-ReLU**](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6l_relu.pt) | 640  | 51.7                                  | 113                                     | 149                                      | 58.5                 | 144.0               |
| [**YOLOv6-L**](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6l.pt) | 640  | 52.5                                  | 98                                      | 121                                      | 58.5                 | 144.0               |
- 速度是在 T4 上测试的，TensorRT 版本为 7.2；

### 量化模型 🚀

| 模型                 | 输入尺寸 | 精度 | mAP<sup>val<br/>0.5:0.95 | 速度<sup>T4<br/>trt b1 <br/>(fps) | 速度<sup>T4<br/>trt b32 <br/>(fps) |
| :-------------------- | ---- | --------- | :----------------------- | ---------------------------------- | ----------------------------------- |
| **YOLOv6-N RepOpt** | 640  | INT8      | 34.8                     | 1114                               | 1828                                |
| **YOLOv6-N**        | 640  | FP16      | 35.9                     | 802                                | 1234                                |
| **YOLOv6-T RepOpt** | 640  | INT8      | 39.8                     | 741                                | 1167                                |
| **YOLOv6-T**        | 640  | FP16      | 40.3                     | 449                                | 659                                 |
| **YOLOv6-S RepOpt** | 640  | INT8      | 43.3                     | 619                                | 924                                 |
| **YOLOv6-S**        | 640  | FP16      | 43.5                     | 377                                | 541                                 |

- 速度是在 T4 上测试的，TensorRT 版本为 8.4；
- 精度是在训练 300 epoch 的模型上测试的；
- mAP 和速度指标是在 [COCO val2017](https://cocodataset.org/#download)  数据集上评估的，输入分辨率为 640×640；
- 复现 YOLOv
6 的速度指标，请查看 [速度测试](./docs/Test_speed.md) 教程；
- YOLOv6 的参数和计算量是在推理模式下计算的；

<details>
<summary>旧版模型</summary>

| 模型           | 输入尺寸        | mAP<sup>val<br/>0.5:0.95 | 速度<sup>V100<br/>fp16 b32 <br/>(ms) | 速度<sup>V100<br/>fp32 b32 <br/>(ms) | 速度<sup>T4<br/>trt fp16 b1 <br/>(fps) | 速度<sup>T4<br/>trt fp16 b32 <br/>(fps) | Params<br/><sup> (M) | FLOPs<br/><sup> (G) |
| :-------------- | ----------- | :----------------------- | :------------------------------------ | :------------------------------------ | ---------------------------------------- | ----------------------------------------- | --------------- | -------------- |
| [**YOLOv6-N**](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6n.pt)    | 416<br/>640 | 30.8<br/>35.0            | 0.3<br/>0.5                           | 0.4<br/>0.7                           | 1100<br/>788                             | 2716<br/>1242                             | 4.3<br/>4.3     | 4.7<br/>11.1   |
| [**YOLOv6-T**](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6t.pt) | 640         | 41.3                     | 0.9                                   | 1.5                                   | 425                                      | 602                                       | 15.0            | 36.7           |
| [**YOLOv6-S**](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.pt)    | 640         | 43.1                     | 1.0                                   | 1.7                                   | 373                                      | 520                                       | 17.2            | 44.2           |


</details>


## 快速开始

<details>
<summary> 安装</summary>


```shell
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
pip install -r requirements.txt
```
</details>

<details open>
<summary> 训练 </summary>

单卡

```shell
python tools/train.py --batch 32 --conf configs/yolov6s_finetune.py --data data/dataset.yaml --device 0
```

多卡 （我们推荐使用 DDP 模式）

```shell
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 256 --conf configs/yolov6s_finetune.py --data data/dataset.yaml --device 0,1,2,3,4,5,6,7
```

- conf: 配置文件路径，里面包含网络结构、优化器配置、超参数信息。如果您是在自己的数据集训练，我们推荐您使用yolov6n/s/m/l_finetune.py配置文件；
- data: 数据集配置文件，以 COCO 数据集为例，您可以在 [COCO](http://cocodataset.org) 下载数据, 在这里下载 [YOLO 格式标签](https://github.com/meituan/YOLOv6/releases/download/0.1.0/coco2017labels.zip)；
- 确保您的数据集按照下面这种格式来组织；
```
├── coco
│   ├── annotations
│   │   ├── instances_train2017.json
│   │   └── instances_val2017.json
│   ├── images
│   │   ├── train2017
│   │   └── val2017
│   ├── labels
│   │   ├── train2017
│   │   ├── val2017
```


<details>
<summary>在COCO数据集复现我们的结果 ⭐️</summary>

Nano 模型
```shell
python -m torch.distributed.launch --nproc_per_node 4 tools/train.py \
									--batch 128 \
									--conf configs/yolov6n.py \
									--data data/coco.yaml \
									--epoch 400 \
									--device 0,1,2,3 \
									--name yolov6n_coco
```

Tiny 和 Small 模型
```shell
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
									--batch 256 \
									--conf configs/yolov6s.py \ # configs/yolov6t.py
									--data data/coco.yaml \
									--epoch 400 \
									--device 0,1,2,3,4,5,6,7 \
									--name yolov6s_coco # yolov6t_coco
```

Medium 和 Large 模型
```shell
# 第一步: 训练一个基础模型
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
									--batch 256 \
									--conf configs/yolov6m.py \ # configs/yolov6l.py
									--data data/coco.yaml \
									--epoch 300 \
									--device 0,1,2,3,4,5,6,7 \
									--name yolov6m_coco # yolov6l_coco


# 第二步: Self-distillation training
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py \
									--batch 256 \ # 128 for distillation of yolov6l
									--conf configs/yolov6m.py \ # configs/yolov6l.py
									--data data/coco.yaml \
									--epoch 300 \
									--device 0,1,2,3,4,5,6,7 \
									--distill \
									--teacher_model_path runs/train/yolov6m_coco/weights/best_ckpt.pt \ # # yolov6l_coco
									--name yolov6m_coco # yolov6l_coco

```
</details>

<details>
<summary>恢复训练</summary>


如果您的训练进程中断了，您可以这样恢复先前的训练进程。
```
# 单卡训练
python tools/train.py --resume

# 多卡训练
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --resume
```
上面的命令将自动在 YOLOv6 目录中找到最新保存的模型，然后恢复训练。

您也可以通过 `--resume` 参数指定要恢复的模型路径
```
# 记得把 /path/to/your/checkpoint/path  替换为您要恢复训练的模型权重路径
--resume /path/to/your/checkpoint/path
```
这将从您提供的模型路径恢复训练。

</details>
</details>

<details>
<summary> 评估</summary>
在 COCO val2017 数据集上复现我们的结果（输入分辨率 640x640） ⭐️

```shell
python tools/eval.py --data data/coco.yaml --batch 32 --weights yolov6s.pt --task val --reproduce_640_eval
```
- verbose: 如果要打印每一类的精度信息，请设置为 True；
- do_coco_metric: 设置 True / False 来打开或关闭 pycocotools 的评估；
- do_pr_metric: 设置 True / False 来显示或不显示精度和召回的指标；
- config-file: 指定一个包含所有评估参数的配置文件，例如 [yolov6n_with_eval_params.py](configs/experiment/yolov6n_with_eval_params.py)
</details>


<details>
<summary>推理</summary>

首先，从 [release页面](https://github.com/meituan/YOLOv6/releases/tag/0.2.0)  下载一个训练好的模型权重文件，或选择您自己训练的模型；

然后，通过 `tools/infer.py`文件进行推理。

```shell
python tools/infer.py --weights yolov6s.pt --source img.jpg / imgdir / video.mp4
```
</details>

<details>
<summary> 部署 </summary>

*  [ONNX](./deploy/ONNX)
*  [OpenCV Python/C++](./deploy/ONNX/OpenCV)
*  [OpenVINO](./deploy/OpenVINO)
*  [TensorRT](./deploy/TensorRT)
</details>

<details open>
<summary> 教程 </summary>

*  [训练自己的数据集](./docs/Train_custom_data.md)
*  [测速](./docs/Test_speed.md)
*  [ YOLOv6 量化教程](./docs/Tutorial%20of%20Quantization.md)
</details>


<details>
<summary> 第三方资源 </summary>

 * YOLOv6 NCNN Android app demo: [ncnn-android-yolov6](https://github.com/FeiGeChuanShu/ncnn-android-yolov6) from [FeiGeChuanShu](https://github.com/FeiGeChuanShu)
 * YOLOv6 ONNXRuntime/MNN/TNN C++: [YOLOv6-ORT](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/ort/cv/yolov6.cpp), [YOLOv6-MNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/mnn/cv/mnn_yolov6.cpp) and [YOLOv6-TNN](https://github.com/DefTruth/lite.ai.toolkit/blob/main/lite/tnn/cv/tnn_yolov6.cpp) from [DefTruth](https://github.com/DefTruth)
 * YOLOv6 TensorRT Python: [yolov6-tensorrt-python](https://github.com/Linaom1214/TensorRT-For-YOLO-Series) from [Linaom1214](https://github.com/Linaom1214)
 * YOLOv6 TensorRT Windows C++: [yolort](https://github.com/zhiqwang/yolov5-rt-stack/tree/main/deployment/tensorrt-yolov6) from [Wei Zeng](https://github.com/Wulingtian)
 * [YOLOv6 web demo](https://huggingface.co/spaces/nateraw/yolov6) on [Huggingface Spaces](https://huggingface.co/spaces) with [Gradio](https://github.com/gradio-app/gradio). [![Hugging Face Spaces](https://img.shields.io/badge/%F0%9F%A4%97%20Hugging%20Face-Spaces-blue)](https://huggingface.co/spaces/nateraw/yolov6)
 * 教程: [如何用 YOLOv6 训练自己的数据集](https://blog.roboflow.com/how-to-train-yolov6-on-a-custom-dataset/) <a href="https://colab.research.google.com/drive/1YnbqOinBZV-c9I7fk_UL6acgnnmkXDMM"><img src="https://colab.research.google.com/assets/colab-badge.svg" alt="Open In Colab"></a>
 * YOLOv6 在 Google Colab 上的推理 Demo [![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/mahdilamb/YOLOv6/blob/main/inference.ipynb)
</details>

### [FAQ（持续更新）](https://github.com/meituan/YOLOv6/wiki/FAQ%EF%BC%88Continuously-updated%EF%BC%89)
