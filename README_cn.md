<p align="center">
  <img src="assets/banner-YOLO.png" align="middle" width = "1000" />
</p>

简体中文 | [English](README.md)

## YOLOv6-Segmentation

基于 [YOLOv6 v4.0 分支](https://github.com/meituan/YOLOv6/tree/main) 代码实现实例分割任务。

新特性
- 参考 YOLACT 和 SOLO 设计了两种不同的分割预测头

## MSCOCO 模型指标
| Model                                                        | Size | mAP<sup>box<br/>50-95 | mAP<sup>mask<br/>50-95 | Speed<sup>T4<br/>trt fp16 b1 <br/>(fps) |
| :----------------------------------------------------------- | ---- | :-------------------- | ---------------------- | --------------------------------------- |
| [**YOLOv6-N-seg**](https://github.com/meituan/YOLOv6/releases/download/0.4.1/yolov6n_seg.pt) | 640  | 35.3                  | 31.2                   | 645                                     |
| [**YOLOv6-S-seg**](https://github.com/meituan/YOLOv6/releases/download/0.4.1/yolov6s_seg.pt) | 640  | 44.0                  | 38.0                   | 292                                     |
| [**YOLOv6-M-seg**](https://github.com/meituan/YOLOv6/releases/download/0.4.1/yolov6m_seg.pt) | 640  | 48.2                  | 41.3                   | 148                                     |
| [**YOLOv6-L-seg**](https://github.com/meituan/YOLOv6/releases/download/0.4.1/yolov6l_seg.pt) | 640  | 51.1                  | 43.7                   | 93                                      |
| [**YOLOv6-X-seg**](https://github.com/meituan/YOLOv6/releases/download/0.4.1/yolov6x_seg.pt) | 640  | 52.2                  | 44.8                   | 47                                      |
|                                                              |      |                       |                        |                                         |
| [**YOLOv6-N-solo**](https://github.com/meituan/YOLOv6/releases/download/0.4.1/yolov6n_solo.pt) | 640  | 35.7                  | 31.3                   | 506                                     |
| [**YOLOv6-S-solo**](https://github.com/meituan/YOLOv6/releases/download/0.4.1/yolov6s_solo.pt) | 640  | 44.2                  | 39.0                   | 243                                     |
| [**YOLOv6-M-solo**](https://github.com/meituan/YOLOv6/releases/download/0.4.1/yolov6m_solo.pt) | 640  | 48.3                  | 42.2                   | 120                                     |
| [**YOLOv6-L-solo**](https://github.com/meituan/YOLOv6/releases/download/0.4.1/yolov6l_solo.pt) | 640  | 50.9                  | 44.4                   | 75                                      |
| [**YOLOv6-X-solo**](https://github.com/meituan/YOLOv6/releases/download/0.4.1/yolov6x_solo.pt) | 640  | 52.2                  | 45.0                   | 41                                      |

#### Table Notes
- YOLOv6-x-seg 模型采用 YOLACT 分割头, YOLOv6-x-solo 模型采用 SOLO 分割头.
- 所有权重都经过 300 个 epoch 的训练，并且没有使用蒸馏技术.
- mAP 和速度指标是在 COCO val2017 数据集上评估的，输入分辨率为640x640.
- 速度是在 T4 上测试的，TensorRT 版本为 8.5，且不包括后处理部分.
- YOLOv6 的参数和计算量是在推理模式下计算的.



## Quick Start
<details open>
<summary> 安装 </summary>


```shell
git clone https://github.com/meituan/YOLOv6
cd YOLOv6
git checkout yolov6-segss
pip install -r requirements.txt
```
</details>

<details open>
<summary> 训练 </summary>

单卡

```shell
python tools/train.py --batch 8 --conf configs/yolov6s_finetune.py --data data/coco.yaml --device 0
```

多卡 （我们推荐使用 DDP 模式）

```shell
python -m torch.distributed.launch --nproc_per_node 8 tools/train.py --batch 64 --conf configs/yolov6s_finetune.py --data data/coco.yaml --device 0,1,2,3,4,5,6,7
```
- fuse_ab: 当前版本暂不支持
- conf: 配置文件路径，里面包含网络结构、优化器配置、超参数信息。如果您是在自己的数据集训练，我们推荐您使用yolov6n/s/m/l_finetune.py配置文件
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
│   ├── LICENSE
│   ├── README.txt
```

YOLOv6 支持不同的输入分辨率模式，详情请参见 [如何设置输入大小](./docs/About_training_size_cn.md).

</details>


<details open>
<summary> 评估 </summary>

在 COCO val2017 数据集上复现我们的结果

```shell
python tools/eval.py --data data/coco.yaml --batch 32 --weights yolov6s.pt --task val
```
</details>


<details>
<summary>推理</summary>

首先，从 [release页面](https://github.com/meituan/YOLOv6/releases/tag/0.4.1)  下载一个训练好的模型权重文件，或选择您自己训练的模型；

然后，通过 `tools/infer.py`文件进行推理。


```shell
python tools/infer.py --weights yolov6s.pt --source img.jpg / imgdir / video.mp4
```
如果您想使用本地摄像头或者网络摄像头，您可以运行:
```shell
python tools/infer.py --weights yolov6s.pt --webcam --webcam-addr 0
```
`webcam-addr` 可以是本地摄像头的 ID，或者是 RTSP 地址。
</details>

<details>
<summary> 部署 </summary>

*  [ONNX](./deploy/ONNX)
*  [OpenCV Python/C++](./deploy/ONNX/OpenCV)
*  [OpenVINO](./deploy/OpenVINO)
*  [TensorRT](./deploy/TensorRT)
</details>

<details>
<summary> 教程 </summary>

*  [用户手册（中文版）](https://yolov6-docs.readthedocs.io/zh_CN/latest/) 
*  [训练 COCO 数据集](./docs/Train_coco_data.md)
*  [训练自定义数据集](./docs/Train_custom_data.md)
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

如果您有任何问题，欢迎加入我们的微信群一起讨论交流！
<p align="center">
  <img src="assets/wechat_qrcode.png" align="middle" width = "1000" />
</p>