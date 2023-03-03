## YOLOv6 基础版模型

简体中文 | [English](./README.md)

### 模型特点

- 仅使用常规卷积和Relu激活函数

- 网络结构均采用CSP (1/2通道) block，Nano网络除外。

优势：
- 采用统一的网络结构和配置，且 PTQ 8位量化模型精度损失较小，适合刚入门或有快速迭代部署8位量化模型需求的用户。


### 模型指标

| 模型 | 尺寸 | mAP<sup>val<br/>0.5:0.95 | 速度<sup>T4<br/>TRT FP16 b1 <br/>(FPS) | 速度<sup>T4<br/>TRT FP16 b32 <br/>(FPS) | 速度<sup>T4<br/>TRT INT8 b1 <br/>(FPS) | 速度<sup>T4<br/>TRT INT8 b32 <br/>(FPS) | Params<br/><sup> (M) | FLOPs<br/><sup> (G) |
| :--------------------------------------------------------------------------------------------- | --- | ----------------- | ----- | ---- | ---- | ---- | ----- | ------ |
| [**YOLOv6-N-base**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6n_base.pt) | 640 | 36.6<sup>distill  | 727   | 1302 | 814  | 1805 | 4.65  | 11.46  |
| [**YOLOv6-S-base**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6s_base.pt) | 640 | 45.3<sup>distill  | 346   | 525  | 487  | 908  | 13.14 | 30.6   |
| [**YOLOv6-M-base**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6m_base.pt) | 640 | 49.4<sup>distill  | 179   | 245  | 284  | 439  | 28.33 | 72.30  |
| [**YOLOv6-L-base**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6l_base.pt) | 640 | 51.1<sup>distill  | 116   | 157  | 196  | 288  | 59.61 | 150.89 |

- 速度是在 T4 上测试的，TensorRT 版本为 8.4.2.4；
- 模型训练、评估、推理流程与原来保持一致，具体可参考 [首页 README 文档](https://github.com/meituan/YOLOv6/blob/main/README_cn.md#%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)。
