## YOLOv6 基础版模型

简体中文 | [English](./README.md)

### 模型特点

- 仅使用常规卷积和Relu激活函数

- 网络结构均采用CSP (1/2通道) block，Nano网络除外。

优势：
- 采用统一的网络结构和配置，且 PTQ 8位量化模型精度损失(约0.4%)较小，适合刚入门或有快速迭代部署8位量化模型需求的用户。

不足：
- COCO上精度对比2.0版本发布模型稍低。

### 模型指标

| 模型                                                         | 输入尺寸 | mAP<sup>val<br/>0.5:0.95 | 速度<sup>T4<br/>trt fp16 b1 <br/>(fps) | 速度<sup>T4<br/>trt fp16 b32 <br/>(fps) | Params<br/><sup> (M) | FLOPs<br/><sup> (G) |
| :----------------------------------------------------------- | -------- | :----------------------- | -------------------------------------- | --------------------------------------- | -------------------- | ------------------- |
| [**YOLOv6-N-base**](https://github.com/meituan/YOLOv6/releases/download/0.2.1/yolov6n_base.pt) | 640      | 35.6<sup>400e            | 832                                    | 1249                                    | 4.3                  | 11.1                |
| [**YOLOv6-S-base**](https://github.com/meituan/YOLOv6/releases/download/0.2.1/yolov6s_base.pt) | 640      | 43.8<sup>400e            | 373                                    | 531                                     | 11.5                 | 27.6                |
| [**YOLOv6-M-base**](https://github.com/meituan/YOLOv6/releases/download/0.2.1/yolov6m_base.pt) | 640      | 48.8<sup>distill         | 179                                    | 246                                     | 27.7                 | 68.4                |
| [**YOLOv6-L-base**](https://github.com/meituan/YOLOv6/releases/download/0.2.1/yolov6l_base.pt) | 640      | 51.0<sup>distill         | 115                                    | 153                                     | 58.5                 | 144.0               |

- 速度是在 T4 上测试的，TensorRT 版本为 7.2；
- 模型训练、评估、推理流程与原来保持一致，具体可参考 [首页 README 文档](https://github.com/meituan/YOLOv6/blob/main/README_cn.md#%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)。
