## YOLOv6 MBLA版模型

简体中文 | [English](./README.md)

### 模型特点

- 网络主体结构均采用MBLABlock(Multi Branch Layer Aggregation Block)

优势：
- 采用统一的网络结构和配置

- 相比3.0版本在s尺度效果提升，相比3.0base版本各尺度效果提升



### 模型指标

| 模型                                                         | 输入尺寸 | mAP<sup>val<br/>0.5:0.95 | 速度<sup>T4<br/>trt fp16 b1 <br/>(fps) | 速度<sup>T4<br/>trt fp16 b32 <br/>(fps) | Params<br/><sup> (M) | FLOPs<br/><sup> (G) |
| :----------------------------------------------------------- | -------- | :----------------------- | -------------------------------------- | --------------------------------------- | -------------------- | ------------------- |
| [**YOLOv6-S-mbla**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s_mbla.pt) | 640      | 47.0<sup>distill            | 300                                    | 424                                    | 11.6                  | 29.8                |
| [**YOLOv6-M-mbla**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m_mbla.pt) | 640      | 50.3<sup>distill            | 168                                    | 216                                     | 26.1                 | 66.7                |
| [**YOLOv6-L-mbla**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l_base.pt) | 640      | 52.0<sup>distill         | 129                                    | 154                                     | 46.3                 | 118.2                |
| [**YOLOv6-X-base**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6x_base.pt) | 640      | 53.5<sup>distill         | 78                                    | 94                                     | 78.8                 | 199.0               |

- 速度是在 T4 上测试的，TensorRT 版本为  8.4.2.4；
- 模型训练、评估、推理流程与原来保持一致，具体可参考 [首页 README 文档](https://github.com/meituan/YOLOv6/blob/main/README_cn.md#%E5%BF%AB%E9%80%9F%E5%BC%80%E5%A7%8B)。
