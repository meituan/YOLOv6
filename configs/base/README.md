## YOLOv6 base model

English | [简体中文](./README_cn.md)

### Features

- Use only regular convolution and Relu activation functions.

- Apply CSP (1/2 channel dim) blocks in the network structure, except for Nano base model.

Advantage:
- Adopt a unified network structure and configuration, and the accuracy loss of the PTQ 8-bit quantization model is negligible, about 0.4%.
- Suitable for users who are just getting started or who need to apply, optimize and deploy an 8-bit quantization model quickly and frequently.

Shortcoming:
- The accuracy on COCO is slightly lower than the v2.0 released models.

### Performance

| Model                                                         | Size | mAP<sup>val<br/>0.5:0.95 | Speed<sup>T4<br/>trt fp16 b1 <br/>(fps) | Speed<sup>T4<br/>trt fp16 b32 <br/>(fps) | Params<br/><sup> (M) | FLOPs<br/><sup> (G) |
| :----------------------------------------------------------- | -------- | :----------------------- | -------------------------------------- | --------------------------------------- | -------------------- | ------------------- |
| [**YOLOv6-N-base**](https://github.com/meituan/YOLOv6/releases/download/0.2.1/yolov6n_base.pt) | 640      | 35.6<sup>400e            | 832                                    | 1249                                    | 4.3                  | 11.1                |
| [**YOLOv6-S-base**](https://github.com/meituan/YOLOv6/releases/download/0.2.1/yolov6s_base.pt) | 640      | 43.8<sup>400e            | 373                                    | 531                                     | 11.5                 | 27.6                |
| [**YOLOv6-M-base**](https://github.com/meituan/YOLOv6/releases/download/0.2.1/yolov6m_base.pt) | 640      | 48.8<sup>distill         | 179                                    | 246                                     | 27.7                 | 68.4                |
| [**YOLOv6-L-base**](https://github.com/meituan/YOLOv6/releases/download/0.2.1/yolov6l_base.pt) | 640      | 51.0<sup>distill         | 115                                    | 153                                     | 58.5                 | 144.0               |

- Speed is tested with TensorRT 7.2 on T4.
- The processes of model training, evaluation, and inference are the same as the original ones. For details, please refer to [this README](https://github.com/meituan/YOLOv6#quick-start).