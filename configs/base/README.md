## YOLOv6 base model

English | [简体中文](./README_cn.md)

### Features

- Use only regular convolution and Relu activation functions.

- Apply CSP (1/2 channel dim) blocks in the network structure, except for Nano base model.

Advantage:
- Adopt a unified network structure and configuration, and the accuracy loss of the PTQ 8-bit quantization model is negligible.
- Suitable for users who are just getting started or who need to apply, optimize and deploy an 8-bit quantization model quickly and frequently.


### Performance

| Model                                                         | Size | mAP<sup>val<br/>0.5:0.95 | Speed<sup>T4<br/>TRT FP16 b1 <br/>(FPS) | Speed<sup>T4<br/>TRT FP16 b32 <br/>(FPS) | Speed<sup>T4<br/>TRT INT8 b1 <br/>(FPS) | Speed<sup>T4<br/>TRT INT8 b32 <br/>(FPS) | Params<br/><sup> (M) | FLOPs<br/><sup> (G) |
| :--------------------------------------------------------------------------------------------- | --- | ----------------- | ----- | ---- | ---- | ---- | ----- | ------ |
| [**YOLOv6-N-base**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6n_base.pt) | 640 | 36.6<sup>distill  | 727   | 1302 | 814  | 1805 | 4.65  | 11.46  |
| [**YOLOv6-S-base**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6s_base.pt) | 640 | 45.3<sup>distill  | 346   | 525  | 487  | 908  | 13.14 | 30.6   |
| [**YOLOv6-M-base**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6m_base.pt) | 640 | 49.4<sup>distill  | 179   | 245  | 284  | 439  | 28.33 | 72.30  |
| [**YOLOv6-L-base**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6l_base.pt) | 640 | 51.1<sup>distill  | 116   | 157  | 196  | 288  | 59.61 | 150.89 |

- Speed is tested with TensorRT 8.2.4.2 on T4.
- The processes of model training, evaluation, and inference are the same as the original ones. For details, please refer to [this README](https://github.com/meituan/YOLOv6#quick-start).
