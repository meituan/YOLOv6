## YOLOv6 mbla model

English | [简体中文](./README_cn.md)

### Features

- Apply MBLABlock(Multi Branch Layer Aggregation Block) blocks in the network structure.

Advantage:
- Adopt a unified network structure and configuration.

- Better performance for Small model comparing to yolov6 3.0 release.

- Better performance comparing to yolov6 3.0 base.



### Performance

| Model                                                         | Size | mAP<sup>val<br/>0.5:0.95 | Speed<sup>T4<br/>trt fp16 b1 <br/>(fps) | Speed<sup>T4<br/>trt fp16 b32 <br/>(fps) | Params<br/><sup> (M) | FLOPs<br/><sup> (G) |
| :----------------------------------------------------------- | -------- | :----------------------- | -------------------------------------- | --------------------------------------- | -------------------- | ------------------- |
| [**YOLOv6-S-mbla**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6s_mbla.pt) | 640      | 47.0<sup>distill            | 300                                    | 424                                    | 11.6                  | 29.8                |
| [**YOLOv6-M-mbla**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6m_mbla.pt) | 640      | 50.3<sup>distill            | 168                                    | 216                                     | 26.1                 | 66.7                |
| [**YOLOv6-L-mbla**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6l_base.pt) | 640      | 52.0<sup>distill         | 129                                    | 154                                     | 46.3                 | 118.2                |
| [**YOLOv6-X-base**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6x_base.pt) | 640      | 53.5<sup>distill         | 78                                    | 94                                     | 78.8                 | 199.0               |

- Speed is tested with TensorRT 8.4.2.4 on T4.
- The processes of model training, evaluation, and inference are the same as the original ones. For details, please refer to [this README](https://github.com/meituan/YOLOv6#quick-start).
