## YOLOv6 base model

English | [简体中文](./README_cn.md)

### Features

- This is a RepOpt-version implementation of YOLOv6 according to [QARepVGG](https://arxiv.org/abs/2212.01593).

- The QARep version models possess slightly lower float accuracy on COCO than the RepVGG version models, but achieve highly improved quantized accuracy.

- The INT8 accuracies listed were obtained using a simple PTQ process, as implemented in the [`onnx_to_trt.py`](../../deploy/TensorRT/onnx_to_trt.py) script. However, higher accuracies could be achieved using Quantization-Aware Training (QAT) due to the specific architecture design of the QARepVGG model.

### Performance

| Model                                                         | Size | Float<br/>mAP<sup>val<br/>0.5:0.95 | INT8<br/>mAP<sup>val<br/>0.5:0.95 | Speed<sup>T4<br/>trt fp16 b32 <br/>(fps) | Speed<sup>T4<br/>trt int8 b32 <br/>(fps) | Params<br/><sup> (M) | FLOPs<br/><sup> (G) |
| :----------------------------------------------------------- | -------- | :----------------------- | -------------------------------------- | --------------------------------------- | -------------------- | ------------------- | -------------------- |
| [**YOLOv6-N**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6n.pt) | 640      | 37.5            | 34.3                                    | 1286                                   | 1773                  |4.7                  | 11.4                |
| [**YOLOv6-N-qa**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6n_qa.pt) | 640      | 37.1            | 36.4                                    | 1286                                     | 1773                 | 4.7                  | 11.4             |
| [**YOLOv6-S**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6s.pt) | 640      | 45.0         | 41.3                                    | 513                                     | 1117                 | 18.5                 | 45.3                 |
| [**YOLOv6-S-qa**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6s_qa.pt) | 640      | 44.7         | 44.0                                    | 513                                     | 1117                 | 18.5                 | 45.3                  |
| [**YOLOv6-M**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6m.pt) | 640      | 50.0         | 48.1                                    | 250                                     | 439                 | 34.9                 | 85.8                 |
| [**YOLOv6-M-qa**](https://github.com/meituan/YOLOv6/releases/download/0.3.0/yolov6m_qa.pt) | 640      | 49.7         | 49.4                                    | 250                                     | 439                 | 34.9                 | 85.8                 |

- Speed is tested with TensorRT 8.4 on T4.
- We have not conducted experiments on the YOLOv6-L model since it does not use the RepVGG architecture.
- The processes of model training, evaluation, and inference are the same as the original ones. For details, please refer to [this README](https://github.com/meituan/YOLOv6#quick-start).
