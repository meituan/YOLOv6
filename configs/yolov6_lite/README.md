## YOLOv6Lite model

English | [简体中文](./README_cn.md)

## Mobile Benchmark
| Model | Size | mAP<sup>val<br/>0.5:0.95 | sm8350<br/><sup>(ms) | mt6853<br/><sup>(ms) | sdm660<br/><sup>(ms) |Params<br/><sup> (M) |   FLOPs<br/><sup> (G) |
| :----------------------------------------------------------- | ---- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| [**YOLOv6Lite-S**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_s.pt) | 320*320 | 22.4                     | 7.99                     | 11.99                     | 41.86                     | 0.55                     | 0.56                     |
| [**YOLOv6Lite-M**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_m.pt) | 320*320 | 25.1                     | 9.08                     | 13.27                     | 47.95                     | 0.79                     | 0.67                     |
| [**YOLOv6Lite-L**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_l.pt) | 320*320 | 28.0                     | 11.37                     | 16.20                     | 61.40                     | 1.09                     | 0.87                     |
| [**YOLOv6Lite-L**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_l.pt) | 320*192 | 25.0                     | 7.02                     | 9.66                     | 36.13                     | 1.09                     | 0.52                     |
| [**YOLOv6Lite-L**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_l.pt) | 224*128 | 18.9                     | 3.63                     | 4.99                     | 17.76                     | 1.09                     | 0.24                     |

<details open>
<summary>Table Notes</summary>

- From the perspective of model size and input image ratio, we have built a series of models on the mobile terminal to facilitate flexible applications in different scenarios.
- All checkpoints are trained with 400 epochs without distillation.
- Results of the mAP and speed are evaluated on [COCO val2017](https://cocodataset.org/#download) dataset, and the input resolution is the Size in the table.
- Speed is tested on MNN 2.3.0 AArch64 with 2 threads by arm82 acceleration. The inference warm-up is performed 10 times, and the cycle is performed 100 times.
- Qualcomm 888(sm8350), Dimensity 720(mt6853) and Qualcomm 660(sdm660) correspond to chips with different performances at the high, middle and low end respectively, which can be used as a reference for model capabilities under different chips.
- Refer to [Test NCNN Speed](./docs/Test_NCNN_speed.md) tutorial to reproduce the NCNN speed results of YOLOv6Lite.
