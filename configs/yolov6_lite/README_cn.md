## YOLOv6 轻量级模型

简体中文 | [English](./README.md)

## 移动端模型指标

| 模型 | 输入尺寸 | mAP<sup>val<br/>0.5:0.95 | sm8350<br/><sup>(ms) | mt6853<br/><sup>(ms) | sdm660<br/><sup>(ms) |Params<br/><sup> (M) |   FLOPs<br/><sup> (G) |
| :----------------------------------------------------------- | ---- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- | -------------------- |
| [**YOLOv6Lite-S**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_s.pt) | 320*320 | 22.4                     | 7.99                     | 11.99                     | 41.86                     | 0.55                     | 0.56                     |
| [**YOLOv6Lite-M**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_m.pt) | 320*320 | 25.1                     | 9.08                     | 13.27                     | 47.95                     | 0.79                     | 0.67                     |
| [**YOLOv6Lite-L**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_l.pt) | 320*320 | 28.0                     | 11.37                     | 16.20                     | 61.40                     | 1.09                     | 0.87                     |
| [**YOLOv6Lite-L**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_l.pt) | 320*192 | 25.0                     | 7.02                     | 9.66                     | 36.13                     | 1.09                     | 0.52                     |
| [**YOLOv6Lite-L**](https://github.com/meituan/YOLOv6/releases/download/0.4.0/yolov6lite_l.pt) | 224*128 | 18.9                     | 3.63                     | 4.99                     | 17.76                     | 1.09                     | 0.24                     |

<details open>
<summary>表格笔记</summary>

- 从模型尺寸和输入图片比例两种角度，在构建了移动端系列模型，方便不同场景下的灵活应用。
- 所有权重都经过 400 个 epoch 的训练，并且没有使用蒸馏技术。
-  mAP 和速度指标是在 COCO val2017 数据集上评估的，输入分辨率为表格中对应展示的。
- 使用 MNN 2.3.0 AArch64 进行速度测试。测速时，采用2个线程，并开启arm82加速，推理预热10次，循环100次。
- 高通888(sm8350)、天玑720(mt6853)和高通660(sdm660)分别对应高中低端不同性能的芯片，可以作为不同芯片下机型能力的参考。
- [NCNN 速度测试](./docs/Test_NCNN_speed.md)教程可以帮助展示及复现 YOLOv6Lite 的 NCNN 速度结果。
