# Quantization Practice for YOLOv6
For industrial deployment, it has been common practice to adopt quantization to further speed up runtime without much performance compromise. However, due to the heavy use of re-parameterization blocks in YOLOv6, previous PTQ techniques fail to produce high performance, while it is hard to incorporate QAT when it comes to matching fake quantizers during training and inference.

In order to solve the quantization problem of YOLOv6, we firstly reconstruct the network with RepOptimizer, and then perform well-designed PTQ and QAT skills on this model. Finally we can obtain a SOTA quantized result(mAP 43.3 at 869 QPS) for YOLOv6s.

Specific tutorials, please refer to the following links:
*  [Tutorial of RepOpt for YOLOv6](./tutorial_repopt.md)
*  [Tutorial of QAT for YOLOv6](../tools/qat/README.md)
*  [Partial Quantization](../tools/partial_quantization)
*  [PPQ Quantization](../tools/quantization/ppq)
