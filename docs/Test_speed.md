# Test speed

This guidence explains how to reproduce speed results of YOLOv6. For fair comparison, the speed results do not contain the time cost of data pre-processing and NMS post-processing.

## 0. Prepare model

Download the models you want to test from the latest release.

## 1. Prepare testing environment

Refer to README, install packages corresponding to CUDA, CUDNN and TensorRT version.

Here, we use Torch1.8.0 inference on V100 and TensorRT 7.2 Cuda 10.2 Cudnn 8.0.2 on T4.

## 2. Reproduce speed

#### 2.1 Torch Inference on V100

To get inference speed without TensorRT on V100, you can run the following command:

```shell
python tools/eval.py --data data/coco.yaml --batch 32 --weights yolov6n.pt --task speed [--half]
```

- Speed results with batchsize = 1 are unstable in multiple runs, thus we do not provide the bs1 speed results.

#### 2.2 TensorRT Inference on T4

To get inference speed with TensorRT in FP16 mode on T4, you can follow the steps below:

First, export pytorch model as onnx format using the following command:

```shell
python deploy/ONNX/export_onnx.py --weights yolov6n.pt --device 0 --simplify --batch [1 or 32]
```

Second, generate an inference trt engine and test speed using `trtexec`:

```
trtexec --explicitBatch --fp16 --inputIOFormats=fp16:chw --outputIOFormats=fp16:chw --buildOnly --workspace=1024 --onnx=yolov6n.onnx --saveEngine=yolov6n.trt

trtexec --fp16 --avgRuns=1000 --workspace=1024 --loadEngine=yolov6n.trt
```
