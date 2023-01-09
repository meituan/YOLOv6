# Export ONNX Model

## Check requirements

```shell
pip install onnx >=1.10.0
```

## Export script

```shell
python ./deploy/ONNX/export_onnx.py \
  --weights yolov6s.pt \
  --img 640 \
  --batch 1 \
  --simplify
```



#### Description of all arguments

- `--weights` : The path of yolov6 model weights.
- `--img-size` : Image size of model inputs.
- `--batch-size` : Batch size of model inputs.
- `--half` : Whether to export half-precision model.
- `--inplace` : Whether to set Detect() inplace.
- `--simplify` : Whether to simplify onnx. Not support in end to end export.
- `--dynamic` : Whether to export dynamic axes onnx model. Only support all or batch .
- `--end2end` : Whether to export end to end onnx model. Only support onnxruntime and TensorRT >= 8.0.0 .
- `--trt-version` :  Export onnx for TensorRT version. Support : 7 or 8.
- `--ort` : Whether to export onnx for onnxruntime backend.
- `--with-preprocess` : Whether to export preprocess with bgr2rgb and normalize (divide by 255)
- `--topk-all` : Topk objects for every image.
- `--iou-thres` : IoU threshold for NMS algorithm.
- `--conf-thres` : Confidence threshold for NMS algorithm.
- `--device` : Export device. Cuda device : 0 or 0,1,2,3 ... , CPU : cpu .

## Download

* [YOLOv6-N](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6n.onnx)
* [YOLOv6-T](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6t.onnx)
* [YOLOv6-S](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6s.onnx)
* [YOLOv6-M](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6m.onnx)
* [YOLOv6-L-ReLU](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6l_relu.onnx)
* [YOLOv6-L](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6l.onnx)


## End2End export

Now YOLOv6 supports end to end detect for onnxruntime and TensorRT !

If you want to deploy in TensorRT, make sure you have installed TensorRT !

### onnxruntime backend

#### Usage

```bash
python ./deploy/ONNX/export_onnx.py \
  --weights yolov6s.pt \
  --img 640 \
  --batch 1 \
  --end2end \
  --ort
```

You will get an onnx with **NonMaxSuppression** operater .

### TensorRT backend (TensorRT version == 7.2.3.4)

#### Usage

```bash
python ./deploy/ONNX/export_onnx.py \
  --weights yolov6s.pt \
  --img 640 \
  --batch 1 \
  --end2end \
  --trt-version 7

```
You will get an onnx with **[BatchedNMSDynamic_TRT](https://github.com/triple-Mu/TensorRT/tree/main/plugin/batchedNMSPlugin)** plugin .


### TensorRT backend (TensorRT version>= 8.0.0)

#### Usage

```bash
python ./deploy/ONNX/export_onnx.py \
    --weights yolov6s.pt \
    --img 640 \
    --batch 1 \
    --end2end \
    --trt-version 8
```

You will get an onnx with **[EfficientNMS_TRT](https://github.com/NVIDIA/TensorRT/tree/main/plugin/efficientNMSPlugin)** plugin .

### Outputs Description

The onnx outputs are as shown :

<img src="https://user-images.githubusercontent.com/92794867/211291328-cbdb6835-2856-4a0d-ada6-ab76e584e804.png" height="200px" />

```num_dets``` means the number of object in every image in its batch .

```boxes``` means topk(100) object's location about [`x0`,`y0`,`x1`,`y1`] .

```scores``` means the confidence score of every topk(100) objects .

```labels``` means the category of every topk(100) objects .


You can export TensorRT engine use [trtexec](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-ovr) tools.

#### Usage

For both TensorRT-7 and TensorRT-8  `trtexec`  tool is avaiable.

```shell
trtexec --onnx=yolov6s.onnx \
  --saveEngine=yolov6s.engine \
  --workspace=8192 \ # 8GB
  --fp16 # if export TensorRT fp16 model
```

## Evaluate TensorRT model's performance

When we get the TensorRT model, we can evalute its performance by:

```bash
python ./deploy/ONNX/eval_trt.py \
  --weights yolov6s.engine \
  --batch-size 1 \
  --data data/coco.yaml
```

## Dynamic Axes Inference

### Dynamic Batch

You can export dynamic batch onnx model as follow:

```bash
python ./deploy/ONNX/export_onnx.py \
  --weights yolov6s.pt \
  --img 640 \
  --batch 1 \
  --dynamic batch
```

Dynamic batch and end2end inference examples:

[export ONNX model with dynamic batch ](YOLOv6-Dynamic-Batch-onnxruntime.ipynb)

[export TensorRT model with dynamic batch](YOLOv6-Dynamic-Batch-tensorrt.ipynb)


### Dynamic Batch Height Width

You can export dynamic axes of batch, height, and width onnx model as follow:

```bash
python ./deploy/ONNX/export_onnx.py \
  --weights yolov6s.pt \
  --img 640 \
  --batch 1 \
  --dynamic all
```
