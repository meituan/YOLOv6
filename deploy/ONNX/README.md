# Export ONNX Model

## Check requirements
```shell
pip install onnx>=1.10.0
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
- `--img` : Image size of model inputs.
- `--batch` : Batch size of model inputs.
- `--half` : Whether to export half-precision model.
- `--inplace` : Whether to set Detect() inplace.
- `--simplify` : Whether to simplify onnx. Not support in end to end export.
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

You will get an onnx with **NonMaxSuppression** operator .

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

<img src="https://user-images.githubusercontent.com/92794867/176650971-a4fa3d65-10d4-4b65-b8ef-00a2ff13406c.png" height="300px" />

```num_dets``` means the number of object in every image in its batch .

```det_boxes``` means topk(100) object's location about [`x0`,`y0`,`x1`,`y1`] .

```det_scores``` means the confidence score of every topk(100) objects .

```det_classes``` means the category of every topk(100) objects .


You can export TensorRT engine use [trtexec](https://docs.nvidia.com/deeplearning/tensorrt/developer-guide/index.html#trtexec-ovr) tools.
#### Usage
For both TensorRT-7 and TensorRT-8  `trtexec`  tool is avaiable.
``` shell
trtexec --onnx=yolov6s.onnx \
        --saveEngine=yolov6s.engine \
        --workspace=8192 # 8GB
        --fp16 # if export TensorRT fp16 model
```

## Evaluate TensorRT model's performance

When we get the TensorRT model, we can evalute its performance by:
```
python deploy/ONNX/eval_trt.py --weights yolov6s.engine --batch-size=1 --data data/coco.yaml
```

## Dynamic Batch Inference

YOLOv6 support dynamic batch export and inference, you can refer to:

[export ONNX model with dynamic batch ](YOLOv6-Dynamic-Batch-onnxruntime.ipynb)

[export TensorRT model with dynamic batch](YOLOv6-Dynamic-Batch-tensorrt.ipynb)
