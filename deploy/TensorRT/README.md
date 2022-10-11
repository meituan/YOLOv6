# YOLOv6-TensorRT in C++

## Dependencies
- TensorRT-8.2.3.0
- OpenCV-4.1.0



## Step 1: Get onnx model

Follow the file [ONNX README](../../tools/quantization/tensorrt/post_training/README.md) to convert the pt mdoel to onnx `yolov6n.onnx`.
**Now do't support end2end onnx model which include the nms plugin**
```shell
python ./deploy/ONNX/export_onnx.py \
    --weights yolov6n.pt \
    --img 640 \
    --batch 1
```

## Step 2: Prepare serialized engine file

Follow the file [post training README](../../tools/quantization/tensorrt/post_training/README.md) to convert and save the serialized engine file `yolov6.engine`.

```shell
python3 onnx_to_tensorrt.py --fp16 --int8 -v \
        --max_calibration_size=${MAX_CALIBRATION_SIZE} \
        --calibration-data=${CALIBRATION_DATA} \
        --calibration-cache=${CACHE_FILENAME} \
        --preprocess_func=${PREPROCESS_FUNC} \
        --explicit-batch \
        --onnx ${ONNX_MODEL} -o ${OUTPUT}
```

## Step 3: build the demo

Please follow the [TensorRT Installation Guide](https://docs.nvidia.com/deeplearning/tensorrt/install-guide/index.html) to install TensorRT.

And you should set the TensorRT path and CUDA path in CMakeLists.txt.

If you train your custom dataset, you may need to modify the value of `num_class, image width height, and class name`.

```c++
const int num_class = 80;
static const int INPUT_W = 640;
static const int INPUT_H = 640;
static const char* class_names[] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };
```

build the demo:

```shell
mkdir build
cd build
cmake ..
make
```

Then run the demo:

```shell
./yolov6 ../you.engine -i image_path
```
# Testing on image
You can do testing on images using .trt weights, just give path of image directory & its annotation path

```
python3 deploy/TensorRT/eval_yolo_trt.py -v -m model.trt \
--imgs-dir /workdir/datasets/coco/images/val2017 \
--annotations /workdir/datasets/coco/annotations/instances_val2017.json \
--conf-thres 0.40 --iou-thres 0.45
```
