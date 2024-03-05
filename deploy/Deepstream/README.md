# YOLOv6 in Deepstream

## Dependencies
- Deepstream6.0
- TensorRT-8.0

## Step 1: Get the TensorRT 
Please Follow the file [TensorRT README](../TensorRT/README.md)  to get TensorRT engine file `yolov6n.trt`.

```shell
python ./deploy/ONNX/export_onnx.py \
    --weights yolov6n.pt \
    --img 640 \
    --batch 1
```

```shell
python3 onnx_to_tensorrt.py --fp16 --int8 -v \
        --max_calibration_size=${MAX_CALIBRATION_SIZE} \
        --calibration-data=${CALIBRATION_DATA} \
        --calibration-cache=${CACHE_FILENAME} \
        --preprocess_func=${PREPROCESS_FUNC} \
        --explicit-batch \
        --onnx ${ONNX_MODEL} -o ${OUTPUT}
```
### Example
```shell
python3 onnx_to_tensorrt.py --fp16 --onnx ${ONNX_MODEL} -o ${OUTPUT}
```

## Step 2: Build the So file 
Execute the following Command, get the libnvdsinfer_custom_impl_Yolov6.so file.
```shell
cd nvdsparsebbox_YoloV6
export CUDA_VER=11.4  # for dGPU  
make
```

## Step 3: Run the Demo
```shell
mv yolov6n.trt Deepstream # move yolov6n.trt file to Deepstream folder
deepstream-app -c ds_app_config_yoloV6.txt 
```

## Additional Experimental Results   

**The TensorRT performance of Yolov6n was tested on COCO2017 val datasets, as shown below.**

---------------------
|Yolov6N(640) FP16 |     AP       | area | maxDet |
|-----------------|------------------------|-------------|-----------------------|
|Average Precision|  (AP) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.357 |
|Average Precision|  (AP) @[ IoU=0.50      | area=   all | maxDets=100 ] = 0.512 |
|Average Precision|  (AP) @[ IoU=0.75      | area=   all | maxDets=100 ] = 0.383 |
|Average Precision|  (AP) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.170 |
|Average Precision|  (AP) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.398 |
|Average Precision|  (AP) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.511 |
|Average Recall   |  (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=  1 ] = 0.299 |
|Average Recall   |  (AR) @[ IoU=0.50:0.95 | area=   all | maxDets= 10 ] = 0.490 |
|Average Recall   |  (AR) @[ IoU=0.50:0.95 | area=   all | maxDets=100 ] = 0.538 |
|Average Recall   |  (AR) @[ IoU=0.50:0.95 | area= small | maxDets=100 ] = 0.296 |
|Average Recall   |  (AR) @[ IoU=0.50:0.95 | area=medium | maxDets=100 ] = 0.605 |
|Average Recall   |  (AR) @[ IoU=0.50:0.95 | area= large | maxDets=100 ] = 0.729 |