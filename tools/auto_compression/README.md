# YOLOv6 Quantization Compression Example

This example uses [ACT](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/auto_compression) from [PaddleSlim](https://github.com/PaddlePaddle/PaddleSlim) for YOLOv6 quantization.
The quantized model can be deployed on TensorRT.

- Benchmark

| Model | Base mAP<sup>val<br>0.5:0.95  | Quant mAP<sup>val<br>0.5:0.95 | Latency<sup><small>FP32</small><sup><br><sup> | Latency<sup><small>FP16</small><sup><br><sup> | Latency<sup><small>INT8</small><sup><br><sup> | Model |
| :-------- |:-------- |:--------: | :--------: | :---------------------: | :----------------: | :----------------: |
| YOLOv6s |  42.4   | 41.3  |  9.06ms  |   2.90ms   |  **1.83ms**  | [ONNX](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.onnx) &#124; [Quant ONNX](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov6s_quant_onnx.tar) |
| YOLOv6-Tiny  |  40.8   | 39.0 |  5.06ms  |   2.32ms   |  **1.68ms** | [ONNX](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6t.onnx) &#124; [Quant ONNX](https://bj.bcebos.com/v1/paddle-slim-models/act/yolov6_tiny_quant_onnx.tar) |

Note:
- The mAP are evaluated in the COCO val2017, and fixed 640*640 size.
- The test environment is: Tesla T4, TensorRT 8.4.1, and batch_size=1.

### Experiment

You also can enter the [YOLOv6_quantization_act.ipynb](./YOLOv6_quantization_act.ipynb) to experiment.

#### 1. Environment Dependencies Installation

- paddlepaddle>=2.3.2
- paddleslim>=2.3.4
- pycocotools

```shell
# Take Ubuntu and CUDA 11.2 as an example for GPU, and other environments can be installed directly according to Paddle's official website.
#  https://www.paddlepaddle.org.cn/install/quick?docurl=/documentation/docs/zh/install/pip/linux-pip.html
python -m pip install paddlepaddle-gpu==2.3.2.post112 -f https://www.paddlepaddle.org.cn/whl/linux/mkl/avx/stable.html

# CPU
#pip install paddlepaddle==2.3.2

pip install paddleslim==2.3.4
```

#### 2. Prepare the dataset

**Choose one of (1) or (2) to prepare the data.**

- (1) Support unlabeled pictures, directly set the picture folder, but does not support the mAP evaluation.

  Modify the `image_path` path in [config](./configs) of the real prediction scenario.
  ```yaml
  Global:
    image_path: dataset/coco/val2017
  ```

- (2) Support loading COCO format dataset, **support the mAP evaluation.**

  You can download [Train](http://images.cocodataset.org/zips/train2017.zip), [Val](http://images.cocodataset.org/zips/val2017.zip), [annotation](http://images.cocodataset.org/annotations/annotations_trainval2017.zip).

  The directory format is as follows:
  ```
  dataset/coco/
  ├── annotations
  │   ├── instances_train2017.json
  │   ├── instances_val2017.json
  │   |   ...
  ├── train2017
  │   ├── 000000000009.jpg
  │   ├── 000000580008.jpg
  │   |   ...
  ├── val2017
  │   ├── 000000000139.jpg
  │   ├── 000000000285.jpg
  ```

  If it is a custom data set, please prepare the data according to the above COCO data format.

  After preparing the dataset, modify the `coco_dataset_dir` path in [config](./configs).
  ```yaml
  Global:
    coco_dataset_dir: dataset/coco/
    coco_train_image_dir: train2017
    coco_train_anno_path: annotations/instances_train2017.json
    coco_val_image_dir: val2017
    coco_val_anno_path: annotations/instances_val2017.json
  ```


#### 3. Prepare the ONNX model

The ONNX model can be prepared through the [Export Tutorial](https://github.com/meituan/YOLOv6/blob/main/deploy/ONNX/README.md). You can also download [yolov6s.onnx](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.onnx).

The model without NMS is currently supported, and the end2end model will be supported in the future.


#### 4. Auto compression

- Single card training:
```
export CUDA_VISIBLE_DEVICES=0
python run.py --config_path=./configs/yolov6s_qat_dis.yaml --save_dir='./output/'
```

- Distributed training:
```
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m paddle.distributed.launch --log_dir=log --gpus 0,1,2,3 run.py \
          --config_path=./configs/yolov6s_qat_dis.yaml --save_dir='./output/'
```


#### 5. Deployment

After executing the program, output files will be generated in the output folder as shown below:
```shell
├── model.pdiparams         # Paddle prediction model weights
├── model.pdmodel           # Paddle prediction model file
├── calibration_table.txt   # Paddle calibration table after quantification
├── ONNX
│   ├── quant_model.onnx      # ONNX model after quantization
│   ├── calibration.cache     # TensorRT can directly load the calibration table
```

##### Deploy with TensorRT

Load `quant_model.onnx` and `calibration.cache`, you can directly use the TensorRT test script to verify, the detailed code can refer to [TensorRT deployment](/TensorRT)

- Python test：
```shell
cd TensorRT
python trt_eval.py --onnx_model_file=output/ONNX/quant_model.onnx \
                   --calibration_file=output/ONNX/calibration.cache \
                   --image_file=image.jpg \
                   --precision_mode=int8
```

- Speed test
```shell
trtexec --onnx=output/ONNX/quant_model.onnx --avgRuns=1000 --workspace=1024 --calib=output/ONNX/calibration.cache --int8
```

##### Deploy with Paddle TensorRT

- Python test:

First install the [PaddlePaddle with TensorRT](https://www.paddlepaddle.org.cn/inference/v2.3/user_guides/download_lib.html#python).

Then use [paddle_trt_infer.py](./paddle_trt_infer.py) to deploy:
```shell
python paddle_trt_infer.py --model_path=output --image_file=image.jpg --benchmark=True --run_mode=trt_int8
```

#### 6.FAQ

- If you want to quantize the model with PTQ, you can use the [YOLOv6 PTQ example](https://github.com/PaddlePaddle/PaddleSlim/tree/develop/example/post_training_quantization/pytorch_yolo_series).
