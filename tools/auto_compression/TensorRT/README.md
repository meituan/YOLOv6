# TensorRT Python deployment

### Evaluation COCO mAP

- FP16
```shell
python trt_eval.py --onnx_model_file=yolov6s.onnx \
                   --precision_mode=fp16 \
                   --dataset_dir=dataset/coco/ \
                   --val_image_dir=val2017 \
                   --val_anno_path=annotations/instances_val2017.json
```

- INT8
```shell
python trt_eval.py --onnx_model_file=yolov6s_quant_onnx/quant_model.onnx \
                   --calibration_file=yolov6s_quant_onnx/calibration.cache \
                   --precision_mode=int8 \
                   --dataset_dir=dataset/coco/ \
                   --val_image_dir=val2017 \
                   --val_anno_path=annotations/instances_val2017.json
```

### Test a single image

- FP16
```shell
python trt_eval.py --onnx_model_file=yolov6s-tiny.onnx --image_file=image.jpg --precision_mode=fp16
```

- INT8
```shell
python trt_eval.py --onnx_model_file=yolov6s_quant_onnx/quant_model.onnx \
                   --calibration_file=yolov6s_quant_onnx/calibration.cache \
                   --image_file=image.jpg \
                   --precision_mode=int8
```
