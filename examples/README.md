# NOTICE ！！！
This PR only support TensorRT ≥ 8.0.0

Please update your TensorRT to the newest version

Trtexec tools is in /the/path/you/uncompress/TensorRT-8.X.X.X/bin/

Or /usr/src/tensorrt/bin if you install with deb file

Download the pretrain weight of yolov6s model into weights/
# TRT END2END
``` shell
python3 deploy/ONNX/export_onnx.py \
        --weights weights/yolov6s.pt  \
        --device 0 \
        --end2end
```
Then
``` shell
trtexec --onnx=weights/yolov6s.onnx --saveEngine=weights/yolov6s.engine
```
Infer
``` shell
cd examples
python3 trt_infer.py
```

# ORT END2END
``` shell
python3 deploy/ONNX/export_onnx.py \
        --weights weights/yolov6s.pt  \
        --device 0 \
        --end2end \
        --max-wh 6400
```
Infer
``` shell
cd examples
python3 ort_infer.py
```
