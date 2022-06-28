#!/usr/bin/env bash
cd ../
mkdir -p weights

# download official weights
wget https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.pt -P weights
# export yolov6s.onnx
python3 deploy/ONNX/export_onnx.py --weights weights/yolov6s.pt  --device 0 --end2end
mv weights/yolov6s.onnx ./examples/yolov6_nms.onnx
cd examples
trtexec --onnx=./yolov6s_nms.onnx --saveEngine=./yolov6s_nms_fp16.engine --fp16

# result test
wget https://oneflow-static.oss-cn-beijing.aliyuncs.com/tripleMu/image1.jpg
python3 trt_infer.py
