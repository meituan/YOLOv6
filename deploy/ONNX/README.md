## Export ONNX Model

### Check requirements
```shell
pip install onnx>=1.10.0
```

### Export script
```shell
python deploy/ONNX/export_onnx.py --weights yolov6s.pt --img 640 --batch 1

```

### Download
* [YOLOv6-nano](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6n.onnx)
* [YOLOv6-tiny](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6t.onnx)
* [YOLOv6-s](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6s.onnx)
