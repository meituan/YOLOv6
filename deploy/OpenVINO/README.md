## Export OpenVINO Model

### Check requirements
```shell
pip install --upgrade pip
pip install openvino-dev
```

### Export script
```shell
python deploy/OpenVINO/export_openvino.py --weights yolov6s.pt --img 640 --batch 1

```

### Download
* [YOLOv6-nano](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6n_openvino.tar.gz)
* [YOLOv6-tiny](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6n_openvino.tar.gz)
* [YOLOv6-s](https://github.com/meituan/YOLOv6/releases/download/0.1.0/yolov6n_openvino.tar.gz)

### Speed test
```shell
benchmark_app -m yolov6s_openvino/yolov6s.xml -i data/images/image1.jpg -d CPU -niter 100 -progress

```
