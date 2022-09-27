## Deepstream Inference

### check requirements
```shell
pip install onnx>=1.10.0
tensorrt>=7.2.2.3
pycuda>=2021.1
deepstream >=6.0
```

### Export ONNX file
```shell
python deploy/ONNX/export_onnx.py --weights yolov6s.pt --img 640 --batch 1 --tensorrt
```

### Modify in YOLOv6/deploy/ONNX/export_onnx.py
```python
line29:    parser.add_argument('--tensorrt', action='store_true', help='set Detect() tensorrt=True')
line59:    m.tensorrt = args.tensorrt
line67:    if(args.tensorrt):
            LOGGER.info('\n export ONNX file for tensorrt engine...')
            torch.onnx.export(model, img, export_file, verbose=False, opset_version=12,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=['images'],
                          output_names=['yololayer_002', 'yololayer_001', 'yololayer_000'],
                        #   output_names=['outputs'],
                         )
```

### Modify in YOLOv6/yolov6/models/effidehead.py
```python
line12:     def __init__(self, num_classes=80, anchors=1, num_layers=3, inplace=True, tensorrt=False, head_layers=None):  # detection layer
line26:     self.tensorrt = tensorrt
line76:         if(self.tensorrt):  # <---- add by yeah
                    z.append(y)
                    continue
line95:     if(self.tensorrt): return z         # <--- add by yeah
```

### Build TensorRT Engine 
```shell
# cp yolov6s.onnx to DEEPSTREAM folder
cp weights/yolov6s.onnx deploy/DEEPSTREAM

cd deploy/DEEPSTREAM/tensorrt_dynamic
python3 onnx-tensorrt.py --half   # generate yolov6s.trt with fp16
```

### Deepstream Inference
```shell
cd deploy/DEEPSTREAM/nvdsinfer_custom_impl_Yolov6
export CUDA_VER=11.4 # for dGPU   CUDA_VER=11.4 for Jetson
make -j32
cd ..
# video inference , output file: output_yolov6.mp4
deepstream-app -c deepstream_app_config_yoloV6.txt

```