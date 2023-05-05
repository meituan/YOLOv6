# Export NCNN Model By PNNX with TorchScript

## Export TorchScript

```shell
python ./deploy/NCNN/export_torchscript.py \
  --weights yolov6lite_s.pt \
  --img 320 320 \
  --batch 1
```

#### Description of all arguments

- `--weights` : The path of yolov6 model weights.
- `--img` : Image size of model inputs.
- `--batch` : Batch size of model inputs.
- `--device` : Export device. Cuda device : 0 or 0,1,2,3 ... , CPU : cpu .

## Export NCNN with TorchScript

- Download tools from [PNNX](https://github.com/pnnx/pnnx/releases)
- [Usage](https://github.com/triple-Mu/ncnn/blob/master/tools/pnnx/README.md)

  Unzip the `pnnx-YYYYMMDD-PLANTFORM.zip` and add the `pnnx` to your `PATH` .

  Then run the following command to export ncnn model :

  ```shell
  mkdir -p work_dir
  mv yolov6lite_s.torchscript work_dir
  cd work_dir
  pnnx yolov6lite_s.torchscript inputshape=[1,3,320,320]f32
  ```

  You will get `yolov6lite_s.ncnn.bin` and `yolov6lite_s.ncnn.param` in `work_dir` .

  If you want to try int8 quantization, you can get more information from [here](https://github.com/Tencent/ncnn/blob/master/docs/how-to-use-and-FAQ/quantized-int8-inference.md) .

## Run inference with NCNN-Python

```shell
python3 deploy/NCNN/infer-ncnn-model.py \
  data/images/image1.jpg \
  work_dir/yolov6lite_s.ncnn.param \
  work_dir/yolov6lite_s.ncnn.bin \
  --img-size 320 320 \
  --max-stride 64 \
  --show
```

#### Description of all arguments

- `img` : The path of image you want to detect.
- `param` : The NCNN param path.
- `bin` : The NCNN bin path.
- `--show` : Whether to show detection resulut.
- `--out-dir` : The output path to save detection result.
- `--img-size` : The image height and width for model input.
- `--max-stride` : The yolov6 lite model max stride.

***Notice!***

If you want to try norm yolov6 model such as `yolov6n/s/m/l`, you should add `--max-stride 32` flags .


## Download

* [YOLOv6-lite-s]()
* [YOLOv6-lite-m]()
* [YOLOv6-lite-l]()
* [YOLOv6-lite-l-320x192]()
* [YOLOv6-lite-l-224x128]()
* [YOLOv6-n]()
* [YOLOv6-s]()
* [YOLOv6-m]()
* [YOLOv6-l]()
