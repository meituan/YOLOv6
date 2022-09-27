# Quantization-Aware Training

As of v0.2.0 release, traditional post-training quantization (PTQ) produces a degraded performance of `YOLOv6-S` from 43.4% to 41.2%. This is however much improved compared with v0.1.0 since the most sensitve layers are removed. Yet it is not ready for deployment. Meanwhile, due to the inconsistency of reparameterization blocks during training and inference, quantization-aware training (QAT) cannot be directly integrated into YOLOv6. As a remedy, we first train a single-branch network called `YOLOv6-S-RepOpt` with [RepOptimizer](https://arxiv.org/pdf/2205.15242.pdf). It reaches 43.1% mAP and is very close to YOLOv6-S. We then apply our quantization strategy on `YOLOv6-S-RepOpt`.

We apply post-training quantization to `YOLOv6-S-RepOpt`, and its mAP slightly drops by 0.5%. Hence it is necessary to use QAT to further improve the accuracy. Besides, we involve   **channel-wise distillation** to accelerate the convergence. We finally reach a quantized model at 43.0% mAP.

To deploy the quantized model on typical NVIDIA GPUs (e.g. T4), we export the model to the ONNX format, then we use TensorRT to build a serialized engine along with the computed scale cache. The performance arrives at **43.3% mAP**, only 0.1% left to match the fully float precision of `YOLOv6-S`.


## Pre-requirements

It is required to install `pytorch_quantization`, on top of which we build our quantization strategy.

```python
pip install --extra-index-url=https://pypi.ngc.nvidia.com --trusted-host pypi.ngc.nvidia.com nvidia-pyindex
pip install --extra-index-url=https://pypi.ngc.nvidia.com --trusted-host pypi.ngc.nvidia.com pytorch_quantization
```

## Training with RepOptimizer
Firstly, train a `YOLOv6-S RepOpt` as follows, or download our realeased [checkpoint](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6s_v2_reopt.pt) and [scales](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6s_v2_scale.pt).
*  [Tutorial of RepOpt for YOLOv6](https://github.com/meituan/YOLOv6/blob/main/docs/tutorial_repopt.md)
## PTQ
We perform PTQ to get the range of activations and weights.
```python
CUDA_VISIBLE_DEVICES=0 python tools/train.py \
       --data ./data/coco.yaml \
       --output-dir ./runs/opt_train_v6s_ptq \
       --conf configs/repopt/yolov6s_opt_qat.py \
       --quant \
       --calib \
       --batch 32 \
       --workers 0
```

## QAT

Our proposed QAT strategy comes with channel-wise distillation. It loades calibrated ReOptimizer-trained model and trains for 10 epochs. To reproduce the result,

```python
CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python -m torch.distributed.launch --nproc_per_node=8 \
       tools/train.py \
       --data ./data/coco.yaml \
       --output-dir ./runs/opt_train_v6s_qat  \
       --conf configs/repopt/yolov6s_opt_qat.py \
       --quant \
       --distill \
       --distill_feat \
       --batch 128 \
       --epochs 10 \
       --workers 32 \
       --teacher_model_path ./assets/yolov6s_v2_reopt_43.1.pt \
       --device 0,1,2,3,4,5,6,7
```
## ONNX Export
To export to ONNX,
```python
python3 qat_export.py --weights yolov6s_v2_reopt_43.1.pt --quant-weights yolov6s_v2_reopt_qat_43.0.pt --graph-opt --export-batch-size 1
```

## TensorRT Deployment

To build a TRT engine,

```python
trtexec --workspace=1024 --percentile=99 --streams=1 --int8 --fp16 --avgRuns=10 --onnx=yolov6s_v2_reopt_qat_43.0_bs1.sim.onnx --calib=yolov6s_v2_reopt_qat_43.0_remove_qdq_bs1_calibration_addscale.cache --saveEngine=yolov6s_v2_reopt_qat_43.0_bs1.sim.trt
```
You can directly build engine with [yolov6s_v2_quant.onnx](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6s_v2_reopt_qat_43.0_remove_qdq_bs1.sim.onnx) and [yolov6s_v2_calibration.cache](https://github.com/meituan/YOLOv6/releases/download/0.2.0/yolov6s_v2_reopt_qat_43.0_remove_qdq_bs1_calibration_addscale.cache)

## Performance Comparison

We release our quantized and graph-optimized YOLOv6-S (v0.2.0) model. The following throughput is tested with TensorRT 8.4 on a NVIDIA Tesla T4 GPU.

| Model           | Size        | Precision        |mAP<sup>val<br/>0.5:0.95 | Speed<sup>T4<br/>trt b1 <br/>(fps) | Speed<sup>T4<br/>trt b32 <br/>(fps) |
| :-------------- | ----------- | ----------- |:----------------------- | ---------------------------------------- | -----------------------------------|
| [**YOLOv6-S RepOpt**] | 640 | INT8         |43.3                     | 619     | 924  |
| [**YOLOv6-S**] | 640         | FP16         |43.4                     | 377    | 541 |
| [**YOLOv6-T RepOpt**] | 640 | INT8         |39.8                     | 741     | 1167  |
| [**YOLOv6-T**] | 640         | FP16         |40.3                     | 449    | 659 |
| [**YOLOv6-N RepOpt**] | 640 | INT8         |34.8                     | 1114     | 1828  |
| [**YOLOv6-N**] | 640         | FP16         |35.9                     | 802    | 1234 |
