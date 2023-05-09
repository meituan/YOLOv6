try:
    from ppq.core.config import PPQ_CONFIG
    if PPQ_CONFIG.VERSION < '0.6.6':
        raise ValueError('为了运行该脚本的内容，你必须安装更高版本的 PPQ(>0.6.6)')

    import ppq.lib as PFL
    from ppq import TargetPlatform, TorchExecutor, graphwise_error_analyse
    from ppq.api import ENABLE_CUDA_KERNEL
    from ppq.api.interface import load_onnx_graph
    from ppq.core import (QuantizationPolicy, QuantizationProperty,
                          RoundingPolicy)
    from ppq.IR import Operation
    from ppq.quantization.optim import (LearnedStepSizePass,
                                        ParameterBakingPass,
                                        ParameterQuantizePass,
                                        QuantAlignmentPass, QuantizeFusionPass,
                                        QuantizeSimplifyPass,
                                        RuntimeCalibrationPass)

except ImportError:
    raise Exception('为了运行脚本内容，你必须安装 PPQ 量化工具(https://github.com/openppl-public/ppq)')
from typing import List

import torch

# ------------------------------------------------------------
# 在这个例子中我们将向你展示如何使用 INT8 量化一个 Yolo v6 模型
# 我们使用随机数据进行量化，这并不能得到好的量化结果。
# 在量化你的网络时，你应当使用真实数据和正确的预处理。
#
# 根据你选取的目标平台，PPQ 可以为 TensorRT, Openvino, Ncnn 等诸多平台生成量化模型
# ------------------------------------------------------------
graph = load_onnx_graph(onnx_import_file='Models/det_model/yolov6s.onnx')
dataset = [torch.rand(size=[1, 3, 640, 640]) for _ in range(64)]

# -----------------------------------------------------------
# 我们将借助 PFL - PPQ Foundation Library, 即 PPQ 基础类库完成量化
# 这是 PPQ 自 0.6.6 以来推出的新的量化 api 接口，这一接口是提供给
# 算法工程师、部署工程师、以及芯片研发人员使用的，它更为灵活。
# 我们将手动使用 Quantizer 完成算子量化信息初始化, 并且手动完成模型的调度工作
#
# 在开始之前，我需要向你介绍量化器、量化信息以及调度表
# 量化信息在 PPQ 中是由 TensorQuantizationConfig(TQC) 进行描述的
# 这个结构体描述了我要如何去量化一个数据，其中包含了量化位宽、量化策略、
# 量化 Scale, offset 等内容。
# ------------------------------------------------------------
from ppq import TensorQuantizationConfig as TQC

MyTQC = TQC(
    policy = QuantizationPolicy(
        QuantizationProperty.SYMMETRICAL +
        QuantizationProperty.LINEAR +
        QuantizationProperty.PER_TENSOR),
    rounding=RoundingPolicy.ROUND_HALF_EVEN,
    num_of_bits=8, quant_min=-128, quant_max=127,
    exponent_bits=0, channel_axis=None,
    observer_algorithm='minmax'
)
# ------------------------------------------------------------
# 作为示例，我们创建了一个 "线性" "对称" "Tensorwise" 的量化信息
# 这三者皆是该量化信息的 QuantizationPolicy 的一部分
# 同时要求该量化信息使用 ROUND_HALF_EVEN 方式进行取整
# 量化位宽为 8 bit，其中指数部分为 0 bit
# 量化上限为 127.0，下限则为 -128.0
# 这是一个 Tensorwise 的量化信息，因此 channel_axis = None
# observer_algorithm 表示在未来使用 minmax calibration 方法确定该量化信息的 scale

# 上述例子完成了该 TQC 的初始化，但并未真正启用该量化信息
# MyTQC.scale, MyTQC.offset 仍然为空，它们必须经过 calibration 才会具有有意义的值
# 并且他目前的状态 MyTQC.state 仍然是 Quantization.INITIAL，这意味着在计算时该 TQC 并不会参与运算。
# ------------------------------------------------------------

# ------------------------------------------------------------
# 接下来我们向你介绍量化器，这是 PPQ 中的一个核心类型
# 它的职责是为网络中所有处于量化区的算子初始化量化信息(TQC)
# PPQ 中实现了一堆不同的量化器，它们分别适配不同的情形
# 在这个例子中，我们分别创建了 TRT_INT8, GRAPHCORE_FP8, TRT_FP8 三种不同的量化器
# 由它们所生成的量化信息是不同的，为此你可以访问它们的源代码
# 位于 ppq.quantization.quantizer 中，查看它们初始化量化信息的逻辑。
# ------------------------------------------------------------
_ = PFL.Quantizer(platform=TargetPlatform.TRT_FP8, graph=graph)          # 取得 TRT_FP8 所对应的量化器
_ = PFL.Quantizer(platform=TargetPlatform.GRAPHCORE_FP8, graph=graph)    # 取得 GRAPHCORE_FP8 所对应的量化器
quantizer = PFL.Quantizer(platform=TargetPlatform.TRT_INT8, graph=graph) # 取得 TRT_INT8 所对应的量化器

# ------------------------------------------------------------
# 调度器是 PPQ 中另一核心类型，它负责切分计算图
# 在量化开始之前，你的计算图将被切分成可量化区域，以及不可量化区域
# 不可量化区域往往就是那些执行 Shape 推断的算子所构成的子图
# *** 量化器只为量化区的算子初始化量化信息 ***
# 调度信息将被写在算子的属性中，你可以通过 op.platform 来访问每一个算子的调度信息
# ------------------------------------------------------------
dispatching = PFL.Dispatcher(graph=graph).dispatch(                       # 生成调度表
    quant_types=quantizer.quant_operation_types)

for op in graph.operations.values():
    # quantize_operation - 为算子初始化量化信息，platform 传递了算子的调度信息
    # 如果你的算子被调度到 TargetPlatform.FP32 上，则该算子不量化
    # 你可以手动修改调度信息
    dispatching['Op1'] = TargetPlatform.FP32        # 将 Op1 强行送往非量化区
    dispatching['Op2'] = TargetPlatform.TRT_INT8    # 将 Op2 强行送往量化区
    quantizer.quantize_operation(
        op_name = op.name, platform = dispatching[op.name])

# ------------------------------------------------------------
# 在创建量化管线之前，我们需要初始化执行器，它用于模拟硬件并执行你的网络
# 请注意，执行器需要对网络结果进行分析并缓存分析结果，如果你的网络结构发生变化
# 你必须重新建立新的执行器。在上一步操作中，我们对算子进行了量化，这使得
# 普通的算子被量化算子替代，这一步操作将会改变网络结构。因此我们必须在其后建立执行器。
# ------------------------------------------------------------
collate_fn = lambda x: x.cuda()
executor = TorchExecutor(graph=graph, device='cuda')
executor.tracing_operation_meta(inputs=collate_fn(dataset[0]))
executor.load_graph(graph=graph)

# ------------------------------------------------------------
# 如果在你的模型中存在 NMS 算子 ———— PPQ 不知道如何计算这个玩意，但它跟量化也没啥关系
# 因此你可以注册一个假的 NMS forward 函数给 PPQ，帮助我们完成网络的前向传播流程
# ------------------------------------------------------------
from ppq.api import register_operation_handler
def nms_forward_function(op: Operation, values: List[torch.Tensor], **kwards) -> List[torch.Tensor]:
    return (
        torch.zeros([1, 1], dtype=torch.int32).cuda(),
        torch.zeros([1, 100, 4],dtype=torch.float32).cuda(),
        torch.zeros([1, 100],dtype=torch.float32).cuda(),
        torch.zeros([1, 100], dtype=torch.int32).cuda()
    )
register_operation_handler(nms_forward_function, 'EfficientNMS_TRT', platform=TargetPlatform.FP32)

# ------------------------------------------------------------
# 下面的过程将创建量化管线，它还是一个 PPQ 的核心类型
# 在 PPQ 中，模型的量化是由一个一个的量化过程(QuantizationOptimizationPass)完成的
# 量化管线 是 量化过程 的集合，在其中的量化过程将被逐个调用
# 从而实现对 TQC 中内容的修改，最终实现模型的量化
# 在这里我们为管线中添加了 7 个量化过程，分别处理不同的内容

# QuantizeSimplifyPass - 用于移除网络中的冗余量化信息
# QuantizeFusionPass - 用于调整量化信息状态，从而模拟推理图融合
# ParameterQuantizePass - 用于为模型中的所有参数执行 Calibration, 生成它们的 scale，并将对应 TQC 的状态调整为 ACTIVED
# RuntimeCalibrationPass - 用于为模型中的所有激活执行 Calibration, 生成它们的 scale，并将对应 TQC 的状态调整为 ACTIVED
# QuantAlignmentPass - 用于执行 concat, add, sum, sub, pooling 算子的定点对齐
# LearnedStepSizePass - 用于训练微调模型的权重，从而降低量化误差
# ParameterBakingPass - 用于执行模型参数烘焙

# 在 PPQ 中我们提供了数十种不同的 QuantizationOptimizationPass
# 你可以组合它们从而实现自定义的功能，也可以继承 QuantizationOptimizationPass 基类
# 从而创造出新的量化优化过程
# ------------------------------------------------------------
pipeline = PFL.Pipeline([
    QuantizeSimplifyPass(),
    QuantizeFusionPass(
        activation_type=quantizer.activation_fusion_types),
    ParameterQuantizePass(),
    RuntimeCalibrationPass(),
    QuantAlignmentPass(force_overlap=True),
    LearnedStepSizePass(
         steps=1000, is_scale_trainable=True,
        lr=1e-5, block_size=4, collecting_device='cuda'),
    ParameterBakingPass()
])

with ENABLE_CUDA_KERNEL():
    # 调用管线完成量化
    pipeline.optimize(
        graph=graph, dataloader=dataset, verbose=True,
        calib_steps=32, collate_fn=collate_fn, executor=executor)

    # 执行量化误差分析
    graphwise_error_analyse(
        graph=graph, running_device='cuda',
        dataloader=dataset, collate_fn=collate_fn)

# ------------------------------------------------------------
# 在最后，我们导出计算图
# 同样地，我们根据不同推理框架的需要，写了一堆不同的网络导出逻辑
# 你通过参数 platform 告诉 PPQ 你的模型最终将部署在何处，
# PPQ 则会返回一个对应的 GraphExporter 对象，它将负责将 PPQ 的量化信息
# 翻译成推理框架所需的内容。你也可以自己写一个 GraphExporter 类并注册到 PPQ 框架中来。
# ------------------------------------------------------------
exporter = PFL.Exporter(platform=TargetPlatform.TRT_INT8)
exporter.export(file_path='Quantized.onnx', config_path='Quantized.json', graph=graph)

# ------------------------------------------------------------
# 导出所需的 onnx 和 json 文件之后，你可以调用在这个文件旁边的 write_qparams_onnx2trt.py 生成 engine
#
# 你需要注意到，我们生成的 onnx 和 json 文件是可以随时迁移的，但 engine 一旦编译完成则不能迁移
# https://github.com/openppl-public/ppq/blob/master/md_doc/deploy_trt_by_OnnxParser.md
#
# 性能分析脚本 https://github.com/openppl-public/ppq/blob/master/ppq/samples/TensorRT/Example_Profiling.py
# ------------------------------------------------------------
