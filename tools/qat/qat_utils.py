from tqdm import tqdm
import torch
import torch.nn as nn

from pytorch_quantization import nn as quant_nn
from pytorch_quantization import tensor_quant
from pytorch_quantization import calib
from pytorch_quantization.tensor_quant import QuantDescriptor

from tools.partial_quantization.utils import set_module, module_quant_disable

def collect_stats(model, data_loader, num_batches):
    """Feed data to the network and collect statistic"""

    # Enable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.disable_quant()
                module.enable_calib()
            else:
                module.disable()

    for i, (image, _, _, _) in tqdm(enumerate(data_loader), total=num_batches):
        image = image.float()/255.0
        model(image.cuda())
        if i >= num_batches:
            break

    # Disable calibrators
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            if module._calibrator is not None:
                module.enable_quant()
                module.disable_calib()
            else:
                module.enable()

def compute_amax(model, **kwargs):
    # Load Calib result
    for name, module in model.named_modules():
        if isinstance(module, quant_nn.TensorQuantizer):
            print(F"{name:40}: {module}")
            if module._calibrator is not None:
                #MinMaxCalib
                if isinstance(module._calibrator, calib.MaxCalibrator):
                    module.load_calib_amax()
                else:
                #HistogramCalib
                    module.load_calib_amax(**kwargs)
    model.cuda()

def ptq_calibrate(model, train_loader, cfg):
    model.eval()
    model.cuda()
    # It is a bit slow since we collect histograms on CPU
    with torch.no_grad():
        collect_stats(model, train_loader, cfg.ptq.calib_batches)
        compute_amax(model, method=cfg.ptq.histogram_amax_method, percentile=cfg.ptq.histogram_amax_percentile)

def qat_init_model_manu(model, cfg, args):
    # print(model)
    conv2d_weight_default_desc = tensor_quant.QUANT_DESC_8BIT_CONV2D_WEIGHT_PER_CHANNEL
    conv2d_input_default_desc = QuantDescriptor(num_bits=cfg.ptq.num_bits, calib_method=cfg.ptq.calib_method)

    convtrans2d_weight_default_desc = tensor_quant.QUANT_DESC_8BIT_CONVTRANSPOSE2D_WEIGHT_PER_CHANNEL
    convtrans2d_input_default_desc = QuantDescriptor(num_bits=cfg.ptq.num_bits, calib_method=cfg.ptq.calib_method)

    for k, m in model.named_modules():
        if 'proj_conv' in k:
            print("Skip Layer {}".format(k))
            continue
        if args.calib is True and cfg.ptq.sensitive_layers_skip is True:
            if k in cfg.ptq.sensitive_layers_list:
                print("Skip Layer {}".format(k))
                continue
        # print(k, m)
        if isinstance(m, nn.Conv2d):
            # print("in_channel = {}".format(m.in_channels))
            # print("out_channel = {}".format(m.out_channels))
            # print("kernel size = {}".format(m.kernel_size))
            # print("stride size = {}".format(m.stride))
            # print("pad size = {}".format(m.padding))
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_conv = quant_nn.QuantConv2d(in_channels,
                                              out_channels,
                                              kernel_size,
                                              stride,
                                              padding,
                                              quant_desc_input = conv2d_input_default_desc,
                                              quant_desc_weight = conv2d_weight_default_desc)
            quant_conv.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_conv.bias.data.copy_(m.bias.detach())
            else:
                quant_conv.bias = None
            set_module(model, k, quant_conv)
        elif isinstance(m, nn.ConvTranspose2d):
            # print("in_channel = {}".format(m.in_channels))
            # print("out_channel = {}".format(m.out_channels))
            # print("kernel size = {}".format(m.kernel_size))
            # print("stride size = {}".format(m.stride))
            # print("pad size = {}".format(m.padding))
            in_channels = m.in_channels
            out_channels = m.out_channels
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            quant_convtrans = quant_nn.QuantConvTranspose2d(in_channels,
                                                       out_channels,
                                                       kernel_size,
                                                       stride,
                                                       padding,
                                                       quant_desc_input = convtrans2d_input_default_desc,
                                                       quant_desc_weight = convtrans2d_weight_default_desc)
            quant_convtrans.weight.data.copy_(m.weight.detach())
            if m.bias is not None:
                quant_convtrans.bias.data.copy_(m.bias.detach())
            else:
                quant_convtrans.bias = None
            set_module(model, k, quant_convtrans)
        elif isinstance(m, nn.MaxPool2d):
            # print("kernel size = {}".format(m.kernel_size))
            # print("stride size = {}".format(m.stride))
            # print("pad size = {}".format(m.padding))
            # print("dilation = {}".format(m.dilation))
            # print("ceil mode = {}".format(m.ceil_mode))
            kernel_size = m.kernel_size
            stride = m.stride
            padding = m.padding
            dilation = m.dilation
            ceil_mode = m.ceil_mode
            quant_maxpool2d = quant_nn.QuantMaxPool2d(kernel_size,
                                                      stride,
                                                      padding,
                                                      dilation,
                                                      ceil_mode,
                                                      quant_desc_input = conv2d_input_default_desc)
            set_module(model, k, quant_maxpool2d)
        else:
            # module can not be quantized, continue
            continue

def skip_sensitive_layers(model, sensitive_layers):
    print('Skip sensitive layers...')
    for name, module in model.named_modules():
        if name in sensitive_layers:
            print(F"Disable {name}")
            module_quant_disable(model, name)
