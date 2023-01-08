import argparse
import time
import sys
import os
ROOT = os.getcwd()
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))
sys.path.append('../../')
from yolov6.models.effidehead import Detect
from yolov6.models.yolo import build_model
from yolov6.layers.common import *
from yolov6.utils.events import LOGGER, load_yaml
from yolov6.utils.checkpoint import load_checkpoint, load_state_dict
from yolov6.utils.config import Config
from tools.partial_quantization.eval import EvalerWrapper
from tools.partial_quantization.utils import get_module, concat_quant_amax_fuse
from tools.qat.qat_utils import qat_init_model_manu
from pytorch_quantization import nn as quant_nn
from onnx_utils import get_remove_qdq_onnx_and_cache

op_concat_fusion_list = [
    ('backbone.ERBlock_5.2.m', 'backbone.ERBlock_5.2.cv2.conv'),
    ('backbone.ERBlock_5.0.conv', 'neck.Rep_p4.conv1.conv', 'neck.upsample_feat0_quant'),
    ('backbone.ERBlock_4.0.conv', 'neck.Rep_p3.conv1.conv', 'neck.upsample_feat1_quant'),
    ('neck.upsample1.upsample_transpose', 'neck.Rep_n3.conv1.conv'),
    ('neck.upsample0.upsample_transpose', 'neck.Rep_n4.conv1.conv'),
    #
    ('detect.reg_convs.0.conv', 'detect.cls_convs.0.conv'),
    ('detect.reg_convs.1.conv', 'detect.cls_convs.1.conv'),
    ('detect.reg_convs.2.conv', 'detect.cls_convs.2.conv'),
]

def zero_scale_fix(model, device):

    for k, m in model.named_modules():
        # print(k, m)
        if isinstance(m, quant_nn.QuantConv2d) or \
            isinstance(m, quant_nn.QuantConvTranspose2d):
            # print(m)
            # print(m._weight_quantizer._amax)
            weight_amax = m._weight_quantizer._amax.detach().cpu().numpy()
            # print(weight_amax)
            print(k)
            ones = np.ones_like(weight_amax)
            print("zero scale number = {}".format(np.sum(weight_amax == 0.0)))
            weight_amax = np.where(weight_amax == 0.0, ones, weight_amax)
            m._weight_quantizer._amax.copy_(torch.from_numpy(weight_amax).to(device))
        else:
            # module can not be quantized, continue
            continue

# python3 qat_export.py --weights yolov6s_v2_reopt.pt --quant-weights yolov6s_v2_reopt_qat_43.0.pt --export-batch-size 1 --conf ../../configs/repopt/yolov6s_opt_qat.py
# python3 qat_export.py --weights v6s_t.pt --quant-weights yolov6t_v2_reopt_qat_40.1.pt --export-batch-size 1 --conf ../../configs/repopt/yolov6_tiny_opt_qat.py
# python3 qat_export.py --weights v6s_n.pt --quant-weights yolov6n_v2_reopt_qat_34.9.pt --export-batch-size 1 --conf ../../configs/repopt/yolov6n_opt_qat.py
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='./yolov6s_v2_reopt.pt', help='weights path')
    parser.add_argument('--quant-weights', type=str, default='./yolov6s_v2_reopt_qat_43.0.pt', help='calib weights path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--conf', type=str, default='../../configs/repopt/yolov6s_opt_qat.py', help='model config')
    parser.add_argument('--export-batch-size', type=int, default=None, help='export batch size')
    parser.add_argument('--calib', action='store_true', default=False, help='calibrated model')
    parser.add_argument('--scale-fix', action='store_true', help='enable scale fix')
    parser.add_argument('--fuse-bn', action='store_true', help='fuse bn')
    parser.add_argument('--graph-opt', action='store_true', help='enable graph optimizer')
    parser.add_argument('--inplace', action='store_true', help='set Detect() inplace=True')
    parser.add_argument('--end2end', action='store_true', help='export end2end onnx')
    parser.add_argument('--trt-version', type=int, default=8, help='tensorrt version')
    parser.add_argument('--with-preprocess', action='store_true', help='export bgr2rgb and normalize')
    parser.add_argument('--max-wh', type=int, default=None, help='None for tensorrt nms, int value for onnx-runtime nms')
    parser.add_argument('--topk-all', type=int, default=100, help='topk objects for every images')
    parser.add_argument('--iou-thres', type=float, default=0.45, help='iou threshold for NMS')
    parser.add_argument('--conf-thres', type=float, default=0.4, help='conf threshold for NMS')
    parser.add_argument('--device', default='0', help='cuda device, i.e. 0 or 0, 1, 2, 3 or cpu')
    parser.add_argument('--eval-yaml', type=str, default='../partial_quantization/eval.yaml', help='evaluation config')
    args = parser.parse_args()
    args.img_size *= 2 if len(args.img_size) == 1 else 1  # expand
    print(args)
    t = time.time()
    # Check device
    cuda = args.device != 'cpu' and torch.cuda.is_available()
    device = torch.device('cuda:0' if cuda else 'cpu')
    assert not (device.type == 'cpu' and args.half), '--half only compatible with GPU export, i.e. use --device 0'
    model = load_checkpoint(args.weights, map_location=device, inplace=args.inplace, fuse=args.fuse_bn)
    yolov6_evaler = EvalerWrapper(eval_cfg=load_yaml(args.eval_yaml))
    # orig_mAP = yolov6_evaler.eval(model)
    for layer in model.modules():
        if isinstance(layer, RepVGGBlock):
            layer.switch_to_deploy()
    for k, m in model.named_modules():
        if isinstance(m, Conv):  # assign export-friendly activations
            if isinstance(m.act, nn.SiLU):
                m.act = SiLU()
        elif isinstance(m, Detect):
            m.inplace = args.inplace
    # Load PyTorch model
    cfg = Config.fromfile(args.conf)
    # init qat model
    qat_init_model_manu(model, cfg, args)
    print(model)
    model.neck.upsample_enable_quant(cfg.ptq.num_bits, cfg.ptq.calib_method)
    ckpt = torch.load(args.quant_weights)
    model.load_state_dict(ckpt['model'].float().state_dict())
    print(model)
    model.to(device)
    if args.scale_fix:
        zero_scale_fix(model, device)
    if args.graph_opt:
        # concat amax fusion
        for sub_fusion_list in op_concat_fusion_list:
            ops = [get_module(model, op_name) for op_name in sub_fusion_list]
            concat_quant_amax_fuse(ops)
    qat_mAP = yolov6_evaler.eval(model)
    print(qat_mAP)
    if args.end2end:
        from yolov6.models.end2end import End2End
        model = End2End(model, max_obj=args.topk_all, iou_thres=args.iou_thres,score_thres=args.conf_thres,
                        max_wh=args.max_wh, device=device, trt_version=args.trt_version, with_preprocess=args.with_preprocess)
    # ONNX export
    quant_nn.TensorQuantizer.use_fb_fake_quant = True
    if args.export_batch_size is None:
        img = torch.zeros(1, 3, *args.img_size).to(device)
        export_file = args.quant_weights.replace('.pt', '_dynamic.onnx')  # filename
        if args.graph_opt:
            export_file = export_file.replace('.onnx', '_graph_opt.onnx')
        if args.end2end:
            export_file = export_file.replace('.onnx', '_e2e.onnx')
        dynamic_axes = {
            "image_arrays": {0: "batch"},
        }
        if args.end2end:
            dynamic_axes["num_dets"] = {0: "batch"}
            dynamic_axes["det_boxes"] = {0: "batch"}
            dynamic_axes["det_scores"] = {0: "batch"}
            dynamic_axes["det_classes"] = {0: "batch"}
        else:
            dynamic_axes["outputs"] = {0: "batch"}
        torch.onnx.export(model,
                          img,
                          export_file,
                          verbose=False,
                          opset_version=13,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=['images'],
                          output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                          if args.end2end else ['outputs'],
                          dynamic_axes=dynamic_axes
                         )
    else:
        img = torch.zeros(args.export_batch_size, 3, *args.img_size).to(device)
        export_file = args.quant_weights.replace('.pt', '_bs{}.onnx'.format(args.export_batch_size))  # filename
        if args.graph_opt:
            export_file = export_file.replace('.onnx', '_graph_opt.onnx')
        if args.end2end:
            export_file = export_file.replace('.onnx', '_e2e.onnx')
        torch.onnx.export(model,
                          img,
                          export_file,
                          verbose=False,
                          opset_version=13,
                          training=torch.onnx.TrainingMode.EVAL,
                          do_constant_folding=True,
                          input_names=['images'],
                          output_names=['num_dets', 'det_boxes', 'det_scores', 'det_classes']
                          if args.end2end else ['outputs'],
                          )

    get_remove_qdq_onnx_and_cache(export_file)
