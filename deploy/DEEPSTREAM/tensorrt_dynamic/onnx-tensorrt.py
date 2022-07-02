from ast import arg
import numpy as np
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
# Utility functions
import utils.inference as inference_utils  # TRT/TF inference wrappers
import ctypes
import argparse


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--sofile', type=str, default='../nvdsinfer_custom_impl_Yolov6/libnvdsinfer_custom_impl_Yolov6.so', 
                            help='dynamic shared object file path')
    parser.add_argument('--onnx', type=str, default='../yolov6s.onnx', help='Yolov6 onnx file path')
    parser.add_argument('--engine', type=str, default='../yolov6s.trt', help='Yolov6 tensorrt engine file path')
    parser.add_argument('--img-size', nargs='+', type=int, default=[640, 640], help='image size')  # height, width
    parser.add_argument('--batch-size', type=int, default=1, help='batch size')
    parser.add_argument('--half', action='store_true', help='FP16 half-precision export')
    args = parser.parse_args()
    print(args)
    # add tensorrt plugin
    ctypes.cdll.LoadLibrary(args.sofile)
    # Precision command line argument -> TRT Engine datatype
    TRT_PRECISION_TO_DATATYPE = {
        16: trt.DataType.HALF,
        32: trt.DataType.FLOAT
    }
    # datatype: float 32
    trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[32]
    if(args.half):
        trt_engine_datatype = TRT_PRECISION_TO_DATATYPE[16]

    # ----------------------- static batch size -------------------- #
    max_batch_size = args.batch_size
    trt_inference_static_wrapper = inference_utils.TRTInference(
        args.engine, args.onnx,
        trt_engine_datatype, max_batch_size
    )
    input_data = np.ones((1, 3, args.img_size[0], args.img_size[1]), dtype=np.float32)
    output_shapes = [ (1, 85, int(args.img_size[0]/8), int(args.img_size[1]/8)), 
                      (1, 85, int(args.img_size[0]/16), int(args.img_size[1]/16)), 
                      (1, 85, int(args.img_size[0]/32), int(args.img_size[1]/32))]
    trt_outputs = trt_inference_static_wrapper.infer(input_data, output_shapes)
    
    print("output layer1 : {}".format(trt_outputs[0].shape))
    print("output layer2 : {}".format(trt_outputs[1].shape))
    print("output layer3 : {}".format(trt_outputs[2].shape))
       