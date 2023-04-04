# onnx_to_tensorrt.py
#
# Copyright 1993-2019 NVIDIA Corporation.  All rights reserved.
#
# NOTICE TO LICENSEE:
#
# This source code and/or documentation ("Licensed Deliverables") are
# subject to NVIDIA intellectual property rights under U.S. and
# international Copyright laws.
#
# These Licensed Deliverables contained herein is PROPRIETARY and
# CONFIDENTIAL to NVIDIA and is being provided under the terms and
# conditions of a form of NVIDIA software license agreement by and
# between NVIDIA and Licensee ("License Agreement") or electronically
# accepted by Licensee.  Notwithstanding any terms or conditions to
# the contrary in the License Agreement, reproduction or disclosure
# of the Licensed Deliverables to any third party without the express
# written consent of NVIDIA is prohibited.
#
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, NVIDIA MAKES NO REPRESENTATION ABOUT THE
# SUITABILITY OF THESE LICENSED DELIVERABLES FOR ANY PURPOSE.  IT IS
# PROVIDED "AS IS" WITHOUT EXPRESS OR IMPLIED WARRANTY OF ANY KIND.
# NVIDIA DISCLAIMS ALL WARRANTIES WITH REGARD TO THESE LICENSED
# DELIVERABLES, INCLUDING ALL IMPLIED WARRANTIES OF MERCHANTABILITY,
# NONINFRINGEMENT, AND FITNESS FOR A PARTICULAR PURPOSE.
# NOTWITHSTANDING ANY TERMS OR CONDITIONS TO THE CONTRARY IN THE
# LICENSE AGREEMENT, IN NO EVENT SHALL NVIDIA BE LIABLE FOR ANY
# SPECIAL, INDIRECT, INCIDENTAL, OR CONSEQUENTIAL DAMAGES, OR ANY
# DAMAGES WHATSOEVER RESULTING FROM LOSS OF USE, DATA OR PROFITS,
# WHETHER IN AN ACTION OF CONTRACT, NEGLIGENCE OR OTHER TORTIOUS
# ACTION, ARISING OUT OF OR IN CONNECTION WITH THE USE OR PERFORMANCE
# OF THESE LICENSED DELIVERABLES.
#
# U.S. Government End Users.  These Licensed Deliverables are a
# "commercial item" as that term is defined at 48 C.F.R. 2.101 (OCT
# 1995), consisting of "commercial computer software" and "commercial
# computer software documentation" as such terms are used in 48
# C.F.R. 12.212 (SEPT 1995) and is provided to the U.S. Government
# only as a commercial end item.  Consistent with 48 C.F.R.12.212 and
# 48 C.F.R. 227.7202-1 through 227.7202-4 (JUNE 1995), all
# U.S. Government End Users acquire the Licensed Deliverables with
# only those rights set forth herein.
#
# Any use of the Licensed Deliverables in individual and commercial
# software must include, in the user documentation and internal
# comments to the code, the above Disclaimer and U.S. Government End
# Users Notice.
#
from __future__ import print_function

import argparse
import traceback
import sys
import tensorrt as trt

MAX_BATCH_SIZE = 1

def build_engine_from_onnx(model_name,
                           dtype,
                           verbose=False,
                           int8_calib=False,
                           calib_loader=None,
                           calib_cache=None,
                           fp32_layer_names=[],
                           fp16_layer_names=[],
                           ):
    """Initialization routine."""
    if dtype == "int8":
        t_dtype = trt.DataType.INT8
    elif dtype == "fp16":
        t_dtype = trt.DataType.HALF
    elif dtype == "fp32":
        t_dtype = trt.DataType.FLOAT
    else:
        raise ValueError("Unsupported data type: %s" % dtype)

    if trt.__version__[0] < '8':
        print('Exit, trt.version should be >=8. Now your trt version is ', trt.__version__[0])

    network_flags = 1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
    if dtype == "int8" and calib_loader is None:
        print('QAT enabled!')
        network_flags = network_flags | (1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION))

    """Build a TensorRT engine from ONNX"""
    TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE) if verbose else trt.Logger()
    with trt.Builder(TRT_LOGGER) as builder, builder.create_network(flags=network_flags) as network, \
            trt.OnnxParser(network, TRT_LOGGER) as parser:
        with open(model_name, 'rb') as model:
            if not parser.parse(model.read()):
                print('ERROR: ONNX Parse Failed')
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                    return None

        print('Building an engine.  This would take a while...')
        print('(Use "--verbose" or "-v" to enable verbose logging.)')
        config = builder.create_builder_config()
        config.max_workspace_size = 2 << 30
        if t_dtype == trt.DataType.HALF:
            config.flags |= 1 << int(trt.BuilderFlag.FP16)

        if t_dtype == trt.DataType.INT8:
            print('trt.DataType.INT8')
            config.flags |= 1 << int(trt.BuilderFlag.INT8)
            config.flags |= 1 << int(trt.BuilderFlag.FP16)

            if int8_calib:
                from calibrator import Calibrator
                config.int8_calibrator = Calibrator(calib_loader, calib_cache)
                print('Int8 calibation is enabled.')

        engine = builder.build_engine(network, config)

        try:
            assert engine
        except AssertionError:
            _, _, tb = sys.exc_info()
            traceback.print_tb(tb)  # Fixed format
            tb_info = traceback.extract_tb(tb)
            _, line, _, text = tb_info[-1]
            raise AssertionError(
                "Parsing failed on line {} in statement {}".format(line, text)
            )

        return engine


def main():
    """Create a TensorRT engine for ONNX-based YOLO."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '-v', '--verbose', action='store_true',
        help='enable verbose output (for debugging)')
    parser.add_argument(
        '-m', '--model', type=str, required=True,
        help=('onnx model path'))
    parser.add_argument(
        '-d', '--dtype', type=str, required=True,
        help='one type of int8, fp16, fp32')
    parser.add_argument(
        '--qat', action='store_true',
        help='whether the onnx model is qat; if it is, the int8 calibrator is not needed')
    # If enable int8(not post-QAT model), then set the following
    parser.add_argument('--img-size', nargs='+', type=int,
                        default=[640, 640], help='image size of model input, the order is: height width')
    parser.add_argument('--batch-size', type=int,
                        default=128, help='batch size for training: default 64')
    parser.add_argument('--num-calib-batch', default=6, type=int,
                        help='Number of batches for calibration')
    parser.add_argument('--calib-img-dir', default='../coco/images/train2017', type=str,
                        help='Number of batches for calibration')
    parser.add_argument('--calib-cache', default='./yolov6s_calibration.cache', type=str,
                        help='Path of calibration cache')

    args = parser.parse_args()


    if args.dtype == "int8" and not args.qat:
        from calibrator import DataLoader, Calibrator
        if len(args.img_size) == 1:
            args.img_size = [args.img_size[0], args.img_size[0]]
        calib_loader = DataLoader(args.batch_size, args.num_calib_batch, args.calib_img_dir,
                                  args.img_size[1], args.img_size[0])
        engine = build_engine_from_onnx(args.model, args.dtype, args.verbose,
                              int8_calib=True, calib_loader=calib_loader, calib_cache=args.calib_cache)
    else:
        engine = build_engine_from_onnx(args.model, args.dtype, args.verbose)

    if engine is None:
        raise SystemExit('ERROR: failed to build the TensorRT engine!')

    engine_path = args.model.replace('.onnx', '.trt')
    if args.dtype == "int8" and not args.qat:
        engine_path = args.model.replace('.onnx', '-int8-{}-{}-minmax.trt'.format(args.batch_size, args.num_calib_batch))

    with open(engine_path, 'wb') as f:
        f.write(engine.serialize())
    print('Serialized the TensorRT engine to file: %s' % engine_path)


if __name__ == '__main__':
    main()
