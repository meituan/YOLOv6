# Utility functions for performing image inference
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

import os
import sys
import time
# from PIL import Image
import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import numpy as np
import utils.engine as engine_utils  # TRT Engine creation/save/load utils
import cv2

# TensorRT logger singleton
TRT_LOGGER = trt.Logger(trt.Logger.WARNING)


class TRTInference(object):
    """Manages TensorRT objects for model inference."""

    def __init__(self, trt_engine_path, onnx_model_path, trt_engine_datatype=trt.DataType.FLOAT, batch_size=1, dynamic_shapes={}):
        """Initializes TensorRT objects needed for model inference.

        Args:
            trt_engine_path (str): path where TensorRT engine should be stored
            uff_model_path (str): path of .uff model
            trt_engine_datatype (trt.DataType):
                requested precision of TensorRT engine used for inference
            batch_size (int): batch size for which engine
                should be optimized for
        """

        # Initialize runtime needed for loading TensorRT engine from file
        self.trt_runtime = trt.Runtime(TRT_LOGGER)
        # TRT engine placeholder
        self.trt_engine = None

        # Display requested engine settings to stdout
        print("TensorRT inference engine settings:")
        print("  * Inference precision - {}".format(trt_engine_datatype))
        print("  * Max batch size - {}\n".format(batch_size))

        # If engine is not cached, we need to build it
        if not os.path.exists(trt_engine_path):
            # This function uses supplied .uff file
            # alongside with UffParser to build TensorRT
            # engine. For more details, check implmentation
            self.trt_engine = engine_utils.build_engine(
                onnx_model_path, TRT_LOGGER,
                trt_engine_datatype=trt_engine_datatype,
                batch_size=batch_size, dynamic_shapes=dynamic_shapes)
            # Save the engine to file
            print("Completed creating Engine")
            engine_utils.save_engine(self.trt_engine, trt_engine_path)

        # If we get here, the file with engine exists, so we can load it
        if not self.trt_engine:
            print("Loading cached TensorRT engine from {}".format(
                trt_engine_path))
            self.trt_engine = engine_utils.load_engine(
                self.trt_runtime, trt_engine_path)

        # Execution context is needed for inference
        self.context = self.trt_engine.create_execution_context()

    def infer(self, input_data, output_shapes):
        """Infers model on given image.

        Args:
            image_path (str): image to run object detection model on
        """

        # This allocates memory for network inputs/outputs on both CPU and GPU
        self.inputs, self.outputs, self.bindings, self.stream = \
            engine_utils.allocate_buffers(self.trt_engine, self.context)

        input_data = np.array(input_data, dtype=np.float32, order='C')

        np.copyto(self.inputs[0].host, input_data.ravel())

        # Fetch output from the model
        trt_outputs = do_inference_v2(
            self.context, bindings=self.bindings, inputs=self.inputs,
            outputs=self.outputs, stream=self.stream)

        # Before doing post-processing, we need to reshape the outputs as the common.do_inference will
        # give us flat arrays.
        outputs = [output.reshape(shape) for output, shape in zip(trt_outputs, output_shapes)]
        # And return results
        return outputs


# This function is generalized for multiple inputs/outputs for full dimension networks.
# inputs and outputs are expected to be lists of HostDeviceMem objects.
def do_inference_v2(context, bindings, inputs, outputs, stream):

    # Transfer input data to the GPU.
    [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
    # Run inference.
    context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
    # Transfer predictions back from the GPU.
    [cuda.memcpy_dtoh_async(out.host, out.device, stream) for out in outputs]
    # Synchronize the stream
    stream.synchronize()
    # Return only the host outputs.
    return [out.host for out in outputs]
