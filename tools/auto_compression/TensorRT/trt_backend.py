import tensorrt as trt
import pycuda.driver as cuda
import pycuda.autoinit
import sys
import os
import copy
import numpy as np

EXPLICIT_BATCH = 1 << (int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH)
EXPLICIT_PRECISION = 1 << (
    int)(trt.NetworkDefinitionCreationFlag.EXPLICIT_PRECISION)


class LoadCalibrator(trt.IInt8EntropyCalibrator2):
    def __init__(self, cache_file="calibration.cache"):
        super().__init__()
        self.cache_file = cache_file

    def get_batch_size(self):
        return 1

    def read_calibration_cache(self):
        # If there is a cache, use it instead of calibrating again. Otherwise, implicitly return None.
        if os.path.exists(self.cache_file):
            with open(self.cache_file, "rb") as f:
                print("Using calibration cache to save time: {:}".format(
                    self.cache_file))
                return f.read()


def remove_initializer_from_input(ori_model):
    model = copy.deepcopy(ori_model)
    if model.ir_version < 4:
        print(
            'Model with ir_version below 4 requires to include initilizer in graph input'
        )
        return

    inputs = model.graph.input
    name_to_input = {}
    for input in inputs:
        name_to_input[input.name] = input

    for initializer in model.graph.initializer:
        if initializer.name in name_to_input:
            inputs.remove(name_to_input[initializer.name])
    return model


# Simple helper data class that's a little nicer to use than a 2-tuple.
class HostDeviceMem(object):
    def __init__(self, host_mem, device_mem):
        self.host = host_mem
        self.device = device_mem
        if host_mem:
            self.nbytes = host_mem.nbytes
        else:
            self.nbytes = 0

    def __str__(self):
        return "Host:\n" + str(self.host) + "\nDevice:\n" + str(self.device)

    def __repr__(self):
        return self.__str__()


class TrtEngine:
    def __init__(self,
                 onnx_model_file,
                 shape_info=None,
                 max_batch_size=None,
                 precision_mode="fp32",
                 engine_file_path=None,
                 calibration_cache_file="calibration.cache",
                 verbose=False):
        self.max_batch_size = 1 if max_batch_size is None else max_batch_size
        precision_mode = precision_mode.lower()
        assert precision_mode in [
            "fp32", "fp16", "int8"
        ], "precision_mode must be fp32, fp16 or int8, but your precision_mode is: {}".format(
            precision_mode)
        use_int8 = precision_mode == "int8"
        use_fp16 = precision_mode == "fp16"
        TRT_LOGGER = trt.Logger()
        if verbose:
            TRT_LOGGER = trt.Logger(trt.Logger.VERBOSE)
        if engine_file_path is not None and os.path.exists(engine_file_path):
            # If a serialized engine exists, use it instead of building an engine.
            print("[TRT Backend] Reading engine from file {}".format(
                engine_file_path))
            with open(engine_file_path,
                      "rb") as f, trt.Runtime(TRT_LOGGER) as runtime:
                self.engine = runtime.deserialize_cuda_engine(f.read())
        else:
            builder = trt.Builder(TRT_LOGGER)
            config = builder.create_builder_config()
            network = None

            if use_int8 and not builder.platform_has_fast_int8:
                print("[TRT Backend] INT8 not supported on this platform.")
            if use_fp16 and not builder.platform_has_fast_fp16:
                print("[TRT Backend] FP16 not supported on this platform.")

            if use_int8 and builder.platform_has_fast_int8:
                print("[TRT Backend] Use INT8.")
                network = builder.create_network(EXPLICIT_BATCH |
                                                 EXPLICIT_PRECISION)
                config.int8_calibrator = LoadCalibrator(calibration_cache_file)
                config.set_flag(trt.BuilderFlag.INT8)
            elif use_fp16 and builder.platform_has_fast_fp16:
                print("[TRT Backend] Use FP16.")
                network = builder.create_network(EXPLICIT_BATCH)
                config.set_flag(trt.BuilderFlag.FP16)
            else:
                print("[TRT Backend] Use FP32.")
                network = builder.create_network(EXPLICIT_BATCH)
            parser = trt.OnnxParser(network, TRT_LOGGER)
            runtime = trt.Runtime(TRT_LOGGER)
            config.max_workspace_size = 1 << 28

            import onnx
            print("[TRT Backend] Loading ONNX model ...")
            onnx_model = onnx_model_file
            if not isinstance(onnx_model_file, onnx.ModelProto):
                onnx_model = onnx.load(onnx_model_file)
            onnx_model = remove_initializer_from_input(onnx_model)
            if not parser.parse(onnx_model.SerializeToString()):
                for error in range(parser.num_errors):
                    print(parser.get_error(error))
                raise Exception("ERROR: Failed to parse the ONNX file.")

            if shape_info is None:
                builder.max_batch_size = 1
                for i in range(len(onnx_model.graph.input)):
                    input_shape = [
                        x.dim_value
                        for x in onnx_model.graph.input[0]
                        .type.tensor_type.shape.dim
                    ]
                    for s in input_shape:
                        assert s > 0, "In static shape mode, the input of onnx model should be fixed, but now it's {}".format(
                            onnx_model.graph.input[i])
            else:
                max_batch_size = 1
                if shape_info is not None:
                    assert len(
                        shape_info
                    ) == network.num_inputs, "Length of shape_info: {} is not same with length of model input: {}".format(
                        len(shape_info), network.num_inputs)
                    profile = builder.create_optimization_profile()
                    for k, v in shape_info.items():
                        if v[2][0] > max_batch_size:
                            max_batch_size = v[2][0]
                        print("[TRT Backend] optimize shape: ", k, v[0], v[1],
                              v[2])
                        profile.set_shape(k, v[0], v[1], v[2])
                    config.add_optimization_profile(profile)
                if max_batch_size > self.max_batch_size:
                    self.max_batch_size = max_batch_size
                builder.max_batch_size = self.max_batch_size

            print("[TRT Backend] Completed parsing of ONNX file.")
            print(
                "[TRT Backend] Building an engine from onnx model may take a while..."
            )
            plan = builder.build_serialized_network(network, config)
            print("[TRT Backend] Start Creating Engine.")
            self.engine = runtime.deserialize_cuda_engine(plan)
            print("[TRT Backend] Completed Creating Engine.")
            if engine_file_path is not None:
                with open(engine_file_path, "wb") as f:
                    f.write(self.engine.serialize())

        self.context = self.engine.create_execution_context()
        if shape_info is not None:
            self.context.active_optimization_profile = 0
        self.stream = cuda.Stream()
        self.bindings = []
        self.inputs = []
        self.outputs = []
        for binding in self.engine:
            self.bindings.append(0)
            if self.engine.binding_is_input(binding):
                self.inputs.append(HostDeviceMem(None, None))
            else:
                self.outputs.append(HostDeviceMem(None, None))

        print("[TRT Backend] Completed TrtEngine init ...")

    def infer(self, input_data):
        assert len(self.inputs) == len(
            input_data
        ), "Length of input_data: {} is not same with length of input: {}".format(
            len(input_data), len(self.inputs))

        self.allocate_buffers(input_data)

        return self.do_inference_v2(
            self.context,
            bindings=self.bindings,
            inputs=self.inputs,
            outputs=self.outputs,
            stream=self.stream)

    def do_inference_v2(self, context, bindings, inputs, outputs, stream):
        # Transfer input data to the GPU.
        [cuda.memcpy_htod_async(inp.device, inp.host, stream) for inp in inputs]
        # Run inference.
        context.execute_async_v2(bindings=bindings, stream_handle=stream.handle)
        # Transfer predictions back from the GPU.
        [
            cuda.memcpy_dtoh_async(out.host, out.device, stream)
            for out in outputs
        ]
        # Synchronize the stream
        stream.synchronize()
        # Return only the host outputs.
        return [out.host for out in outputs]

    def allocate_buffers(self, input_data):
        input_idx = 0
        output_idx = 0
        for binding in self.engine:
            idx = self.engine.get_binding_index(binding)
            if self.engine.binding_is_input(binding):
                if not input_data[input_idx].flags['C_CONTIGUOUS']:
                    input_data[input_idx] = np.ascontiguousarray(input_data[
                        input_idx])
                self.context.set_binding_shape(idx,
                                               (input_data[input_idx].shape))
                self.inputs[input_idx].host = input_data[input_idx]
                nbytes = input_data[input_idx].nbytes
                if self.inputs[input_idx].nbytes < nbytes:
                    self.inputs[input_idx].nbytes = nbytes
                    self.inputs[input_idx].device = cuda.mem_alloc(nbytes)
                    self.bindings[idx] = int(self.inputs[input_idx].device)
                input_idx += 1
            else:
                dtype = trt.nptype(self.engine.get_binding_dtype(binding))
                shape = self.context.get_binding_shape(idx)
                self.outputs[output_idx].host = np.ascontiguousarray(
                    np.empty(
                        shape, dtype=dtype))
                nbytes = self.outputs[output_idx].host.nbytes
                if self.outputs[output_idx].nbytes < nbytes:
                    self.outputs[output_idx].nbytes = nbytes
                    self.outputs[output_idx].device = cuda.mem_alloc(
                        self.outputs[output_idx].host.nbytes)
                    self.bindings[idx] = int(self.outputs[output_idx].device)
                output_idx += 1
