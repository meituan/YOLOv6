//
// Created by DefTruth on 2022/7/3.
//

#ifndef YOLOV6_HANDLER_H
#define YOLOV6_HANDLER_H

#include "types.h"

#include "onnxruntime/core/session/onnxruntime_cxx_api.h"
/* Need to define USE_CUDA macro manually by users who want to
 * enable onnxruntime and lite.ai.toolkit with CUDA support. It
 * seems that the latest onnxruntime will no longer pre-defined the
 * USE_CUDA macro and just let the decision make by users
 * who really know the environments of running device.*/
// #define USE_CUDA
#if defined(USE_CUDA)
#include "onnxruntime/core/providers/cuda/cuda_provider_factory.h"
#endif

namespace yolov6
{
  // single input & multi outputs. not support for dynamic shape currently.
  class YOLOV6_EXPORTS BasicOrtHandler
  {
  protected:
    Ort::Env ort_env;
    Ort::Session *ort_session = nullptr;
    const char *input_name = nullptr;
    std::vector<const char *> input_node_names;
    std::vector<int64_t> input_node_dims; // 1 input only.
    std::size_t input_tensor_size = 1;
    std::vector<float> input_values_handler;
    Ort::MemoryInfo memory_info_handler = Ort::MemoryInfo::CreateCpu(
        OrtArenaAllocator, OrtMemTypeDefault);
    std::vector<const char *> output_node_names;
    std::vector<std::vector<int64_t>> output_node_dims; // >=1 outputs
    const YOLOV6_ORT_CHAR *onnx_path = nullptr;
    const char *log_id = nullptr;
    int num_outputs = 1;

  protected:
    const unsigned int num_threads; // initialize at runtime.

  protected:
    explicit BasicOrtHandler(const std::string &_onnx_path, unsigned int _num_threads = 1);

    virtual ~BasicOrtHandler();

    // un-copyable
  protected:
    BasicOrtHandler(const BasicOrtHandler &) = delete;
    BasicOrtHandler(BasicOrtHandler &&) = delete;
    BasicOrtHandler &operator=(const BasicOrtHandler &) = delete;
    BasicOrtHandler &operator=(BasicOrtHandler &&) = delete;

  protected:
    virtual Ort::Value transform(const cv::Mat &mat) = 0;

  private:
    void initialize_handler();

    void print_debug_string();
  };
}

#endif //YOLOV6_HANDLER_H
