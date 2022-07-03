//
// Created by DefTruth on 2022/7/3.
//

#include "handler.h"

using yolov6::BasicMNNHandler;

BasicMNNHandler::BasicMNNHandler(
    const std::string &_mnn_path, unsigned int _num_threads) :
    log_id(_mnn_path.data()), mnn_path(_mnn_path.data()),
    num_threads(_num_threads)
{
  initialize_handler();
}

void BasicMNNHandler::initialize_handler()
{
  // 1. init interpreter
  mnn_interpreter = std::shared_ptr<MNN::Interpreter>(MNN::Interpreter::createFromFile(mnn_path));
  // 2. init schedule_config
  schedule_config.numThread = (int) num_threads;
  MNN::BackendConfig backend_config;
  backend_config.precision = MNN::BackendConfig::Precision_High; // default Precision_High
  schedule_config.backendConfig = &backend_config;
  // 3. create session
  mnn_session = mnn_interpreter->createSession(schedule_config);
  // 4. init input tensor
  input_tensor = mnn_interpreter->getSessionInput(mnn_session, nullptr);
  // 5. init input dims
  input_batch = input_tensor->batch();
  input_channel = input_tensor->channel();
  input_height = input_tensor->height();
  input_width = input_tensor->width();
  dimension_type = input_tensor->getDimensionType();
  // 6. resize tensor & session needed ???
  if (dimension_type == MNN::Tensor::CAFFE)
  {
    // NCHW
    mnn_interpreter->resizeTensor(
        input_tensor, {input_batch, input_channel, input_height, input_width});
    mnn_interpreter->resizeSession(mnn_session);
  } // NHWC
  else if (dimension_type == MNN::Tensor::TENSORFLOW)
  {
    mnn_interpreter->resizeTensor(
        input_tensor, {input_batch, input_height, input_width, input_channel});
    mnn_interpreter->resizeSession(mnn_session);
  } // NC4HW4
  else if (dimension_type == MNN::Tensor::CAFFE_C4)
  {
#if defined(YOLOV6_DEBUG)
    std::cout << "Dimension Type is CAFFE_C4, skip resizeTensor & resizeSession!\n";
#endif
  }
  // output count
  num_outputs = mnn_interpreter->getSessionOutputAll(mnn_session).size();
#if defined(YOLOV6_DEBUG)
  this->print_debug_string();
#endif
}

BasicMNNHandler::~BasicMNNHandler()
{
  mnn_interpreter->releaseModel();
  if (mnn_session)
    mnn_interpreter->releaseSession(mnn_session);
#if defined(YOLOV6_DEBUG)
  std::cout << "[MEM DELETE] BasicMNNHandler done!" << "\n";
#endif
}

void BasicMNNHandler::print_debug_string()
{
  std::cout << "YOLOV6_DEBUG LogId: " << log_id << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  if (input_tensor) input_tensor->printShape();
  if (dimension_type == MNN::Tensor::CAFFE)
    std::cout << "Dimension Type: (CAFFE/PyTorch/ONNX)NCHW" << "\n";
  else if (dimension_type == MNN::Tensor::TENSORFLOW)
    std::cout << "Dimension Type: (TENSORFLOW)NHWC" << "\n";
  else if (dimension_type == MNN::Tensor::CAFFE_C4)
    std::cout << "Dimension Type: (CAFFE_C4)NC4HW4" << "\n";
  std::cout << "=============== Output-Dims ==============\n";
  auto tmp_output_map = mnn_interpreter->getSessionOutputAll(mnn_session);
  std::cout << "getSessionOutputAll done!\n";
  for (auto it = tmp_output_map.cbegin(); it != tmp_output_map.cend(); ++it)
  {
    std::cout << "Output: " << it->first << ": ";
    it->second->printShape();
  }
  std::cout << "========================================\n";
}