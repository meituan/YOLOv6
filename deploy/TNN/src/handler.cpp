//
// Created by DefTruth on 2022/7/3.
//

#include <fstream>
#include "handler.h"

using yolov6::BasicTNNHandler;

BasicTNNHandler::BasicTNNHandler(
    const std::string &_proto_path, const std::string &_model_path,
    unsigned int _num_threads) : proto_path(_proto_path.data()),
                                 model_path(_model_path.data()),
                                 log_id(_proto_path.data()),
                                 num_threads(_num_threads)
{
  initialize_handler();
}

BasicTNNHandler::~BasicTNNHandler()
{
  net = nullptr;
  instance = nullptr;
  input_mat = nullptr;
#if defined(YOLOV6_DEBUG)
  std::cout << "[MEM DELETE] BasicTNNHandler done!" << "\n";
#endif
}

void BasicTNNHandler::initialize_handler()
{
  std::string proto_content_buffer, model_content_buffer;
  proto_content_buffer = BasicTNNHandler::content_buffer_from(proto_path);
  model_content_buffer = BasicTNNHandler::content_buffer_from(model_path);

  tnn::ModelConfig model_config;
  model_config.model_type = tnn::MODEL_TYPE_TNN;
  model_config.params = {proto_content_buffer, model_content_buffer};

  // 1. init TNN net
  tnn::Status status;
  net = std::make_shared<tnn::TNN>();
  status = net->Init(model_config);
  if (status != tnn::TNN_OK || !net)
  {
#if defined(YOLOV6_DEBUG)
    std::cout << "net->Init failed!\n";
#endif
    return;
  }
  // 2. init device type, change this default setting
  // for better performance. such as CUDA/OPENCL/...
#ifdef __ANDROID__
  network_device_type = tnn::DEVICE_ARM; // CPU,GPU
  input_device_type = tnn::DEVICE_ARM; // CPU only
  output_device_type = tnn::DEVICE_ARM;
#else
  network_device_type = tnn::DEVICE_X86; // CPU,GPU
  input_device_type = tnn::DEVICE_X86; // CPU only
  output_device_type = tnn::DEVICE_X86;
#endif
  // 3. init instance
  tnn::NetworkConfig network_config;
  network_config.library_path = {""};
  network_config.device_type = network_device_type;

  instance = net->CreateInst(network_config, status);
  if (status != tnn::TNN_OK || !instance)
  {
#if defined(YOLOV6_DEBUG)
    std::cout << "CreateInst failed!" << status.description().c_str() << "\n";
#endif
    return;
  }
  // 4. setting up num_threads
  instance->SetCpuNumThreads((int) num_threads);
  // 5. init input information.
  input_name = this->get_input_names().front();
  input_shape = this->get_input_shape(input_name);
  if (input_shape.size() != 4)
  {
#if defined(YOLOV6_DEBUG)
    throw std::runtime_error("Found input_shape.size()!=4, but "
                             "BasicTNNHandler only support 4 dims."
                             "Such as NCHW, NHWC ...");
#else
    return;
#endif
  }
  input_mat_type = this->get_input_mat_type(input_name);
  input_data_format = this->get_input_data_format(input_name);
  // This BasicTNNHandler only support NC_INT32 & NCHW_FLOAT
  if (input_data_format == tnn::DATA_FORMAT_NCHW)
  {
    input_batch = input_shape.at(0);
    input_channel = input_shape.at(1);
    input_height = input_shape.at(2);
    input_width = input_shape.at(3);
  } // NHWC
  else if (input_data_format == tnn::DATA_FORMAT_NHWC)
  {
    input_batch = input_shape.at(0);
    input_height = input_shape.at(1);
    input_width = input_shape.at(2);
    input_channel = input_shape.at(3);
  } // unsupport
  else
  {
#if defined(YOLOV6_DEBUG)
    std::cout << "BasicTNNHandler only support NCHW and NHWC "
                 "input_data_format, but found others.\n";
#endif
    return;
  }
  // 6. init input_mat
  input_value_size = input_batch * input_channel * input_height * input_width;
  // 7. init output information, debug only.
  output_names = this->get_output_names();
  num_outputs = output_names.size();
  for (auto &name: output_names)
    output_shapes[name] = this->get_output_shape(name);
#if defined(YOLOV6_DEBUG)
  this->print_debug_string();
#endif
}

inline tnn::DimsVector BasicTNNHandler::get_input_shape(std::string name)
{
  return BasicTNNHandler::get_input_shape(instance, name);
}

inline tnn::DimsVector BasicTNNHandler::get_output_shape(std::string name)
{
  return BasicTNNHandler::get_output_shape(instance, name);
}

inline std::vector<std::string> BasicTNNHandler::get_input_names()
{
  return BasicTNNHandler::get_input_names(instance);
}

inline std::vector<std::string> BasicTNNHandler::get_output_names()
{
  return BasicTNNHandler::get_output_names(instance);
}

inline tnn::MatType BasicTNNHandler::get_output_mat_type(std::string name)
{
  return BasicTNNHandler::get_output_mat_type(instance, name);
}

inline tnn::DataFormat BasicTNNHandler::get_output_data_format(std::string name)
{
  return BasicTNNHandler::get_output_data_format(instance, name);
}

inline tnn::MatType BasicTNNHandler::get_input_mat_type(std::string name)
{
  return BasicTNNHandler::get_input_mat_type(instance, name);
}

inline tnn::DataFormat BasicTNNHandler::get_input_data_format(std::string name)
{
  return BasicTNNHandler::get_input_data_format(instance, name);
}

void BasicTNNHandler::print_debug_string()
{
  std::cout << "YOLOV6_DEBUG LogId: " << log_id << "\n";
  std::cout << "=============== Input-Dims ==============\n";
  BasicTNNHandler::print_name_shape(input_name, input_shape);
  std::string data_format_string =
      (input_data_format == tnn::DATA_FORMAT_NCHW) ? "NCHW" : "NHWC";
  std::cout << "Input Data Format: " << data_format_string << "\n";
  std::cout << "=============== Output-Dims ==============\n";
  for (auto &out: output_shapes)
    BasicTNNHandler::print_name_shape(out.first, out.second);
  std::cout << "========================================\n";
}

// static methods.
void BasicTNNHandler::print_name_shape(std::string name, tnn::DimsVector &shape)
{
  std::cout << name << ": [";
  for (const auto &d: shape) std::cout << d << " ";
  std::cout << "]\n";
}

// static methods.
// reference: https://github.com/Tencent/TNN/blob/master/examples/base/utils/utils.cc
std::string BasicTNNHandler::content_buffer_from(const char *proto_or_model_path)
{
  std::ifstream file(proto_or_model_path, std::ios::binary);
  if (file.is_open())
  {
    file.seekg(0, file.end);
    int size = file.tellg();
    char *content = new char[size];
    file.seekg(0, file.beg);
    file.read(content, size);
    std::string file_content;
    file_content.assign(content, size);
    delete[] content;
    file.close();
    return file_content;
  } // empty buffer
  else
  {
#if defined(YOLOV6_DEBUG)
    std::cout << "Can not open " << proto_or_model_path << "\n";
#endif
    return "";
  }
}

// static methods.
tnn::DimsVector BasicTNNHandler::get_input_shape(
    const std::shared_ptr<tnn::Instance> &_instance,
    std::string name)
{
  tnn::DimsVector shape = {};
  tnn::BlobMap blob_map = {};
  if (_instance)
  {
    _instance->GetAllInputBlobs(blob_map);
  }

  if (name == "" && blob_map.size() > 0)
    if (blob_map.begin()->second)
      shape = blob_map.begin()->second->GetBlobDesc().dims;

  if (blob_map.find(name) != blob_map.end()
      && blob_map[name])
  {
    shape = blob_map[name]->GetBlobDesc().dims;
  }

  return shape;
}

// static methods.
tnn::DimsVector BasicTNNHandler::get_output_shape(
    const std::shared_ptr<tnn::Instance> &_instance,
    std::string name)
{
  tnn::DimsVector shape = {};
  tnn::BlobMap blob_map = {};
  if (_instance)
  {
    _instance->GetAllOutputBlobs(blob_map);
  }

  if (name == "" && blob_map.size() > 0)
    if (blob_map.begin()->second)
      shape = blob_map.begin()->second->GetBlobDesc().dims;

  if (blob_map.find(name) != blob_map.end()
      && blob_map[name])
  {
    shape = blob_map[name]->GetBlobDesc().dims;
  }

  return shape;
}

// static methods.
std::vector<std::string> BasicTNNHandler::get_input_names(
    const std::shared_ptr<tnn::Instance> &_instance)
{
  std::vector<std::string> names;
  if (_instance)
  {
    tnn::BlobMap blob_map;
    _instance->GetAllInputBlobs(blob_map);
    for (const auto &item: blob_map)
    {
      names.push_back(item.first);
    }
  }
  return names;
}

// static method
std::vector<std::string> BasicTNNHandler::get_output_names(
    const std::shared_ptr<tnn::Instance> &_instance)
{
  std::vector<std::string> names;
  if (_instance)
  {
    tnn::BlobMap blob_map;
    _instance->GetAllOutputBlobs(blob_map);
    for (const auto &item: blob_map)
    {
      names.push_back(item.first);
    }
  }
  return names;
}

// static method
tnn::MatType BasicTNNHandler::get_output_mat_type(
    const std::shared_ptr<tnn::Instance> &_instance,
    std::string name)
{
  if (_instance)
  {
    tnn::BlobMap output_blobs;
    _instance->GetAllOutputBlobs(output_blobs);
    auto blob = (name == "") ? output_blobs.begin()->second : output_blobs[name];
    if (blob->GetBlobDesc().data_type == tnn::DATA_TYPE_INT32)
    {
      return tnn::NC_INT32;
    }
  }
  return tnn::NCHW_FLOAT;
}

// static method
tnn::DataFormat BasicTNNHandler::get_output_data_format(
    const std::shared_ptr<tnn::Instance> &_instance,
    std::string name)
{
  if (_instance)
  {
    tnn::BlobMap output_blobs;
    _instance->GetAllOutputBlobs(output_blobs);
    auto blob = (name == "") ? output_blobs.begin()->second : output_blobs[name];
    return blob->GetBlobDesc().data_format;
  }
  return tnn::DATA_FORMAT_NCHW;
}

// static method
tnn::MatType BasicTNNHandler::get_input_mat_type(
    const std::shared_ptr<tnn::Instance> &_instance,
    std::string name)
{
  if (_instance)
  {
    tnn::BlobMap input_blobs;
    _instance->GetAllInputBlobs(input_blobs);
    auto blob = (name == "") ? input_blobs.begin()->second : input_blobs[name];
    if (blob->GetBlobDesc().data_type == tnn::DATA_TYPE_INT32)
    {
      return tnn::NC_INT32;
    }
  }
  return tnn::NCHW_FLOAT;
}

// static method
tnn::DataFormat BasicTNNHandler::get_input_data_format(
    const std::shared_ptr<tnn::Instance> &_instance,
    std::string name)
{
  if (_instance)
  {
    tnn::BlobMap input_blobs;
    _instance->GetAllInputBlobs(input_blobs);
    auto blob = (name == "") ? input_blobs.begin()->second : input_blobs[name];
    return blob->GetBlobDesc().data_format;
  }
  return tnn::DATA_FORMAT_NCHW;
}

