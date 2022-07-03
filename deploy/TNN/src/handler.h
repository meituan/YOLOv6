//
// Created by DefTruth on 2022/7/3.
//

#ifndef YOLOV6_HANDLER_H
#define YOLOV6_HANDLER_H

#include "types.h"

#include "tnn/core/macro.h"
#include "tnn/core/tnn.h"
#include "tnn/core/mat.h"
#include "tnn/utils/blob_converter.h"
#include "tnn/utils/mat_utils.h"
#include "tnn/utils/dims_vector_utils.h"

namespace yolov6
{
  class YOLOV6_EXPORTS BasicTNNHandler
  {
  protected:
    const char *log_id = nullptr;
    const char *proto_path = nullptr;
    const char *model_path = nullptr;
    // Note, tnn:: actually is TNN_NS::, I prefer the first one.
    std::shared_ptr<tnn::TNN> net;
    std::shared_ptr<tnn::Instance> instance;
    std::shared_ptr<tnn::Mat> input_mat; // assume single input.

  protected:
    const unsigned int num_threads; // initialize at runtime.
    int input_batch;
    int input_channel;
    int input_height;
    int input_width;
    int num_outputs = 1;
    unsigned int input_value_size;
    tnn::DataFormat input_data_format;  // e.g DATA_FORMAT_NHWC
    tnn::MatType input_mat_type; // e.g NCHW_FLOAT
    tnn::DeviceType input_device_type; // only CPU, namely ARM or X86
    tnn::DeviceType output_device_type; // only CPU, namely ARM or X86
    tnn::DeviceType network_device_type; // e.g DEVICE_X86 DEVICE_NAIVE DEVICE_ARM
    // Actually, i prefer to hardcode the input/output names
    // into subclasses, but we just let the auto detection here
    // to make sure the debug information can show more details.
    std::string input_name; // assume single input only.
    std::vector<std::string> output_names; // assume >= 1 outputs.
    tnn::DimsVector input_shape; // vector<int>
    std::map<std::string, tnn::DimsVector> output_shapes;

  protected:
    explicit BasicTNNHandler(const std::string &_proto_path,
                             const std::string &_model_path,
                             unsigned int _num_threads = 1);

    virtual ~BasicTNNHandler();

    // un-copyable
  protected:
    BasicTNNHandler(const BasicTNNHandler &) = delete; //
    BasicTNNHandler(BasicTNNHandler &&) = delete; //
    BasicTNNHandler &operator=(const BasicTNNHandler &) = delete; //
    BasicTNNHandler &operator=(BasicTNNHandler &&) = delete; //

  private:
    virtual void transform(const cv::Mat &mat) = 0;

  private:
    void initialize_handler(); // init net & instance
    void print_debug_string(); // debug information

  protected:
    // helper functions.
    tnn::DimsVector get_input_shape(std::string name);

    tnn::DimsVector get_output_shape(std::string name);

    tnn::MatType get_output_mat_type(std::string name);

    tnn::DataFormat get_output_data_format(std::string name);

    tnn::MatType get_input_mat_type(std::string name);

    tnn::DataFormat get_input_data_format(std::string name);

    std::vector<std::string> get_input_names();

    std::vector<std::string> get_output_names();

  public:
    // helper functions. override for user firendly
    static tnn::DimsVector get_input_shape(
        const std::shared_ptr<tnn::Instance> &_instance, std::string name);

    static tnn::DimsVector get_output_shape(
        const std::shared_ptr<tnn::Instance> &_instance, std::string name);

    static tnn::MatType get_output_mat_type(
        const std::shared_ptr<tnn::Instance> &_instance, std::string name);

    static tnn::DataFormat get_output_data_format(
        const std::shared_ptr<tnn::Instance> &_instance, std::string name);

    static tnn::MatType get_input_mat_type(
        const std::shared_ptr<tnn::Instance> &_instance, std::string name);

    static tnn::DataFormat get_input_data_format(
        const std::shared_ptr<tnn::Instance> &_instance, std::string name);

    static std::vector<std::string> get_input_names(
        const std::shared_ptr<tnn::Instance> &_instance);

    static std::vector<std::string> get_output_names(
        const std::shared_ptr<tnn::Instance> &_instance);

  public:
    static std::string content_buffer_from(
        const char *proto_or_model_path);

    static void print_name_shape(std::string name, tnn::DimsVector &shape);

  };
}

#endif //YOLOV6_HANDLER_H
