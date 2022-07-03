//
// Created by DefTruth on 2022/7/3.
//

#ifndef YOLOV6_HANDLER_H
#define YOLOV6_HANDLER_H

#include "types.h"

#include "MNN/Interpreter.hpp"
#include "MNN/MNNDefine.h"
#include "MNN/Tensor.hpp"
#include "MNN/ImageProcess.hpp"

namespace yolov6
{
  class YOLOV6_EXPORTS BasicMNNHandler
  {
  protected:
    std::shared_ptr<MNN::Interpreter> mnn_interpreter;
    MNN::Session *mnn_session = nullptr;
    MNN::Tensor *input_tensor = nullptr; // assume single input.
    MNN::ScheduleConfig schedule_config;
    std::shared_ptr<MNN::CV::ImageProcess> pretreat; // init at subclass
    const char *log_id = nullptr;
    const char *mnn_path = nullptr;

  protected:
    const unsigned int num_threads; // initialize at runtime.
    int input_batch;
    int input_channel;
    int input_height;
    int input_width;
    int dimension_type;
    int num_outputs = 1;

  protected:
    explicit BasicMNNHandler(const std::string &_mnn_path, unsigned int _num_threads = 1);

    virtual ~BasicMNNHandler();

    // un-copyable
  protected:
    BasicMNNHandler(const BasicMNNHandler &) = delete; //
    BasicMNNHandler(BasicMNNHandler &&) = delete; //
    BasicMNNHandler &operator=(const BasicMNNHandler &) = delete; //
    BasicMNNHandler &operator=(BasicMNNHandler &&) = delete; //

  protected:
    virtual void transform(const cv::Mat &mat) = 0; // ? needed ?

  private:
    void initialize_handler();

    void print_debug_string();
  };
}

#endif //YOLOV6_HANDLER_H
