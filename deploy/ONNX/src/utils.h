//
// Created by DefTruth on 2022/7/3.
//

#ifndef YOLOV6_UTILS_H
#define YOLOV6_UTILS_H

#include "types.h"
#include <string>
#include "onnxruntime/core/session/onnxruntime_cxx_api.h"

namespace yolov6
{
  namespace utils
  {
    // String Utils
    YOLOV6_EXPORTS std::wstring to_wstring(const std::string &str);
    YOLOV6_EXPORTS std::string to_string(const std::wstring &wstr);
    // Drawing Utils
    YOLOV6_EXPORTS cv::Mat draw_boxes(const cv::Mat &mat, const std::vector<types::Boxf> &boxes);
    YOLOV6_EXPORTS void draw_boxes_inplace(cv::Mat &mat_inplace, const std::vector<types::Boxf> &boxes);
    // Object Detection Utils
    YOLOV6_EXPORTS void hard_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold, unsigned int topk);
    YOLOV6_EXPORTS void blending_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold, unsigned int topk);
    YOLOV6_EXPORTS void offset_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output, float iou_threshold, unsigned int topk);
    // ONNXRuntime Utils
    namespace transform
    {
      enum
      {
        CHW = 0, HWC = 1
      };

      /**
       * @param mat CV:Mat with type 'CV_32FC3|2|1'
       * @param tensor_dims e.g {1,C,H,W} | {1,H,W,C}
       * @param memory_info It needs to be a global variable in a class
       * @param tensor_value_handler It needs to be a global variable in a class
       * @param data_format CHW | HWC
       * @return
       */
      YOLOV6_EXPORTS Ort::Value create_tensor(const cv::Mat &mat, const std::vector<int64_t> &tensor_dims,
                                              const Ort::MemoryInfo &memory_info_handler,
                                              std::vector<float> &tensor_value_handler,
                                              unsigned int data_format = CHW) throw(std::runtime_error);
      YOLOV6_EXPORTS cv::Mat normalize(const cv::Mat &mat, float mean, float scale);
      YOLOV6_EXPORTS cv::Mat normalize(const cv::Mat &mat, const float mean[3], const float scale[3]);
      YOLOV6_EXPORTS void normalize(const cv::Mat &inmat, cv::Mat &outmat, float mean, float scale);
      YOLOV6_EXPORTS void normalize_inplace(cv::Mat &mat_inplace, float mean, float scale);
      YOLOV6_EXPORTS void normalize_inplace(cv::Mat &mat_inplace, const float mean[3], const float scale[3]);
    }
  }
}

#endif //YOLOV6_UTILS_H
