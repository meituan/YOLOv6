//
// Created by DefTruth on 2022/7/3.
//

#ifndef YOLOV6_UTILS_H
#define YOLOV6_UTILS_H

#include "types.h"
#include <string>

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
  }
}

#endif //YOLOV6_UTILS_H
