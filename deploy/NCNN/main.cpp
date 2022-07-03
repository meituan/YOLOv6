//
// Created by DefTruth on 2022/7/3.
//

#include "yolov6.h"

static void test_yolov6()
{
  std::string param_path = "../model/yolov6s-640x640-for-ncnn.opt.param";
  std::string bin_path = "../model/yolov6s-640x640-for-ncnn.opt.bin";
  std::string test_img_path = "../test.jpg";
  std::string save_img_path = "../output.jpg";

  auto yolov6_det = yolov6::create(param_path, bin_path, 1, 640, 640); // 1 threads

  std::vector<yolov6::types::Boxf> detected_boxes;
  cv::Mat img_bgr = cv::imread(test_img_path);
  yolov6_det->detect(img_bgr, detected_boxes);

  yolov6::utils::draw_boxes_inplace(img_bgr, detected_boxes);

  cv::imwrite(save_img_path, img_bgr);

  std::cout << "Detected Boxes Num: " << detected_boxes.size() << std::endl;
}

int main(__unused int argc, __unused char *argv[])
{
  test_yolov6();
  return 0;
}
