//
// Created by DefTruth on 2022/7/3.
//

#ifndef YOLOV6_YOLOV6_IMPL_H
#define YOLOV6_YOLOV6_IMPL_H

#include "types.h"
#include "handler.h"

namespace yolov6
{
  class YOLOV6_EXPORTS YOLOv6Impl
  {
  private:
    ncnn::Net *net = nullptr;
    const char *log_id = nullptr;
    const char *param_path = nullptr;
    const char *bin_path = nullptr;
    std::vector<const char *> input_names;
    std::vector<const char *> output_names;
    std::vector<int> input_indexes;
    std::vector<int> output_indexes;

  public:
    explicit YOLOv6Impl(const std::string &_param_path,
                        const std::string &_bin_path,
                        unsigned int _num_threads = 1,
                        int _input_height = 640,
                        int _input_width = 640); //
    ~YOLOv6Impl();

  private:
    // nested classes
    typedef struct GridAndStride
    {
      int grid0;
      int grid1;
      int stride;
    } YOLOv6Anchor;

    typedef struct
    {
      float r;
      int dw;
      int dh;
      int new_unpad_w;
      int new_unpad_h;
      bool flag;
    } YOLOv6ScaleParams;

  private:
    const unsigned int num_threads; // initialize at runtime.
    const int input_height; // 640/320
    const int input_width; // 640/320

    const char *class_names[80] = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard",
        "cell phone", "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase",
        "scissors", "teddy bear", "hair drier", "toothbrush"
    };
    enum NMS
    {
      HARD = 0, BLEND = 1, OFFSET = 2
    };
    const float mean_vals[3] = {0.f, 0.f, 0.f}; // RGB
    const float norm_vals[3] = {1.0 / 255.f, 1.0 / 255.f, 1.0 / 255.f};
    static constexpr const unsigned int max_nms = 30000;

  protected:
    YOLOv6Impl(const YOLOv6Impl &) = delete; //
    YOLOv6Impl(YOLOv6Impl &&) = delete; //
    YOLOv6Impl &operator=(const YOLOv6Impl &) = delete; //
    YOLOv6Impl &operator=(YOLOv6Impl &&) = delete; //

  private:
    void print_debug_string();

    void transform(const cv::Mat &mat_rs, ncnn::Mat &in);

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        YOLOv6ScaleParams &scale_params);

    void generate_anchors(int target_height, int target_width,
                          std::vector<int> &strides,
                          std::vector<YOLOv6Anchor> &anchors);

    void generate_bboxes(const YOLOv6ScaleParams &scale_params,
                         std::vector<types::Boxf> &bbox_collection,
                         ncnn::Extractor &extractor,
                         float score_threshold, int img_height,
                         int img_width); // rescale & exclude

    void nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
             float iou_threshold, unsigned int topk, unsigned int nms_type);

  public:
    void detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                float score_threshold = 0.25f, float iou_threshold = 0.45f,
                unsigned int topk = 100, unsigned int nms_type = NMS::OFFSET);
  };

  // creator
  YOLOV6_EXPORTS static std::unique_ptr<YOLOv6Impl> create(
      const std::string &_param_path, const std::string &_bin_path,
      unsigned int _num_threads = 1, int _input_height = 640,
      int _input_width = 640)
  {
    return std::unique_ptr<YOLOv6Impl>(new YOLOv6Impl(
        _param_path, _bin_path, _num_threads,
        _input_height, _input_width));
  }
}

#endif //YOLOV6_YOLOV6_IMPL_H