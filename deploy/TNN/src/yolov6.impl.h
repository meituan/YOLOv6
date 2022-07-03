//
// Created by DefTruth on 2022/7/3.
//

#ifndef YOLOV6_YOLOV6_IMPL_H
#define YOLOV6_YOLOV6_IMPL_H

#include "types.h"
#include "handler.h"

namespace yolov6
{
  class YOLOV6_EXPORTS YOLOv6Impl : public BasicTNNHandler
  {
  public:
    explicit YOLOv6Impl(const std::string &_proto_path,
                        const std::string &_model_path,
                        unsigned int _num_threads = 1); //
    ~YOLOv6Impl() override = default;

  private:
    // nested classes
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
    // In TNN: x*scale + bias
    std::vector<float> scale_vals = {1.0 / 255.f, 1.0 / 255.f, 1.0 / 255.f}; // RGB
    std::vector<float> bias_vals = {0.f, 0.f, 0.f};
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
    static constexpr const unsigned int max_nms = 30000;

  private:
    void transform(const cv::Mat &mat_rs) override; // without resize

    void resize_unscale(const cv::Mat &mat,
                        cv::Mat &mat_rs,
                        int target_height,
                        int target_width,
                        YOLOv6ScaleParams &scale_params);

    void generate_bboxes(const YOLOv6ScaleParams &scale_params,
                         std::vector<types::Boxf> &bbox_collection,
                         std::shared_ptr<tnn::Instance> &_instance,
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
      const std::string &_proto_path, const std::string &_model_path,
      unsigned int _num_threads = 1)
  { return std::unique_ptr<YOLOv6Impl>(new YOLOv6Impl(
      _proto_path, _model_path, _num_threads)); }
}

#endif //YOLOV6_YOLOV6_IMPL_H