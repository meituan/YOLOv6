//
// Created by DefTruth on 2022/7/3.
//

#include "yolov6.impl.h"
#include "utils.h"

using yolov6::YOLOv6Impl;

YOLOv6Impl::YOLOv6Impl(const std::string &_proto_path,
                       const std::string &_model_path,
                       unsigned int _num_threads) :
    BasicTNNHandler(_proto_path, _model_path, _num_threads)
{
}

// letterbox
void YOLOv6Impl::resize_unscale(const cv::Mat &mat, cv::Mat &mat_rs,
                                int target_height, int target_width,
                                YOLOv6ScaleParams &scale_params)
{
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);

  mat_rs = cv::Mat(target_height, target_width, CV_8UC3,
                   cv::Scalar(114, 114, 114));
  // scale ratio (new / old) new_shape(h,w)
  float w_r = (float) target_width / (float) img_width;
  float h_r = (float) target_height / (float) img_height;
  float r = std::min(w_r, h_r);
  // compute padding
  int new_unpad_w = static_cast<int>((float) img_width * r); // floor
  int new_unpad_h = static_cast<int>((float) img_height * r); // floor
  int pad_w = target_width - new_unpad_w; // >=0
  int pad_h = target_height - new_unpad_h; // >=0

  int dw = pad_w / 2;
  int dh = pad_h / 2;

  // resize with unscaling
  cv::Mat new_unpad_mat;
  // cv::Mat new_unpad_mat = mat.clone(); // may not need clone.
  cv::resize(mat, new_unpad_mat, cv::Size(new_unpad_w, new_unpad_h));
  new_unpad_mat.copyTo(mat_rs(cv::Rect(dw, dh, new_unpad_w, new_unpad_h)));

  // record scale params.
  scale_params.r = r;
  scale_params.dw = dw;
  scale_params.dh = dh;
  scale_params.new_unpad_w = new_unpad_w;
  scale_params.new_unpad_h = new_unpad_h;
  scale_params.flag = true;
}

void YOLOv6Impl::transform(const cv::Mat &mat_rs)
{
  // push into input_mat
  // be carefully, no deepcopy inside this tnn::Mat constructor,
  // so, we can not pass a local cv::Mat to this constructor.
  input_mat = std::make_shared<tnn::Mat>(input_device_type, tnn::N8UC3,
                                         input_shape, (void *) mat_rs.data);
  if (!input_mat->GetData())
  {
#if defined(YOLOV6_DEBUG)
    std::cout << "input_mat == nullptr! transform failed\n";
#endif
  }
}

void YOLOv6Impl::detect(const cv::Mat &mat, std::vector<types::Boxf> &detected_boxes,
                        float score_threshold, float iou_threshold,
                        unsigned int topk, unsigned int nms_type)
{
  if (mat.empty()) return;
  int img_height = static_cast<int>(mat.rows);
  int img_width = static_cast<int>(mat.cols);
  // resize & unscale
  cv::Mat mat_rs;
  YOLOv6ScaleParams scale_params;
  this->resize_unscale(mat, mat_rs, input_height, input_width, scale_params);

  // 1. make input tensor
  cv::Mat mat_rs_;
  cv::cvtColor(mat_rs, mat_rs_, cv::COLOR_BGR2RGB);
  this->transform(mat_rs_);
  // 2. set input_mat
  tnn::MatConvertParam input_cvt_param;
  input_cvt_param.scale = scale_vals;
  input_cvt_param.bias = bias_vals;

  tnn::Status status;
  status = instance->SetInputMat(input_mat, input_cvt_param);
  if (status != tnn::TNN_OK)
  {
#if defined(YOLOV6_DEBUG)
    std::cout << "instance->SetInputMat failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }

  // 3. forward
  status = instance->Forward();
  if (status != tnn::TNN_OK)
  {
#if defined(YOLOV6_DEBUG)
    std::cout << "instance->Forward failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }
  // 5. rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(scale_params, bbox_collection, instance, score_threshold, img_height, img_width);
  // 6. hard|blend|offset nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void YOLOv6Impl::generate_bboxes(const YOLOv6ScaleParams &scale_params,
                                 std::vector<types::Boxf> &bbox_collection,
                                 std::shared_ptr<tnn::Instance> &_instance,
                                 float score_threshold, int img_height,
                                 int img_width)
{
  // 4. fetch output mat
  std::shared_ptr<tnn::Mat> pred_mat;
  tnn::MatConvertParam pred_cvt_param; // default
  tnn::Status status;

  // (1,n,85=5+80=cxcy+cwch+obj_conf+cls_conf)
  status = _instance->GetOutputMat(pred_mat, pred_cvt_param, "outputs", output_device_type);
  if (status != tnn::TNN_OK)
  {
#if defined(YOLOV6_DEBUG)
    std::cout << "instance->GetOutputMat failed!:"
              << status.description().c_str() << "\n";
#endif
    return;
  }

  auto pred_dims = pred_mat->GetDims();
  const unsigned int num_anchors = pred_dims.at(1); // n = ?
  const unsigned int num_classes = pred_dims.at(2) - 5; // 80

  float r_ = scale_params.r;
  int dw_ = scale_params.dw;
  int dh_ = scale_params.dh;

  bbox_collection.clear();
  unsigned int count = 0;
  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    const float *offset_obj_cls_ptr =
        (float *) pred_mat->GetData() + (i * (num_classes + 5)); // row ptr
    float obj_conf = offset_obj_cls_ptr[4];
    if (obj_conf < score_threshold) continue; // filter first.

    float cls_conf = offset_obj_cls_ptr[5];
    unsigned int label = 0;
    for (unsigned int j = 0; j < num_classes; ++j)
    {
      float tmp_conf = offset_obj_cls_ptr[j + 5];
      if (tmp_conf > cls_conf)
      {
        cls_conf = tmp_conf;
        label = j;
      }
    } // argmax

    float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
    if (conf < score_threshold) continue; // filter

    float cx = offset_obj_cls_ptr[0];
    float cy = offset_obj_cls_ptr[1];
    float w = offset_obj_cls_ptr[2];
    float h = offset_obj_cls_ptr[3];
    float x1 = ((cx - w / 2.f) - (float) dw_) / r_;
    float y1 = ((cy - h / 2.f) - (float) dh_) / r_;
    float x2 = ((cx + w / 2.f) - (float) dw_) / r_;
    float y2 = ((cy + h / 2.f) - (float) dh_) / r_;

    types::Boxf box;
    box.x1 = std::max(0.f, x1);
    box.y1 = std::max(0.f, y1);
    box.x2 = std::min(x2, (float) img_width - 1.f);
    box.y2 = std::min(y2, (float) img_height - 1.f);
    box.score = conf;
    box.label = label;
    box.label_text = class_names[label];
    box.flag = true;
    bbox_collection.push_back(box);

    count += 1; // limit boxes for nms.
    if (count > max_nms)
      break;
  }
#if defined(YOLOV6_DEBUG)
  std::cout << "detected num_anchors: " << num_anchors << "\n";
  std::cout << "generate_bboxes num: " << bbox_collection.size() << "\n";
#endif
}

void YOLOv6Impl::nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                     float iou_threshold, unsigned int topk,
                     unsigned int nms_type)
{
  if (nms_type == NMS::BLEND) yolov6::utils::blending_nms(input, output, iou_threshold, topk);
  else if (nms_type == NMS::OFFSET) yolov6::utils::offset_nms(input, output, iou_threshold, topk);
  else yolov6::utils::hard_nms(input, output, iou_threshold, topk);
}

