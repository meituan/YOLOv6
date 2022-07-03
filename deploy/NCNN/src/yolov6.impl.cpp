//
// Created by DefTruth on 2022/7/3.
//

#include "yolov6.impl.h"
#include "utils.h"

using yolov6::YOLOv6Impl;

YOLOv6Impl::YOLOv6Impl(const std::string &_param_path,
                       const std::string &_bin_path,
                       unsigned int _num_threads,
                       int _input_height,
                       int _input_width) :
    log_id(_param_path.data()), param_path(_param_path.data()),
    bin_path(_bin_path.data()), num_threads(_num_threads),
    input_height(_input_height), input_width(_input_width)
{
  net = new ncnn::Net();
  // init net, change this setting for better performance.
  net->opt.use_fp16_arithmetic = false;
  net->opt.use_vulkan_compute = false; // default
  net->load_param(param_path);
  net->load_model(bin_path);
#if defined(YOLOV6_DEBUG)
  this->print_debug_string();
#endif
}

YOLOv6Impl::~YOLOv6Impl()
{
  if (net) delete net;
  net = nullptr;
#if defined(YOLOV6_DEBUG)
  std::cout << "[MEM DELETE] YOLOv6Impl done!" << "\n";
#endif
}

void YOLOv6Impl::transform(const cv::Mat &mat_rs, ncnn::Mat &in)
{
  // BGR NHWC -> RGB NCHW
  in = ncnn::Mat::from_pixels(mat_rs.data, ncnn::Mat::PIXEL_BGR2RGB, input_width, input_height);
  in.substract_mean_normalize(mean_vals, norm_vals);
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
  ncnn::Mat input;
  this->transform(mat_rs, input);
  // 2. inference & extract
  auto extractor = net->create_extractor();
  extractor.set_light_mode(false);  // default
  extractor.set_num_threads(num_threads);
  extractor.input("image_arrays", input);
  // 3.rescale & exclude.
  std::vector<types::Boxf> bbox_collection;
  this->generate_bboxes(scale_params, bbox_collection, extractor, score_threshold, img_height, img_width);
  // 4. hard|blend|offset nms with topk.
  this->nms(bbox_collection, detected_boxes, iou_threshold, topk, nms_type);
}

void YOLOv6Impl::generate_anchors(int target_height, int target_width,
                                  std::vector<int> &strides,
                                  std::vector<YOLOv6Anchor> &anchors)
{
  for (auto stride: strides)
  {
    int num_grid_w = target_width / stride;
    int num_grid_h = target_height / stride;
    for (int g1 = 0; g1 < num_grid_h; ++g1)
    {
      for (int g0 = 0; g0 < num_grid_w; ++g0)
      {
        YOLOv6Anchor anchor;
        anchor.grid0 = g0;
        anchor.grid1 = g1;
        anchor.stride = stride;
        anchors.push_back(anchor);
      }
    }
  }
}

static inline float sigmoid(float x)
{ return static_cast<float>(1.f / (1.f + std::exp(-x))); }

void YOLOv6Impl::generate_bboxes(const YOLOv6ScaleParams &scale_params,
                                 std::vector<types::Boxf> &bbox_collection,
                                 ncnn::Extractor &extractor,
                                 float score_threshold, int img_height,
                                 int img_width)
{
  ncnn::Mat outputs;
  extractor.extract("outputs", outputs); // (1,n=?,85=5+80=cxcy+cwch+obj_conf+cls_conf)

#if defined(YOLOV6_DEBUG)
  BasicNCNNHandler::print_shape(outputs, "outputs");
#endif

  const unsigned int num_anchors = outputs.h;
  const unsigned int num_classes = outputs.w - 5;

  std::vector<YOLOv6Anchor> anchors;
  std::vector<int> strides = {8, 16, 32}; // might have stride=64
  this->generate_anchors(input_height, input_width, strides, anchors);

  float r_ = scale_params.r;
  int dw_ = scale_params.dw;
  int dh_ = scale_params.dh;

  bbox_collection.clear();
  unsigned int count = 0;

  for (unsigned int i = 0; i < num_anchors; ++i)
  {
    const float *offset_obj_cls_ptr =
        (float *) outputs.data + (i * (num_classes + 5)); // row ptr
    float obj_conf = sigmoid(offset_obj_cls_ptr[4]);
    if (obj_conf < score_threshold) continue; // filter first.

    float cls_conf = sigmoid(offset_obj_cls_ptr[5]);
    unsigned int label = 0;
    for (unsigned int j = 0; j < num_classes; ++j)
    {
      float tmp_conf = sigmoid(offset_obj_cls_ptr[j + 5]);
      if (tmp_conf > cls_conf)
      {
        cls_conf = tmp_conf;
        label = j;
      }
    } // argmax

    float conf = obj_conf * cls_conf; // cls_conf (0.,1.)
    if (conf < score_threshold) continue; // filter

    const int grid0 = anchors.at(i).grid0;
    const int grid1 = anchors.at(i).grid1;
    const int stride = anchors.at(i).stride;

    float dx = offset_obj_cls_ptr[0];
    float dy = offset_obj_cls_ptr[1];
    float dw = offset_obj_cls_ptr[2];
    float dh = offset_obj_cls_ptr[3];

    float cx = (dx + (float) grid0) * (float) stride;
    float cy = (dy + (float) grid1) * (float) stride;
    float w = std::exp(dw) * (float) stride;
    float h = std::exp(dh) * (float) stride;
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


void YOLOv6Impl::print_debug_string()
{
  std::cout << "YOLOV6_DEBUG LogId: " << log_id << "\n";
  input_indexes = net->input_indexes();
  output_indexes = net->output_indexes();
#ifdef NCNN_STRING
  input_names = net->input_names();
  output_names = net->output_names();
#endif
  std::cout << "=============== Input-Dims ==============\n";
  for (int i = 0; i < input_indexes.size(); ++i)
  {
    std::cout << "Input: ";
    auto tmp_in_blob = net->blobs().at(input_indexes.at(i));
#ifdef NCNN_STRING
    std::cout << input_names.at(i) << ": ";
#endif
    std::cout << "shape: c=" << tmp_in_blob.shape.c
              << " h=" << tmp_in_blob.shape.h << " w=" << tmp_in_blob.shape.w << "\n";
  }

  std::cout << "=============== Output-Dims ==============\n";
  for (int i = 0; i < output_indexes.size(); ++i)
  {
    auto tmp_out_blob = net->blobs().at(output_indexes.at(i));
    std::cout << "Output: ";
#ifdef NCNN_STRING
    std::cout << output_names.at(i) << ": ";
#endif
    std::cout << "shape: c=" << tmp_out_blob.shape.c
              << " h=" << tmp_out_blob.shape.h << " w=" << tmp_out_blob.shape.w << "\n";
  }
  std::cout << "========================================\n";
}


