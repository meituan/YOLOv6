//
// Created by DefTruth on 2022/7/3.
//

#include "utils.h"

// String Utils
std::string yolov6::utils::to_string(const std::wstring &wstr)
{
  unsigned len = wstr.size() * 4;
  setlocale(LC_CTYPE, "");
  char *p = new char[len];
  wcstombs(p, wstr.c_str(), len);
  std::string str(p);
  delete[] p;
  return str;
}

std::wstring yolov6::utils::to_wstring(const std::string &str)
{
  unsigned len = str.size() * 2;
  setlocale(LC_CTYPE, "");
  wchar_t *p = new wchar_t[len];
  mbstowcs(p, str.c_str(), len);
  std::wstring wstr(p);
  delete[] p;
  return wstr;
}

// Drawing Utils
cv::Mat yolov6::utils::draw_boxes(const cv::Mat &mat, const std::vector<types::Boxf> &boxes)
{
  if (boxes.empty()) return mat;
  cv::Mat mat_copy = mat.clone();
  for (const auto &box: boxes)
  {
    if (box.flag)
    {
      cv::rectangle(mat_copy, box.rect(), cv::Scalar(255, 255, 0), 2);
      if (box.label_text)
      {
        std::string label_text(box.label_text);
        label_text = label_text + ":" + std::to_string(box.score).substr(0, 4);
        cv::putText(mat_copy, label_text, box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                    0.6f, cv::Scalar(0, 255, 0), 2);

      }
    }
  }
  return mat_copy;
}

void yolov6::utils::draw_boxes_inplace(cv::Mat &mat_inplace, const std::vector<types::Boxf> &boxes)
{
  if (boxes.empty()) return;
  for (const auto &box: boxes)
  {
    if (box.flag)
    {
      cv::rectangle(mat_inplace, box.rect(), cv::Scalar(255, 255, 0), 2);
      if (box.label_text)
      {
        std::string label_text(box.label_text);
        label_text = label_text + ":" + std::to_string(box.score).substr(0, 4);
        cv::putText(mat_inplace, label_text, box.tl(), cv::FONT_HERSHEY_SIMPLEX,
                    0.6f, cv::Scalar(0, 255, 0), 2);

      }
    }
  }
}

// Object Detection Utils
void yolov6::utils::hard_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                             float iou_threshold, unsigned int topk)
{
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const types::Boxf &a, const types::Boxf &b)
            { return a.score > b.score; });
  const unsigned int box_num = input.size();
  std::vector<int> merged(box_num, 0);

  unsigned int count = 0;
  for (unsigned int i = 0; i < box_num; ++i)
  {
    if (merged[i]) continue;
    std::vector<types::Boxf> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    for (unsigned int j = i + 1; j < box_num; ++j)
    {
      if (merged[j]) continue;

      float iou = static_cast<float>(input[i].iou_of(input[j]));

      if (iou > iou_threshold)
      {
        merged[j] = 1;
        buf.push_back(input[j]);
      }

    }
    output.push_back(buf[0]);

    // keep top k
    count += 1;
    if (count >= topk)
      break;
  }
}

void yolov6::utils::blending_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                                 float iou_threshold, unsigned int topk)
{
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const types::Boxf &a, const types::Boxf &b)
            { return a.score > b.score; });
  const unsigned int box_num = input.size();
  std::vector<int> merged(box_num, 0);

  unsigned int count = 0;
  for (unsigned int i = 0; i < box_num; ++i)
  {
    if (merged[i]) continue;
    std::vector<types::Boxf> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    for (unsigned int j = i + 1; j < box_num; ++j)
    {
      if (merged[j]) continue;

      float iou = static_cast<float>(input[i].iou_of(input[j]));
      if (iou > iou_threshold)
      {
        merged[j] = 1;
        buf.push_back(input[j]);
      }
    }

    float total = 0.f;
    for (unsigned int k = 0; k < buf.size(); ++k)
    {
      total += std::exp(buf[k].score);
    }
    types::Boxf rects;
    for (unsigned int l = 0; l < buf.size(); ++l)
    {
      float rate = std::exp(buf[l].score) / total;
      rects.x1 += buf[l].x1 * rate;
      rects.y1 += buf[l].y1 * rate;
      rects.x2 += buf[l].x2 * rate;
      rects.y2 += buf[l].y2 * rate;
      rects.score += buf[l].score * rate;
    }
    rects.flag = true;
    output.push_back(rects);

    // keep top k
    count += 1;
    if (count >= topk)
      break;
  }
}

// reference: https://github.com/ultralytics/yolov5/blob/master/utils/general.py
void yolov6::utils::offset_nms(std::vector<types::Boxf> &input, std::vector<types::Boxf> &output,
                               float iou_threshold, unsigned int topk)
{
  if (input.empty()) return;
  std::sort(input.begin(), input.end(),
            [](const types::Boxf &a, const types::Boxf &b)
            { return a.score > b.score; });
  const unsigned int box_num = input.size();
  std::vector<int> merged(box_num, 0);

  const float offset = 4096.f;
  /** Add offset according to classes.
   * That is, separate the boxes into categories, and each category performs its
   * own NMS operation. The same offset will be used for those predicted to be of
   * the same category. Therefore, the relative positions of boxes of the same
   * category will remain unchanged. Box of different classes will be farther away
   * after offset, because offsets are different. In this way, some overlapping but
   * different categories of entities are not filtered out by the NMS. Very clever!
   */
  for (unsigned int i = 0; i < box_num; ++i)
  {
    input[i].x1 += static_cast<float>(input[i].label) * offset;
    input[i].y1 += static_cast<float>(input[i].label) * offset;
    input[i].x2 += static_cast<float>(input[i].label) * offset;
    input[i].y2 += static_cast<float>(input[i].label) * offset;
  }

  unsigned int count = 0;
  for (unsigned int i = 0; i < box_num; ++i)
  {
    if (merged[i]) continue;
    std::vector<types::Boxf> buf;

    buf.push_back(input[i]);
    merged[i] = 1;

    for (unsigned int j = i + 1; j < box_num; ++j)
    {
      if (merged[j]) continue;

      float iou = static_cast<float>(input[i].iou_of(input[j]));

      if (iou > iou_threshold)
      {
        merged[j] = 1;
        buf.push_back(input[j]);
      }

    }
    output.push_back(buf[0]);

    // keep top k
    count += 1;
    if (count >= topk)
      break;
  }

  /** Substract offset.*/
  if (!output.empty())
  {
    for (unsigned int i = 0; i < output.size(); ++i)
    {
      output[i].x1 -= static_cast<float>(output[i].label) * offset;
      output[i].y1 -= static_cast<float>(output[i].label) * offset;
      output[i].x2 -= static_cast<float>(output[i].label) * offset;
      output[i].y2 -= static_cast<float>(output[i].label) * offset;
    }
  }
}

// ONNXRuntime Utils
Ort::Value yolov6::utils::transform::create_tensor(const cv::Mat &mat,
                                                   const std::vector<int64_t> &tensor_dims,
                                                   const Ort::MemoryInfo &memory_info_handler,
                                                   std::vector<float> &tensor_value_handler,
                                                   unsigned int data_format)
throw(std::runtime_error)
{
  const unsigned int rows = mat.rows;
  const unsigned int cols = mat.cols;
  const unsigned int channels = mat.channels();

  cv::Mat mat_ref;
  if (mat.type() != CV_32FC(channels)) mat.convertTo(mat_ref, CV_32FC(channels));
  else mat_ref = mat; // reference only. zero-time cost. support 1/2/3/... channels

  if (tensor_dims.size() != 4) throw std::runtime_error("dims mismatch.");
  if (tensor_dims.at(0) != 1) throw std::runtime_error("batch != 1");

  // CXHXW
  if (data_format == transform::CHW)
  {

    const unsigned int target_height = tensor_dims.at(2);
    const unsigned int target_width = tensor_dims.at(3);
    const unsigned int target_channel = tensor_dims.at(1);
    const unsigned int target_tensor_size = target_channel * target_height * target_width;
    if (target_channel != channels) throw std::runtime_error("channel mismatch.");

    tensor_value_handler.resize(target_tensor_size);

    cv::Mat resize_mat_ref;
    if (target_height != rows || target_width != cols)
      cv::resize(mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
    else resize_mat_ref = mat_ref; // reference only. zero-time cost.

    std::vector<cv::Mat> mat_channels;
    cv::split(resize_mat_ref, mat_channels);
    // CXHXW
    for (unsigned int i = 0; i < channels; ++i)
      std::memcpy(tensor_value_handler.data() + i * (target_height * target_width),
                  mat_channels.at(i).data, target_height * target_width * sizeof(float));

    return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
                                           target_tensor_size, tensor_dims.data(),
                                           tensor_dims.size());
  }

  // HXWXC
  const unsigned int target_height = tensor_dims.at(1);
  const unsigned int target_width = tensor_dims.at(2);
  const unsigned int target_channel = tensor_dims.at(3);
  const unsigned int target_tensor_size = target_channel * target_height * target_width;
  if (target_channel != channels) throw std::runtime_error("channel mismatch!");
  tensor_value_handler.resize(target_tensor_size);

  cv::Mat resize_mat_ref;
  if (target_height != rows || target_width != cols)
    cv::resize(mat_ref, resize_mat_ref, cv::Size(target_width, target_height));
  else resize_mat_ref = mat_ref; // reference only. zero-time cost.

  std::memcpy(tensor_value_handler.data(), resize_mat_ref.data, target_tensor_size * sizeof(float));

  return Ort::Value::CreateTensor<float>(memory_info_handler, tensor_value_handler.data(),
                                         target_tensor_size, tensor_dims.data(),
                                         tensor_dims.size());
}

cv::Mat yolov6::utils::transform::normalize(const cv::Mat &mat, float mean, float scale)
{
  cv::Mat matf;
  if (mat.type() != CV_32FC3) mat.convertTo(matf, CV_32FC3);
  else matf = mat; // reference
  return (matf - mean) * scale;
}

cv::Mat yolov6::utils::transform::normalize(const cv::Mat &mat, const float *mean, const float *scale)
{
  cv::Mat mat_copy;
  if (mat.type() != CV_32FC3) mat.convertTo(mat_copy, CV_32FC3);
  else mat_copy = mat.clone();
  for (unsigned int i = 0; i < mat_copy.rows; ++i)
  {
    cv::Vec3f *p = mat_copy.ptr<cv::Vec3f>(i);
    for (unsigned int j = 0; j < mat_copy.cols; ++j)
    {
      p[j][0] = (p[j][0] - mean[0]) * scale[0];
      p[j][1] = (p[j][1] - mean[1]) * scale[1];
      p[j][2] = (p[j][2] - mean[2]) * scale[2];
    }
  }
  return mat_copy;
}

void yolov6::utils::transform::normalize(const cv::Mat &inmat, cv::Mat &outmat,
                                         float mean, float scale)
{
  outmat = yolov6::utils::transform::normalize(inmat, mean, scale);
}

void yolov6::utils::transform::normalize_inplace(cv::Mat &mat_inplace, float mean, float scale)
{
  if (mat_inplace.type() != CV_32FC3) mat_inplace.convertTo(mat_inplace, CV_32FC3);
  yolov6::utils::transform::normalize(mat_inplace, mat_inplace, mean, scale);
}

void yolov6::utils::transform::normalize_inplace(cv::Mat &mat_inplace, const float *mean, const float *scale)
{
  if (mat_inplace.type() != CV_32FC3) mat_inplace.convertTo(mat_inplace, CV_32FC3);
  for (unsigned int i = 0; i < mat_inplace.rows; ++i)
  {
    cv::Vec3f *p = mat_inplace.ptr<cv::Vec3f>(i);
    for (unsigned int j = 0; j < mat_inplace.cols; ++j)
    {
      p[j][0] = (p[j][0] - mean[0]) * scale[0];
      p[j][1] = (p[j][1] - mean[1]) * scale[1];
      p[j][2] = (p[j][2] - mean[2]) * scale[2];
    }
  }
}

