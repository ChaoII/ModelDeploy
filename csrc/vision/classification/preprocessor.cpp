//
// Created by aichao on 2025/2/24.
//

#include "preprocessor.h"
#include "csrc/function/concat.h"
#include "../common/processors/center_crop.h"
#include "../common/processors/resize.h"
#include "../common/processors/convert.h"
#include "../common/processors/color_space_convert.h"
#include "../common/processors/normalize_and_permute.h"

namespace modeldeploy {
namespace vision {
namespace classification {

YOLOv5ClsPreprocessor::YOLOv5ClsPreprocessor() {
  size_ = {224, 224};  //{h,w}
}

bool YOLOv5ClsPreprocessor::Preprocess(
    cv::Mat* mat, MDTensor* output,
    std::map<std::string, std::array<float, 2>>* im_info) {
  // Record the shape of image and the shape of preprocessed image
  (*im_info)["input_shape"] = {static_cast<float>(mat->rows),
                               static_cast<float>(mat->cols)};

  // process after image load
  double ratio = (size_[0] * 1.0) / std::max(static_cast<float>(mat->rows),
                                             static_cast<float>(mat->cols));

  // yolov5cls's preprocess steps
  // 1. CenterCrop
  // 2. Normalize
  // CenterCrop
  int crop_size = std::min(mat->rows, mat->cols);
  CenterCrop::Run(mat, crop_size, crop_size);
  Resize::Run(mat, size_[0], size_[1], -1, -1, cv::INTER_LINEAR);
  // Normalize
  BGR2RGB::Run(mat);
  std::vector<float> alpha = {1.0f / 255.0f, 1.0f / 255.0f, 1.0f / 255.0f};
  std::vector<float> beta = {0.0f, 0.0f, 0.0f};
  Convert::Run(mat, alpha, beta);
  std::vector<float> mean = {0.485f, 0.456f, 0.406f};
  std::vector<float> std = {0.229f, 0.224f, 0.225f};
  NormalizeAndPermute::Run(mat, mean, std, false);

  // Record output shape of preprocessed image
  (*im_info)["output_shape"] = {static_cast<float>(mat->rows),
                                static_cast<float>(mat->cols)};

  mat_to_tensor(*mat, output);
  output->expand_dim(0); // reshape to n, c, h, w
  return true;
}

bool YOLOv5ClsPreprocessor::Run(
    std::vector<cv::Mat>* images, std::vector<MDTensor>* outputs,
    std::vector<std::map<std::string, std::array<float, 2>>>* ims_info) {
  if (images->size() == 0) {
    std::cerr << "The size of input images should be greater than 0."
            << std::endl;
    return false;
  }
  ims_info->resize(images->size());
  outputs->resize(1);
  // Concat all the preprocessed data to a batch tensor
  std::vector<MDTensor> tensors(images->size());
  for (size_t i = 0; i < images->size(); ++i) {
    if (!Preprocess(&(*images)[i], &tensors[i], &(*ims_info)[i])) {
      std::cerr << "Failed to preprocess input image." << std::endl;
      return false;
    }
  }

  if (tensors.size() == 1) {
    (*outputs)[0] = std::move(tensors[0]);
  } else {
    function::Concat(tensors, &((*outputs)[0]), 0);
  }
  return true;
}

}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
