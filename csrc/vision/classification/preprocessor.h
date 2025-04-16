//
// Created by aichao on 2025/2/24.
//

#pragma once

#include "csrc/vision/common/result.h"
#include <map>
#include "csrc/vision/utils.h"
#include "csrc/core/md_decl.h"
#include "csrc/core/tensor.h"

namespace modeldeploy {
namespace vision {

namespace classification {
/*! @brief Preprocessor object for YOLOv5Cls serials model.
 */
class MODELDEPLOY_CXX_EXPORT YOLOv5ClsPreprocessor {
 public:
  /** \brief Create a preprocessor instance for YOLOv5Cls serials model
   */
  YOLOv5ClsPreprocessor();

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input image data list, all the elements are returned by cv::imread()
   * \param[in] outputs The output tensors which will feed in runtime
   * \param[in] ims_info The shape info list, record input_shape and output_shape
   * \return true if the preprocess successed, otherwise false
   */
  bool Run(std::vector<cv::Mat>* images, std::vector<Tensor>* outputs,
           std::vector<std::map<std::string, std::array<float, 2>>>* ims_info);

  /// Set target size, tuple of (width, height), default size = {224, 224}
  void SetSize(const std::vector<int>& size) { size_ = size; }

  /// Get target size, tuple of (width, height), default size = {224, 224}
  std::vector<int> GetSize() const { return size_; }

 protected:
  bool Preprocess(cv::Mat* mat, Tensor* output,
                  std::map<std::string, std::array<float, 2>>* im_info);

  // target size, tuple of (width, height), default size = {224, 224}
  std::vector<int> size_;
};

}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
