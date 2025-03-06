//
// Created by aichao on 2025/2/24.
//
#pragma once

#include "csrc/vision/common/result.h"
#include <map>
#include "csrc/core/md_tensor.h"
namespace modeldeploy {
namespace vision {

namespace classification {
/*! @brief Postprocessor object for YOLOv5Cls serials model.
 */
class  YOLOv5ClsPostprocessor {
 public:
  /** \brief Create a postprocessor instance for YOLOv5Cls serials model
   */
  YOLOv5ClsPostprocessor();

  /** \brief Process the result of runtime and fill to ClassifyResult structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of classification
   * \param[in] ims_info The shape info list, record input_shape and output_shape
   * \return true if the postprocess successed, otherwise false
   */
  bool Run(const std::vector<MDTensor>& tensors,
      std::vector<ClassifyResult>* results,
      const std::vector<std::map<std::string, std::array<float, 2>>>& ims_info);

  /// Set topk, default 1
  void SetTopK(const int& topk) {
    topk_ = topk;
  }

  /// Get topk, default 1
  float GetTopK() const { return topk_; }

 protected:
  int topk_;
};

}  // namespace classification
}  // namespace vision
}  // namespace fastdeploy
