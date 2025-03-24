//
// Created by aichao on 2025/3/24.
//

#pragma once
#include "csrc/core/md_tensor.h"
#include "csrc/vision/common/result.h"




namespace modeldeploy::vision::faceid {
/*! @brief Postprocessor object for AdaFace serials model.
 */
class MODELDEPLOY_CXX_EXPORT AdaFacePostprocessor {
 public:
  /** \brief Create a postprocessor instance for AdaFace serials model
   */
  AdaFacePostprocessor();

  /** \brief Process the result of runtime and fill to FaceRecognitionResult structure
   *
   * \param[in] tensors The inference result from runtime
   * \param[in] result The output result of FaceRecognitionResult
   * \return true if the postprocess successed, otherwise false
   */
  bool Run(std::vector<MDTensor>& infer_result,std::vector<FaceRecognitionResult>* results);

  void SetL2Normalize(bool& l2_normalize) { l2_normalize_ = l2_normalize; }

  bool GetL2Normalize() { return l2_normalize_; }

 private:
  bool l2_normalize_;
};

} // namespace modeldeploy::vision::faceid

