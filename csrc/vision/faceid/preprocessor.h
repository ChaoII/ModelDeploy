//
// Created by aichao on 2025/3/24.
//

#pragma once
#include "csrc/core/md_decl.h"
#include "csrc/vision/common/result.h"
#include "csrc/core/md_tensor.h"




namespace modeldeploy::vision::faceid {
/*! @brief Preprocessor object for AdaFace serials model.
 */
class MODELDEPLOY_CXX_EXPORT AdaFacePreprocessor {
 public:
  /** \brief Create a preprocessor instance for AdaFace serials model
   */
  AdaFacePreprocessor();

  /** \brief Process the input image and prepare input tensors for runtime
   *
   * \param[in] images The input image data list, all the elements are returned by cv::imread()
   * \param[in] outputs The output tensors which will feed in runtime
   * \return true if the preprocess successed, otherwise false
   */
  bool Run(std::vector<cv::Mat>* images, std::vector<MDTensor>* outputs);

  /// Get Size
  std::vector<int> GetSize() { return size_; }

  /// Set size.
  void SetSize(std::vector<int>& size) { size_ = size; }

  /// Get alpha
  std::vector<float> GetAlpha() { return alpha_; }

  /// Set alpha.
  void SetAlpha(std::vector<float>& alpha) { alpha_ = alpha; }

  /// Get beta
  std::vector<float> GetBeta() { return beta_; }

  /// Set beta.
  void SetBeta(std::vector<float>& beta) { beta_ = beta; }

  bool GetPermute() { return permute_; }

  /// Set permute.
  void SetPermute(bool permute) { permute_ = permute; }

 protected:
  bool Preprocess(cv::Mat* mat, MDTensor* output);
  // Argument for image preprocessing step, tuple of (width, height),
  // decide the target size after resize, default (112, 112)
  std::vector<int> size_;
  // Argument for image preprocessing step, alpha values for normalization,
  // default alpha = {1.f / 127.5f, 1.f / 127.5f, 1.f / 127.5f};
  std::vector<float> alpha_;
  // Argument for image preprocessing step, beta values for normalization,
  // default beta = {-1.f, -1.f, -1.f}
  std::vector<float> beta_;
  // Argument for image preprocessing step, whether to swap the B and R channel,
  // such as BGR->RGB, default true.
  bool permute_;
};

} // namespace modeldeploy::vision::faceid
